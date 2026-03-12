#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from src.utils.gaussian_model_utils import (RGB2SH, build_scaling_rotation,
                                            get_expon_lr_func, inverse_sigmoid,
                                            strip_symmetric, BasicPointCloud)


class GaussianModel:
    def __init__(self, sh_degree: int = 3, isotropic=True):
        self.gaussian_param_names = [
            "active_sh_degree",
            "xyz",
            "features_dc",
            "features_rest",
            "rgb",
            "scaling",
            "rotation",
            "opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "spatial_lr_scale",
            "optimizer",
        ]
        self.max_sh_degree = sh_degree
        self.active_sh_degree = sh_degree  # temp
        self._xyz = torch.empty(0).cuda()
        self._features_dc = torch.empty(0).cuda()
        self._features_rest = torch.empty(0).cuda()
        self._scaling = torch.empty(0).cuda()
        self._rotation = torch.empty(0, 4).cuda()
        self._rgb = torch.empty(0, 3, device='cuda')  # (N, 3) for RGB
        self._opacity = torch.empty(0).cuda()
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()
        self.isotropic = isotropic

    def restore_from_params(self, params_dict, training_args):
        self.training_setup(training_args)
        self.densification_postfix(
            params_dict["xyz"],
            params_dict["rgb"],
            params_dict["features_dc"],
            params_dict["features_rest"],
            params_dict["opacity"],
            params_dict["scaling"],
            params_dict["rotation"])

    def build_covariance_from_scaling_rotation(self, scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    def capture_dict(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz.clone().detach().cpu(),
            "features_dc": self._features_dc.clone().detach().cpu(),
            "features_rest": self._features_rest.clone().detach().cpu(),
            "rgb": self._rgb.clone().detach().cpu(),
            "scaling": self._scaling.clone().detach().cpu(),
            "rotation": self._rotation.clone().detach().cpu(),
            "opacity": self._opacity.clone().detach().cpu(),
            "max_radii2D": self.max_radii2D.clone().detach().cpu(),
            "xyz_gradient_accum": self.xyz_gradient_accum.clone().detach().cpu(),
            "denom": self.denom.clone().detach().cpu(),
            "spatial_lr_scale": self.spatial_lr_scale,
            "optimizer": self.optimizer.state_dict(),
        }

    def get_size(self):
        return self._xyz.shape[0]

    def get_scaling(self):
        if self.isotropic:
            scale = self.scaling_activation(self._scaling)[:, 0:1]  # Extract the first column
            scales = scale.repeat(1, 2)  # Replicate this column three times
            return scales
        return self.scaling_activation(self._scaling)

    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    def get_xyz(self):
        return self._xyz

    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    def get_rgb(self):
        return self._rgb
    
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_active_sh_degree(self):
        return self.active_sh_degree

    def get_covariance(self, scaling_modifier=1):
        return self.build_covariance_from_scaling_rotation(self.get_scaling(), scaling_modifier, self._rotation)

    def add_points(self, pcd: o3d.geometry.PointCloud, global_scale_init=True):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        rgb_tensor = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        print("rgb_max: ", torch.max(rgb_tensor), "rgb_min: ", torch.min(rgb_tensor), "rgb_mean: ", torch.mean(rgb_tensor))
        fused_color = RGB2SH(torch.tensor(
            np.asarray(pcd.colors)).float().cuda())
        features = (torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda())
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of added points: ", fused_point_cloud.shape[0])

        if global_scale_init:
            global_points = torch.cat((self.get_xyz(),torch.from_numpy(np.asarray(pcd.points)).float().cuda()))
            dist2 = torch.clamp_min(distCUDA2(global_points), 0.0000001)
            dist2 = dist2[self.get_size():]
        else:
            dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scale_cols = 3 if not self.isotropic else 2
        scales = torch.log(1.0 * torch.sqrt(dist2))[..., None].repeat(1, scale_cols)
        # scales = torch.log(0.001 * torch.ones_like(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities =torch.zeros((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_rgb = nn.Parameter(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        new_features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        new_features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacities = nn.Parameter(opacities.requires_grad_(True))
        self.densification_postfix(
            new_xyz,
            new_rgb,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def training_setup(self, training_args, exposure_ab=None):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": training_args.position_lr_init, "name": "xyz"},
            {"params": [self._features_dc], "lr": training_args.feature_lr, "name": "f_dc"},
            {"params": [self._features_rest], "lr": training_args.feature_lr / 20.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": training_args.opacity_lr, "name": "opacity"},
            {"params": [self._scaling], "lr": training_args.scaling_lr, "name": "scaling"},
            {"params": [self._rotation], "lr": training_args.rotation_lr, "name": "rotation"},
            {"params": [self._rgb], "lr": training_args.rgb_lr, "name": "rgb",},
        ]

        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

    def training_setup_camera(self, cam_rot, cam_trans, cfg, exposure_ab=None):
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz().shape[0], 1), device="cuda"
        )
        self.denom = torch.zeros((self.get_xyz().shape[0], 1), device="cuda")

        params = [
            {"params": [self._xyz], "lr": 0.0, "name": "xyz"},
            {"params": [self._features_dc], "lr": 0.0, "name": "f_dc"},
            {"params": [self._features_rest], "lr": 0.0, "name": "f_rest"},
            {"params": [self._opacity], "lr": 0.0, "name": "opacity"},
            {"params": [self._scaling], "lr": 0.0, "name": "scaling"},
            {"params": [self._rotation], "lr": 0.0, "name": "rotation"},
            {"params": [cam_rot], "lr": cfg["cam_rot_lr"],
                "name": "cam_unnorm_rot"},
            {"params": [cam_trans], "lr": cfg["cam_trans_lr"],
                "name": "cam_trans"},
        ]
        if exposure_ab is not None:
            params.extend([
                {"params": [exposure_ab[0]], "lr": 0.01, "name": "exposure_a"},
                {"params": [exposure_ab[1]], "lr": 0.01, "name": "exposure_b"}]
            )
        self.optimizer = torch.optim.Adam(params, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.98, patience=10, verbose=False)

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        n_rgb = self._rgb.shape[1] if self._rgb.dim() >= 2 else 0
        for i in range(n_rgb):
            l.append("rgb_{}".format(i))
        return l

    def save_ply(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy())
        opacities = self._opacity.detach().cpu().numpy()
        if self.isotropic:
            # tile into shape (P, 2)
            scale = np.tile(self._scaling.detach().cpu().numpy()[:, 0].reshape(-1, 1), (1, 2))
        else:
            scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        rgb = self._rgb.detach().cpu().numpy()
        if rgb.ndim == 1 or rgb.shape[0] != xyz.shape[0]:
            rgb = np.zeros((xyz.shape[0], 3), dtype=np.float32)
        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attrs_to_concat = [xyz, normals, f_dc, f_rest, opacities, scale, rotation]
        if self._rgb.dim() >= 2 and self._rgb.shape[1] > 0:  # match construct_list_of_attributes
            attrs_to_concat.append(rgb)
        attributes = np.concatenate(attrs_to_concat, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"])),
                axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rgb_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rgb")]
        rgb_names = sorted(rgb_names, key=lambda x: int(x.split("_")[-1]))
        rgbs = np.zeros((xyz.shape[0], len(rgb_names)))
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rgb = nn.Parameter(
            torch.tensor(
                rgbs, dtype=torch.float, device=self.cfg["device"]
            ).requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "exposure" not in group["name"]:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                    del self.optimizer.state[group["params"][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                    self.optimizer.state[group["params"][0]] = stored_state
                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        mask = mask.detach().bool().to(self._xyz.device).contiguous()
        valid_points_mask = ~mask
        # Verify all optimizer params have consistent size before pruning
        n_points = self._xyz.shape[0]
        if mask.shape[0] != n_points:
            return
        for group in self.optimizer.param_groups:
            if "exposure" not in group["name"] and group["params"][0].shape[0] != n_points:
                return  # Skip prune if param sizes are inconsistent
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._rgb = optimizable_tensors["rgb"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        # #region agent log
        import json as _json
        _logpath = "/home/wujie/ws_0122/loopsplat_2dgs/.cursor/debug-25a42b.log"
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            if group["name"] in tensors_dict:
                extension_tensor = tensors_dict[group["name"]]
                current_param = group["params"][0]
                try:
                    with open(_logpath, "a") as _f:
                        _f.write(_json.dumps({"hypothesisId": "H1-H5", "location": "gaussian_model.cat_tensors_to_optimizer", "message": "shapes before cat", "data": {"group": group["name"], "current_shape": list(current_param.shape), "extension_shape": list(extension_tensor.shape), "dim1_match": current_param.dim() > 1 and extension_tensor.dim() > 1 and current_param.shape[1:] == extension_tensor.shape[1:]}, "timestamp": __import__("time").time()}) + "\n")
                except Exception:
                    pass
                # #endregion
                new_param = nn.Parameter(
                    torch.cat((current_param, extension_tensor), dim=0).requires_grad_(True))
                stored_state = self.optimizer.state.get(current_param, None)
                if stored_state is not None:
                    exp_avg = stored_state["exp_avg"]
                    exp_avg_sq = stored_state["exp_avg_sq"]
                    # Ensure non-dim0 shapes match (e.g. (N, 15, 3) vs (K, 15, 3)).
                    # After many prune/add cycles or inconsistent state, exp_avg can have wrong shape.
                    if exp_avg.shape[1:] == extension_tensor.shape[1:]:
                        new_exp_avg = torch.cat(
                            (exp_avg, torch.zeros_like(extension_tensor)), dim=0)
                        new_exp_avg_sq = torch.cat(
                            (exp_avg_sq, torch.zeros_like(extension_tensor)), dim=0)
                    else:
                        # Reinit state to avoid "Sizes of tensors must match except in dimension 0".
                        new_exp_avg = torch.zeros_like(new_param)
                        new_exp_avg_sq = torch.zeros_like(new_param)
                    stored_state["exp_avg"] = new_exp_avg
                    stored_state["exp_avg_sq"] = new_exp_avg_sq

                    del self.optimizer.state[current_param]
                    group["params"][0] = new_param
                    self.optimizer.state[new_param] = stored_state
                else:
                    group["params"][0] = new_param
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_rgb,new_features_dc, new_features_rest,
                              new_opacities, new_scaling, new_rotation):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "rgb": new_rgb,
        }

        if self.optimizer is None:
            # Initial load: set parameters directly (caller will call training_setup next).
            # Avoids cat with empty tensors of wrong shape (e.g. (0,) vs (N, 3)).
            self._xyz = nn.Parameter(d["xyz"].float().cuda().requires_grad_(True))
            self._features_dc = nn.Parameter(d["f_dc"].float().cuda().contiguous().requires_grad_(True))
            self._features_rest = nn.Parameter(d["f_rest"].float().cuda().contiguous().requires_grad_(True))
            self._opacity = nn.Parameter(d["opacity"].float().cuda().requires_grad_(True))
            self._scaling = nn.Parameter(d["scaling"].float().cuda().requires_grad_(True))
            self._rotation = nn.Parameter(d["rotation"].float().cuda().requires_grad_(True))
            self._rgb = nn.Parameter(d["rgb"].float().cuda().requires_grad_(True))
        else:
            optimizable_tensors = self.cat_tensors_to_optimizer(d)
            self._xyz = optimizable_tensors["xyz"]
            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]
            self._rgb = optimizable_tensors["rgb"]

        n = self.get_xyz().shape[0]
        self.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
        self.denom = torch.zeros((n, 1), device="cuda")
        self.max_radii2D = torch.zeros((n,), device="cuda")

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(
        self,
        iteration: int,
        densify_grad_threshold: float,
        densify_until_iter: int,
        percent_dense: float,
        prune_opacity_threshold: float,
    ):
        """Clone/split under- or over-reconstructed Gaussians and prune low opacity (3DGS-style)."""
        if self.optimizer is None or iteration >= densify_until_iter:
            return
        with torch.no_grad():
            n = self.get_xyz().shape[0]
            if n == 0:
                return

            # 1. Prune low opacity
            prune_mask = (self.get_opacity() < prune_opacity_threshold).squeeze()
            if prune_mask.any():
                valid = ~prune_mask
                if valid.sum() > 0:
                    self.prune_points(prune_mask)
                    n = self.get_xyz().shape[0]
                    if n == 0:
                        return
            # 插入: 打印要删除多少高斯
            if prune_mask.any():
                print(f"[densify_and_prune] Pruning {prune_mask.sum().item()} Gaussians due to low opacity.")
            # 2. Gradient and scale stats
            grad_avg = self.xyz_gradient_accum / self.denom.clamp(min=1)
            grad_avg = grad_avg.squeeze(-1)  # (N,)
            scales = self.get_scaling()  # (N, 2) or (N, 3)
            scale_max = scales.max(dim=1).values  # (N,)

            # Scene extent for percent_dense threshold
            xyz = self._xyz
            extent = (xyz.max(dim=0).values - xyz.min(dim=0).values).clamp(min=1e-6)
            thresh = extent.max().item() * percent_dense

            # 3. Clone: high gradient, small scale
            clone_mask = (grad_avg >= densify_grad_threshold) & (scale_max < thresh)
            # 4. Split: large scale
            split_mask = scale_max >= thresh

            scale_cols = self._scaling.shape[1]  # 2 or 3
            to_add_xyz, to_add_rgb = [], []
            to_add_f_dc, to_add_f_rest = [], []
            to_add_opacity, to_add_scaling, to_add_rotation = [], [], []

            if clone_mask.any():
                idx = torch.where(clone_mask)[0]
                # Slightly smaller scale and lower opacity for clones
                clone_scale = self._scaling.detach()[idx] + np.log(0.8)  # log space
                opa_linear = self.opacity_activation(self._opacity.detach()[idx])
                clone_opacity = self.inverse_opacity_activation(opa_linear * 0.5)
                to_add_xyz.append(self._xyz.detach()[idx])
                to_add_rgb.append(self._rgb.detach()[idx])
                to_add_f_dc.append(self._features_dc.detach()[idx])
                to_add_f_rest.append(self._features_rest.detach()[idx])
                to_add_opacity.append(clone_opacity)
                to_add_scaling.append(clone_scale)
                to_add_rotation.append(self._rotation.detach()[idx])

            if split_mask.any():
                idx = torch.where(split_mask)[0]
                scales_act = self.get_scaling()[idx]  # (K, 2) or (K, 3)
                if scale_cols == 2:
                    scale_3d = torch.cat(
                        [scales_act, scales_act[:, 1:2]], dim=1
                    )  # (K, 3) e.g. (s0,s1,s1)
                else:
                    scale_3d = scales_act
                rots = self.get_rotation()[idx]
                L = build_scaling_rotation(scale_3d, rots)  # (K, 3, 3)
                cov = L @ L.transpose(1, 2)
                # Principal direction: eigenvector for largest eigenvalue
                evals, evecs = torch.linalg.eigh(cov)  # evals (K,3), evecs (K,3,3)
                principal = evecs[:, :, -1]  # (K, 3) largest eval last
                scale_max_idx = scale_max[idx]  # (K,)
                offset = scale_max_idx.unsqueeze(1) * 0.2 * principal  # (K, 3)
                xyz_idx = self._xyz.detach()[idx]
                half_scale_log = self._scaling.detach()[idx] - np.log(2.0)
                opa_linear = self.opacity_activation(self._opacity.detach()[idx])
                split_opacity = self.inverse_opacity_activation(opa_linear * 0.5)

                # Two new points per split
                xyz_1 = xyz_idx + offset
                xyz_2 = xyz_idx - offset
                to_add_xyz.extend([xyz_1, xyz_2])
                to_add_rgb.extend(
                    [self._rgb.detach()[idx], self._rgb.detach()[idx]]
                )
                to_add_f_dc.extend(
                    [self._features_dc.detach()[idx], self._features_dc.detach()[idx]]
                )
                to_add_f_rest.extend(
                    [
                        self._features_rest.detach()[idx],
                        self._features_rest.detach()[idx],
                    ]
                )
                to_add_opacity.extend([split_opacity, split_opacity])
                to_add_scaling.extend([half_scale_log, half_scale_log])
                to_add_rotation.extend(
                    [self._rotation.detach()[idx], self._rotation.detach()[idx]]
                )

            if not to_add_xyz:
                return

            new_xyz = torch.cat(to_add_xyz, dim=0)
            new_rgb = torch.cat(to_add_rgb, dim=0)
            new_f_dc = torch.cat(to_add_f_dc, dim=0)
            new_f_rest = torch.cat(to_add_f_rest, dim=0)
            new_opacity = torch.cat(to_add_opacity, dim=0)
            new_scaling = torch.cat(to_add_scaling, dim=0)
            new_rotation = torch.cat(to_add_rotation, dim=0)

            self.densification_postfix(
                new_xyz, new_rgb, new_f_dc, new_f_rest,
                new_opacity, new_scaling, new_rotation,
            )

    def reset_opacity_for_densification(self, decay: float = 0.8):
        """Reset opacity (global gentle shrink) so optimization can prune; sync optimizer."""
        if self.optimizer is None:
            return
        with torch.no_grad():
            opa = self.opacity_activation(self._opacity)
            opa_new = (opa * decay).clamp(1e-6, 1.0 - 1e-6)
            new_opacity_tensor = self.inverse_opacity_activation(opa_new)
            self.replace_tensor_to_optimizer(new_opacity_tensor, "opacity")
            for group in self.optimizer.param_groups:
                if group["name"] == "opacity":
                    self._opacity = group["params"][0]
                    break

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        if type(pcd) is BasicPointCloud:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd.points)).float().cuda()
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd.colors)).float().cuda())
        else:
            fused_point_cloud = torch.tensor(
                np.asarray(pcd._xyz)).float().cuda()
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd._rgb)).float().cuda())
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ",
              fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(
            distCUDA2(fused_point_cloud.detach().clone().float().cuda()), 0.0000001)
        scale_cols = 3 if not self.isotropic else 2
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, scale_cols)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        if type(pcd) is BasicPointCloud:
            rgb_data = np.asarray(pcd.colors)
        else:
            rgb_data = np.asarray(pcd._rgb)
        self._rgb = nn.Parameter(torch.tensor(rgb_data).float().cuda().requires_grad_(True))
        self.max_radii2D = torch.zeros(
            (self.get_xyz().shape[0]), device="cuda")
