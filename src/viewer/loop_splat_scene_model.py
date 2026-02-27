"""
LoopSplat Scene Model - adapts output directory data for the GaussianViewer.
Loads from output dir (PLY or submaps), provides render() and keyframe interface.
"""
from pathlib import Path
import math
import numpy as np
import torch
import yaml
from diff_surfel_rasterization import GaussianRasterizationSettings

from src.entities.gaussian_model import GaussianModel
from src.entities.arguments import OptimizationParams
from argparse import ArgumentParser
from src.utils.utils import render_gaussian_model


class SimpleKeyframe:
    """Minimal keyframe object for viewer compatibility."""
    def __init__(self, c2w: np.ndarray):
        self._c2w = torch.from_numpy(c2w).float().cuda()
        self.approx_centre = self._c2w[:3, 3]

    def get_Rt(self) -> torch.Tensor:
        """Return world-to-camera 4x4 (Rt = w2c)."""
        w2c = np.linalg.inv(self._c2w.cpu().numpy())
        return torch.from_numpy(w2c).float().cuda()


class LoopSplatSceneModel:
    """
    Scene model adapter for LoopSplat output. Compatible with GaussianViewer.
    """
    def __init__(self, gaussian_model: GaussianModel, config: dict, keyframe_poses: np.ndarray):
        self._gaussian_model = gaussian_model
        self._config = config
        self._keyframe_poses = keyframe_poses  # (N, 4, 4) c2w
        self._build_intrinsics()

    def _build_intrinsics(self):
        cam = self._config.get("cam", {})
        self.width = int(cam.get("W", 640))
        self.height = int(cam.get("H", 480))
        fx = float(cam.get("fx", 525.0))
        fy = float(cam.get("fy", 525.0))
        cx = float(cam.get("cx", self.width / 2))
        cy = float(cam.get("cy", self.height / 2))
        crop = int(cam.get("crop_edge", 0))
        self.width -= 2 * crop
        self.height -= 2 * crop
        cx -= crop
        cy -= crop
        self.intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)
        self.FoVx = 2 * math.atan(self.width / (2 * fx))
        self.FoVy = 2 * math.atan(self.height / (2 * fy))
        self.f = (fx + fy) / 2.0  # approx focal for pose drawing

    @property
    def gaussian_model(self):
        return self._gaussian_model

    @property
    def xyz(self) -> torch.Tensor:
        return self._gaussian_model.get_xyz()

    @property
    def rotation(self) -> torch.Tensor:
        return self._gaussian_model.get_rotation()

    @property
    def scaling(self) -> torch.Tensor:
        return self._gaussian_model.get_scaling()

    @property
    def opacity(self) -> torch.Tensor:
        return self._gaussian_model.get_opacity()

    @property
    def f_dc(self) -> torch.Tensor:
        return self._gaussian_model._features_dc

    @property
    def n_active_gaussians(self) -> int:
        return self._gaussian_model.get_size()

    @property
    def keyframes(self) -> list:
        return [SimpleKeyframe(p) for p in self._keyframe_poses]

    def get_Rts(self) -> torch.Tensor:
        """Return (N, 4, 4) world-to-camera matrices."""
        w2cs = []
        for c2w in self._keyframe_poses:
            w2c = np.linalg.inv(c2w)
            w2cs.append(w2c)
        return torch.from_numpy(np.stack(w2cs)).float().cuda()

    def get_gt_Rts(self, _unused: bool) -> torch.Tensor:
        """GT poses not available for LoopSplat - return empty."""
        return torch.empty(0, 4, 4).cuda()

    def get_closest_keyframe(self, position: torch.Tensor, n: int = 1) -> list:
        """Return n closest keyframes by distance to position."""
        if len(self._keyframe_poses) == 0:
            return []
        centres = np.array([p[:3, 3] for p in self._keyframe_poses])
        pos_np = position.detach().cpu().numpy()
        dists = np.linalg.norm(centres - pos_np, axis=1)
        idx = np.argsort(dists)[:n]
        kfs = self.keyframes
        return [kfs[i] for i in idx]

    @property
    def anchors(self):
        return None

    @property
    def anchor_weights(self):
        return None

    def _get_render_settings(self, width: int, height: int, fov_x: float, fov_y: float,
                             w2c: torch.Tensor, scale_modifier: float = 1.0):
        """Build GaussianRasterizationSettings for arbitrary resolution and FOV."""
        fx = width / (2 * math.tan(fov_x / 2))
        fy = height / (2 * math.tan(fov_y / 2))
        cx = width / 2
        cy = height / 2
        near, far = 0.01, 100.0
        w2c = torch.eye(4).cuda().float()
        cam_center = w2c.inverse()[3, :3]
        opengl_proj = torch.tensor([[2 * fx / width, 0.0, -(width - 2 * cx) / width, 0.0],
                                [0.0, 2 * fy / height, -(height - 2 * cy) / height, 0.0],
                                [0.0, 0.0, far /
                                    (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]], device='cuda').float().transpose(0, 1)
        full_proj_matrix = w2c.unsqueeze(0).bmm(opengl_proj.unsqueeze(0)).squeeze(0)
        return GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=width / (2 * fx),
            tanfovy=height / (2 * fy),
            bg=torch.tensor([0, 0, 0], device="cuda").float(),
            scale_modifier=scale_modifier,
            viewmatrix=w2c,
            projmatrix=full_proj_matrix,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False,
            debug=False,
        )

    def render(self, width: int, height: int, viewmatrix: torch.Tensor, scaling_factor: float,
               bg_color: torch.Tensor, top_view: bool, fov_x: float, fov_y: float) -> dict:
        """
        Render scene. viewmatrix from viewer is camera.to_camera.T (w2c transposed for OpenGL).
        Returns dict with 'render' (3,H,W) and 'invdepth' (1,H,W).
        """
        w2c = viewmatrix.float().cuda().contiguous()
        if w2c.dim() == 2 and w2c.shape[0] == 4 and w2c.shape[1] == 4:
            w2c = w2c.T
        est_w2c = w2c
        settings = self._get_render_settings(width, height, fov_x, fov_y, w2c, scaling_factor)
        render_dict = render_gaussian_model(
            self._gaussian_model, settings, est_w2c
        )
        color = render_dict["color"].clamp(0, 1.0)
        depth = render_dict["depth"][0:1]
        eps = 1e-6
        invdepth = 1.0 / (depth + eps)
        invdepth = torch.nan_to_num(invdepth, 0.0, 0.0)
        return {"render": color, "invdepth": invdepth}

    def get_scaling_for_ellipsoid(self) -> np.ndarray:
        """Return (N, 3) scales for EllipsoidViewer. 2DGS has (N,2), pad 3rd dim."""
        s = self.scaling.detach().cpu().numpy()
        if s.ndim == 1:
            s = s[:, None]
        if s.shape[1] == 2:
            s3 = np.minimum(s[:, 0], s[:, 1])
            s = np.concatenate([s, s3[:, None]], axis=1)
        elif s.shape[1] == 1:
            s = np.repeat(s, 3, axis=1)
        return s.astype(np.float32)

    def get_colors_for_ellipsoid(self) -> np.ndarray:
        """Return (N, 48) SH colors for EllipsoidViewer. f_dc is (N,1,3), expand to 48."""
        f_dc = self.f_dc.detach().cpu().numpy()
        N = f_dc.shape[0]
        colors = np.zeros((N, 48), dtype=np.float32)
        colors[:, 0] = f_dc[:, 0, 0]
        colors[:, 1] = f_dc[:, 0, 1]
        colors[:, 2] = f_dc[:, 0, 2]
        return colors

    @classmethod
    def from_output_dir(cls, output_dir: str):
        output_path = Path(output_dir)
        if not output_path.exists():
            raise FileNotFoundError(f"Output dir not found: {output_dir}")

        config_path = output_path / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found in {output_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        poses_path = output_path / "estimated_c2w.ckpt"
        if not poses_path.exists():
            raise FileNotFoundError(f"estimated_c2w.ckpt not found in {output_path}")
        estimated_c2w = torch.load(poses_path, map_location="cpu")
        if isinstance(estimated_c2w, torch.Tensor):
            keyframe_poses = estimated_c2w.numpy()
        else:
            keyframe_poses = np.array(estimated_c2w)

        scene_name = config.get("data", {}).get("scene_name", "scene")
        ply_path = output_path / f"{scene_name}_global_splats.ply"
        if not ply_path.exists():
            found = list(output_path.glob("*_global_splats.ply"))
            ply_path = found[0] if found else None

        gaussian_model = GaussianModel(sh_degree=3, isotropic=False)
        gaussian_model.active_sh_degree = 0

        if ply_path is not None and ply_path.exists():
            gaussian_model.cfg = {"device": "cuda"}
            gaussian_model.load_ply(str(ply_path))
        else:
            submaps_dir = output_path / "submaps"
            if not submaps_dir.exists():
                raise FileNotFoundError(
                    f"Neither {ply_path.name} nor submaps/ found. "
                    "Run evaluation to generate global_splats.ply, or ensure submaps exist."
                )
            submap_paths = sorted(submaps_dir.glob("*.ckpt"))
            if not submap_paths:
                raise FileNotFoundError(f"No .ckpt files in {submaps_dir}")
            opt = OptimizationParams(ArgumentParser(description="Viewer"))
            gaussian_model.training_setup(opt)
            all_xyz, all_rgb, all_fdc, all_frest = [], [], [], []
            all_opacity, all_scaling, all_rotation = [], [], []
            for p in submap_paths:
                ckpt = torch.load(p, map_location="cuda")
                gp = ckpt["gaussian_params"]
                all_xyz.append(gp["xyz"])
                all_rgb.append(gp["rgb"])
                all_fdc.append(gp["features_dc"])
                all_frest.append(gp["features_rest"])
                all_opacity.append(gp["opacity"])
                all_scaling.append(gp["scaling"])
                all_rotation.append(gp["rotation"])
            xyz = torch.cat(all_xyz, dim=0).cuda()
            rgb = torch.cat(all_rgb, dim=0).cuda()
            fdc = torch.cat(all_fdc, dim=0).cuda()
            frest = torch.cat(all_frest, dim=0).cuda()
            opacity = torch.cat(all_opacity, dim=0).cuda()
            scaling = torch.cat(all_scaling, dim=0).cuda()
            rotation = torch.cat(all_rotation, dim=0).cuda()
            gaussian_model.densification_postfix(xyz, rgb, fdc, frest, opacity, scaling, rotation)

        return cls(gaussian_model, config, keyframe_poses)
