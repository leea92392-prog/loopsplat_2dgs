""" This module is responsible for merging submaps. """
from argparse import ArgumentParser

import faiss
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.gaussian_model import GaussianModel
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.utils import (batch_search_faiss, get_render_settings,
                             np2ptcloud, render_gaussian_model, torch2np)
from src.utils.gaussian_model_utils import BasicPointCloud, RGB2SH, inverse_sigmoid


class RenderFrames(Dataset):
    """A dataset class for loading keyframes along with their estimated camera poses and render settings."""
    def __init__(self, dataset, render_poses: np.ndarray, height: int, width: int, fx: float, fy: float, exposures_ab=None):
        self.dataset = dataset
        self.render_poses = render_poses
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.device = "cuda"
        self.stride = 1
        self.exposures_ab = exposures_ab
        if len(dataset) > 1000:
            self.stride = len(dataset) // 1000

    def __len__(self) -> int:
        return len(self.dataset) // self.stride

    def __getitem__(self, idx):
        idx = idx * self.stride
        color = (torch.from_numpy(
            self.dataset[idx][1]) / 255.0).float().to(self.device)
        depth = torch.from_numpy(self.dataset[idx][2]).float().to(self.device)
        estimate_c2w = self.render_poses[idx]
        estimate_w2c = np.linalg.inv(estimate_c2w)
        frame = {
            "frame_id": idx,
            "color": color,
            "depth": depth,
            "render_settings": get_render_settings(
                self.width, self.height, self.dataset.intrinsics, estimate_w2c)
        }
        if self.exposures_ab is not None:
            frame["exposure_ab"] = self.exposures_ab[idx]
        return frame


def render_frames_collate_fn(batch):
    """Custom collate that returns the single frame as-is (avoids batching render_settings tuple)."""
    return batch[0]


def merge_submaps(submaps_paths: list, radius: float = 0.01, device: str = "cuda") -> o3d.geometry.PointCloud:
    """ Merge submaps into a single point cloud, which is then used for global map refinement.
    Args:
        segments_paths (list): Folder path of the submaps.
        radius (float, optional): Nearest neighbor distance threshold for adding a point. Defaults to 0.0001.
        device (str, optional): Defaults to "cuda".

    Returns:
        o3d.geometry.PointCloud: merged point cloud
    """
    pts_index = faiss.IndexFlatL2(3)
    if device == "cuda":
        pts_index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(),
            0,
            faiss.IndexIVFFlat(faiss.IndexFlatL2(3), 3, 500, faiss.METRIC_L2))
        pts_index.nprobe = 5
    merged_pts = []
    for submap_path in tqdm(submaps_paths, desc="Merging submaps"):
        gaussian_params = torch.load(submap_path)["gaussian_params"]
        current_pts = gaussian_params["xyz"].to(device).float().contiguous()
        pts_index.train(current_pts)
        distances, _ = batch_search_faiss(pts_index, current_pts, 8)
        neighbor_num = (distances < radius).sum(axis=1).int()
        ids_to_include = torch.where(neighbor_num == 0)[0]
        pts_index.add(current_pts[ids_to_include])
        merged_pts.append(current_pts[ids_to_include])
    pts = torch2np(torch.vstack(merged_pts))
    pt_cloud = np2ptcloud(pts, np.zeros_like(pts))

    # Downsampling if the total number of points is too large
    if len(pt_cloud.points) > 800_000:
        voxel_size = 0.06
        pt_cloud = pt_cloud.voxel_down_sample(voxel_size)
        print(f"Downsampled point cloud to {len(pt_cloud.points)} points")
    filtered_pt_cloud, _ = pt_cloud.remove_statistical_outlier(nb_neighbors=40, std_ratio=3.0)
    del pts_index
    return filtered_pt_cloud


def backproject_frame_to_pointcloud(
    gt_color: torch.Tensor,
    gt_depth: torch.Tensor,
    est_w2c: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    mask: torch.Tensor,
    device: str = "cuda",
):
    """Back-project masked pixels of current frame to world point cloud (xyz+rgb) and mean_sq_dist for scaling.
    gt_color: (3, H, W) or (H, W, 3); gt_depth: (H, W); mask: (H*W,) boolean.
    Returns: new_pt_cld (N, 6), mean3_sq_dist (N,) on device.
    """
    if gt_color.dim() == 3 and gt_color.shape[0] == 3:
        color = gt_color  # (3, H, W)
    elif gt_color.dim() == 3 and gt_color.shape[-1] == 3:
        color = gt_color.permute(2, 0, 1)
    else:
        color = gt_color
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    height, width = gt_depth.shape
    x_grid, y_grid = torch.meshgrid(
        torch.arange(width, device=device).float(),
        torch.arange(height, device=device).float(),
        indexing="xy",
    )
    xx = (x_grid - cx) / fx
    yy = (y_grid - cy) / fy
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = gt_depth.reshape(-1)
    pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
    pix_ones = torch.ones(height * width, 1, device=device).float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.linalg.inv(est_w2c)
    pts = (c2w @ pts4.T).T[:, :3]
    scale_gaussian = depth_z / ((fx + fy) / 2)
    mean3_sq_dist = scale_gaussian ** 2
    cols = color.permute(1, 2, 0).reshape(-1, 3)
    point_cld = torch.cat((pts, cols), dim=-1)
    point_cld = point_cld[mask]
    mean3_sq_dist = mean3_sq_dist[mask]
    return point_cld, mean3_sq_dist


def _scale_mode_1d(scales_1d: np.ndarray, bins: int = 100) -> float:
    """Approximate mode of 1D scale distribution via histogram (bin center with max count)."""
    if scales_1d.size == 0:
        return 0.0
    mn, mx = scales_1d.min(), scales_1d.max()
    if mn >= mx:
        return float(mn)
    counts, bin_edges = np.histogram(scales_1d, bins=min(bins, max(2, scales_1d.size)))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return float(bin_centers[np.argmax(counts)])


def prune_oversized_gaussians(gaussian_model: GaussianModel, max_scale_ratio: float = 3.0) -> int:
    """Remove Gaussians whose max scale > max_scale_ratio * mode(scale). Returns number pruned."""
    scales = gaussian_model.get_scaling()
    scale_max = scales.max(dim=-1).values.detach().cpu().numpy()
    mode_scale = _scale_mode_1d(scale_max)
    threshold = max_scale_ratio * mode_scale
    prune_mask = torch.from_numpy(scale_max > threshold).to(scales.device)
    n_prune = prune_mask.sum().item()
    if n_prune > 0:
        n_before = gaussian_model.get_size()
        gaussian_model.prune_points(prune_mask)
        n_after = gaussian_model.get_size()
        print(f"prune_oversized: mode_scale={mode_scale:.6f} threshold={max_scale_ratio}*mode={threshold:.6f} "
              f"removed {n_prune} Gaussians (before={n_before} after={n_after})")
    return n_prune


def refine_global_map(pt_cloud: o3d.geometry.PointCloud, training_frames: list, max_iterations: int,
                      export_refine_mesh=True, output_dir=".",
                      len_frames=None, o3d_intrinsic=None, enable_sh=True, enable_exposure=False,
                      max_scale_ratio_to_mode: float = 5.0,
                      add_gaussians_every: int = 0,
                      height=None, width=None, fx=None, fy=None, cx=None, cy=None,
                      add_gaussians_max_points: int = 5000,
                      depth_error_median_k: float = 10.0,
                      prune_opacity_threshold: float = 0.1,
                      new_gaussian_scale_max: float = 0.02) -> GaussianModel:
    """Refines a global map based on the merged point cloud and training keyframes.
    Optionally adds new Gaussians periodically from current frame where detail is missing
    (low alpha, large depth error, or weak normal), by back-projecting from depth.
    Args:
        pt_cloud: The merged point cloud used for refinement.
        training_frames: Iterator of training frames for map refinement.
        max_iterations: Maximum number of refinement iterations.
        add_gaussians_every: Add Gaussians every N iterations (0 to disable).
        height, width, fx, fy, cx, cy: Intrinsics for back-projection when add_gaussians_every > 0.
        add_gaussians_max_points: Max new points to add per injection (subsample if more).
        depth_error_median_k: Depth error threshold = k * median(valid depth error).
        prune_opacity_threshold: Prune Gaussians with opacity below this (sigmoid space); higher = prune less.
        new_gaussian_scale_max: Cap initial scale (world units) for new Gaussians to avoid sticking at upper limit.
    Returns:
        GaussianModel: The refined global map.
    """
    has_intrinsics = all(x is not None for x in (height, width, fx, fy, cx, cy))
    do_add_gaussians = add_gaussians_every > 0 and has_intrinsics
    opt_params = OptimizationParams(ArgumentParser(description="Training script parameters"))

    gaussian_model = GaussianModel(3)
    gaussian_model.active_sh_degree = 0
    if pt_cloud is None:
        output_mesh = output_dir / "mesh" / "cleaned_mesh.ply"
        output_mesh = o3d.io.read_triangle_mesh(str(output_mesh))
        pcd = o3d.geometry.PointCloud()
        pcd.points = output_mesh.vertices
        pcd.colors = output_mesh.vertex_colors
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd = BasicPointCloud(points=np.asarray(pcd.points),
                            colors=np.asarray(pcd.colors))
        gaussian_model.create_from_pcd(pcd, 1.0)
        gaussian_model.training_setup(opt_params)
    else:
        gaussian_model.training_setup(opt_params)
        gaussian_model.add_points(pt_cloud)

    if max_scale_ratio_to_mode is not None and max_scale_ratio_to_mode > 0:
        prune_oversized_gaussians(gaussian_model, max_scale_ratio=max_scale_ratio_to_mode)

    iteration = 0
    for iteration in tqdm(range(max_iterations), desc="Refinement"):
        torch.cuda.empty_cache()
        training_frame = next(training_frames)
        gaussian_model.update_learning_rate(iteration)
        if enable_sh and iteration > 0 and iteration % 1000 == 0:
            gaussian_model.oneupSHdegree()
        gt_color = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
        gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
        render_settings, est_w2c = training_frame["render_settings"]

        render_dict = render_gaussian_model(gaussian_model, render_settings, est_w2c)
        rendered_color, rendered_depth = (render_dict["color"].permute(1, 2, 0), render_dict["depth"])
        if enable_exposure and training_frame.get("exposure_ab") is not None:
            rendered_color = torch.clamp(
                rendered_color * torch.exp(training_frame["exposure_ab"][0,0]) + training_frame["exposure_ab"][0,1], 0, 1.)

        reg_loss = isotropic_loss(gaussian_model.get_scaling())
        depth_mask = (gt_depth > 0)
        color_loss = (1.0 - opt_params.lambda_dssim) * l1_loss(
            rendered_color[depth_mask, :], gt_color[depth_mask, :]
        ) + opt_params.lambda_dssim * (1.0 - ssim(rendered_color, gt_color))
        depth_loss = l1_loss(
            rendered_depth[:, depth_mask], gt_depth[depth_mask])

        rend_dist = render_dict["rend_dist"]
        dist_loss = 1000*rend_dist.mean()
        rend_normal  = render_dict['rend_normal']
        surf_normal = render_dict['normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = 0.05*(normal_error).mean()
        total_loss = color_loss + depth_loss + reg_loss + dist_loss + normal_loss

        total_loss.backward()

        with torch.no_grad():
            # Periodically add Gaussians from current frame where detail is missing (alpha/depth/normal)
            # Skip prune at iteration 0; delay first prune to let model converge
            if iteration > 0 and iteration % 1000 == 0:
                prune_mask = (gaussian_model.get_opacity() < prune_opacity_threshold).squeeze()
                n_valid = (~prune_mask).sum().item()
                n_prune = prune_mask.sum().item()
                n_before = gaussian_model.get_size()
                if n_valid > 0:  # Only prune if we keep at least some points
                    gaussian_model.prune_points(prune_mask)
                n_after = gaussian_model.get_size()
                print(f"prune: removing {n_prune} low-opacity points, before={n_before} after={n_after}")
                if max_scale_ratio_to_mode is not None and max_scale_ratio_to_mode > 0:
                    prune_oversized_gaussians(gaussian_model, max_scale_ratio=max_scale_ratio_to_mode)

            if do_add_gaussians and iteration > 0 and iteration % add_gaussians_every == 0:
                gt_color_add = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
                gt_depth_add = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
                if gt_depth_add.dim() == 2:
                    gt_depth_add = gt_depth_add.unsqueeze(0)
                valid_depth_mask = (gt_depth_add > 0.1).squeeze(0)
                # Reuse render_dict from this iteration (same frame)
                silhouette = render_dict["alpha"]
                non_presence_sil = (silhouette < 0.5).squeeze(0)
                render_depth_add = render_dict["depth"].squeeze(0)
                gt_d = gt_depth_add.squeeze(0)
                depth_error = torch.abs(gt_d - render_depth_add) * (gt_d > 0.05)
                valid_err = depth_error[gt_d > 0.05]
                non_presence_depth = depth_error > (depth_error_median_k * valid_err.median() if valid_err.numel() > 0 else 1e6)
                surf_norm = render_dict["normal"]
                norm_mag = torch.sqrt((surf_norm ** 2).sum(dim=0))
                norm_mask = norm_mag < 0.5
                non_presence_mask = (non_presence_sil | non_presence_depth | norm_mask).reshape(-1) & valid_depth_mask.reshape(-1)
                print(f"non_presence_mask: {non_presence_mask.sum().item()}")
                if non_presence_mask.any():
                    w2c = est_w2c.squeeze().float().cuda()
                    new_pt_cld, mean3_sq_dist = backproject_frame_to_pointcloud(
                        gt_color_add, gt_d, w2c, fx, fy, cx, cy, non_presence_mask, device="cuda")
                    if new_pt_cld.shape[0] > 0:
                        if new_pt_cld.shape[0] > add_gaussians_max_points:
                            perm = torch.randperm(new_pt_cld.shape[0], device=new_pt_cld.device)[:add_gaussians_max_points]
                            new_pt_cld = new_pt_cld[perm]
                            mean3_sq_dist = mean3_sq_dist[perm]
                        new_rgb = new_pt_cld[:, 3:6].float().cuda()
                        fused_color = RGB2SH(new_pt_cld[:, 3:6].float().cuda())
                        new_features = torch.zeros(
                            (fused_color.shape[0], 3, (gaussian_model.max_sh_degree + 1) ** 2), device="cuda").float()
                        new_features[:, :3, 0] = fused_color
                        new_features[:, 3:, 1:] = 0.0
                        new_rots = torch.zeros((new_pt_cld.shape[0], 4), device="cuda")
                        new_rots[:, 0] = 1
                        # 新高斯激活后透明度设为 0.5（参数存 inverse_sigmoid(0.5)=0）
                        new_opacities = inverse_sigmoid(torch.tensor(0.8, device="cuda")).expand(
                            new_pt_cld.shape[0], 1)
                        # 限制初始 scale 上限，避免优化时卡在 scale 上限
                        scale_val = torch.sqrt(mean3_sq_dist).clamp(min=1e-6, max=new_gaussian_scale_max)
                        new_scaling = torch.log(scale_val)[..., None].repeat(1, 2)
                        gaussian_model.densification_postfix(
                            new_xyz=new_pt_cld[:, :3],
                            new_rgb=new_rgb,
                            new_features_dc=new_features[:, :, 0:1].transpose(1, 2).contiguous(),
                            new_features_rest=new_features[:, :, 1:].transpose(1, 2).contiguous(),
                            new_opacities=new_opacities,
                            new_scaling=new_scaling,
                            new_rotation=new_rots,
                        )
                        print(f"refine_global_map: added {new_pt_cld.shape[0]} Gaussians at iter {iteration} (total now {gaussian_model.get_size()})")
            # Optimizer step
            gaussian_model.optimizer.step()
            gaussian_model.optimizer.zero_grad(set_to_none=True)
        iteration += 1
    
    try:
        if export_refine_mesh:
            output_dir = output_dir / "mesh" / "refined_mesh.ply"
            scale = 1.0
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=5.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for i in tqdm(range(len_frames), desc="Integrating mesh"):  # one cycle
                training_frame = next(training_frames)
                gt_color = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
                gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
                render_settings, estimate_w2c = training_frame["render_settings"]

                render_dict = render_gaussian_model(gaussian_model, render_settings, estimate_w2c)
                rendered_color, rendered_depth = (
                    render_dict["color"].permute(1, 2, 0), render_dict["depth"])
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth.squeeze())
                # rendered_depth = filter_depth_outliers(
                #     rendered_depth, kernel_size=20, threshold=0.1)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    depth_trunc=30,
                    convert_rgb_to_intensity=False)
                volume.integrate(
                    rgbd, o3d_intrinsic, estimate_w2c.squeeze().cpu().numpy().astype(np.float64))

            o3d_mesh = volume.extract_triangle_mesh()
            o3d.io.write_triangle_mesh(str(output_dir), o3d_mesh)
            print(f"Refined mesh saved to {output_dir}")

    except Exception as e:
        print(f"Error export_refine_mesh in refine_global_map:\n {e}")

    return gaussian_model
