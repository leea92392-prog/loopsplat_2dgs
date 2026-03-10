""" This module is responsible for merging submaps. """
from argparse import ArgumentParser
from pathlib import Path

import cv2
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
from src.utils.vis_utils import show_render_result


class RenderFrames(Dataset):
    """A dataset class for loading keyframes along with their estimated camera poses and render settings."""
    def __init__(self, dataset, render_poses: np.ndarray, height: int, width: int, fx: float, fy: float, exposures_ab=None, config=None):
        self.dataset = dataset
        self.render_poses = render_poses
        self.height = height
        self.width = width
        self.fx = fx
        self.fy = fy
        self.device = "cuda"
        self.stride = 1
        self.exposures_ab = exposures_ab
        self.config = config
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
        renderer_type = self.config.get("renderer", "2dgs") if self.config else "2dgs"
        render_settings, est_w2c = get_render_settings(
            self.width, self.height, self.dataset.intrinsics, estimate_w2c, renderer_type=renderer_type)
        frame = {
            "frame_id": idx,
            "color": color,
            "depth": depth,
            "render_settings": (render_settings, est_w2c),
            "renderer_type": renderer_type,
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


def prune_oversized_gaussians(gaussian_model: GaussianModel, max_scale_ratio: float = 3.0, verbose: bool = True) -> int:
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
        if verbose:
            print(f"prune_oversized: mode_scale={mode_scale:.6f} threshold={max_scale_ratio}*mode={threshold:.6f} "
                  f"removed {n_prune} Gaussians (before={n_before} after={n_after})")
    return n_prune


def _vis_refine_frame(render_dict: dict, training_frame: dict, save_id: str = None, save_path: str = None) -> None:
    """用 OpenCV 可视化当前帧的渲染结果：RGB、深度、法线、轮廓(alpha)。复用 show_render_result。"""
    gt_color = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
    gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
    # show_render_result 需要 gt_rgb (3,H,W), gt_depth (H,W)
    if gt_color.dim() == 3 and gt_color.shape[-1] == 3:
        gt_rgb = gt_color.permute(2, 0, 1)
    else:
        gt_rgb = gt_color
    if gt_depth.dim() == 3:
        gt_depth = gt_depth.squeeze(0)
    render_rgb = render_dict["color"]
    render_depth = render_dict["depth"]
    render_normal = render_dict.get("rend_normal")
    render_alpha = render_dict["alpha"]
    show_render_result(
        # gt_rgb=gt_rgb,
        # gt_depth=gt_depth,
        render_rgb=render_rgb,
        render_depth=render_depth,
        render_normal=render_normal,
        # render_alpha=render_alpha,
        save_id=save_id,
        save_path=save_path,
    )


def _compute_pixel_metrics_for_vis(render_dict: dict, training_frame: dict):
    """与加点条件一致的 per-pixel 原始值，用于交互查看。(H,W) numpy，与渲染同分辨率。"""
    gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
    if gt_depth.dim() == 2:
        gt_depth = gt_depth.unsqueeze(0)
    gt_d = gt_depth.squeeze(0)
    silhouette = render_dict["alpha"].squeeze(0)
    render_depth = render_dict["depth"].squeeze(0)
    depth_error = torch.abs(gt_d - render_depth) * (gt_d > 0.05)
    surf_norm = render_dict.get("normal")
    norm_mag = torch.sqrt((surf_norm ** 2).sum(dim=0)) if surf_norm is not None else torch.zeros_like(silhouette)
    return (
        silhouette.detach().cpu().numpy().astype(np.float32),
        norm_mag.detach().cpu().numpy().astype(np.float32),
        depth_error.detach().cpu().numpy().astype(np.float32),
    )


def _compute_add_mask_for_vis(
    render_dict: dict,
    training_frame: dict,
    depth_error_median_k: float,
    add_valid_depth_min: float,
    add_depth_valid_for_error_min: float,
    add_silhouette_threshold: float,
    add_norm_mag_threshold: float,
) -> np.ndarray:
    """与 _add_gaussians_from_frame 完全一致的“满足添加条件”二值掩码，(H,W) bool。"""
    gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
    if gt_depth.dim() == 2:
        gt_depth = gt_depth.unsqueeze(0)
    gt_d = gt_depth.squeeze(0)
    valid_depth_mask = (gt_d > add_valid_depth_min).detach().cpu().numpy()
    silhouette = render_dict["alpha"].squeeze(0)
    non_presence_sil = (silhouette < add_silhouette_threshold).detach().cpu().numpy()
    render_depth = render_dict["depth"].squeeze(0)
    depth_error = torch.abs(gt_d - render_depth) * (gt_d > add_depth_valid_for_error_min)
    valid_err = depth_error[gt_d > add_depth_valid_for_error_min]
    thresh = depth_error_median_k * valid_err.median().item() if valid_err.numel() > 0 else 1e6
    non_presence_depth = (depth_error > thresh).detach().cpu().numpy()
    surf_norm = render_dict.get("normal")
    if surf_norm is not None:
        norm_mag = torch.sqrt((surf_norm ** 2).sum(dim=0))
        norm_mask = (norm_mag < add_norm_mag_threshold).detach().cpu().numpy()
    else:
        norm_mask = np.zeros_like(non_presence_sil, dtype=bool)
    add_mask = (non_presence_sil | non_presence_depth | norm_mask) & valid_depth_mask
    return add_mask


def _compute_prune_projection_for_vis(
    gaussian_model: GaussianModel,
    prune_opacity_threshold: float,
    est_w2c: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    height: int, width: int,
):
    """将被修剪（opacity < 阈值）的高斯投影到当前视图，返回 (N, 5) numpy: u, v, opacity, scale0, scale1。"""
    with torch.no_grad():
        opacities = gaussian_model.get_opacity().squeeze(-1)
        scales = gaussian_model.get_scaling()
        prune_mask = opacities < prune_opacity_threshold
        if not prune_mask.any():
            return np.zeros((0, 5), dtype=np.float32)
        xyz = gaussian_model.get_xyz()[prune_mask]
        opa = opacities[prune_mask].cpu().numpy().astype(np.float32)
        sc = scales[prune_mask].cpu().numpy().astype(np.float32)
        w2c = est_w2c.squeeze().float().cuda()
        xyz_cam = (w2c[:3, :3] @ xyz.T).T + w2c[:3, 3]
        z = xyz_cam[:, 2]
        valid = (z > 1e-4).cpu().numpy()
        u = (xyz_cam[:, 0] / z * fx + cx).cpu().numpy()
        v = (xyz_cam[:, 1] / z * fy + cy).cpu().numpy()
        u, v, opa, sc = u[valid], v[valid], opa[valid], sc[valid]
        in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, opa = u[in_frame], v[in_frame], opa[in_frame]
        sc = sc[in_frame]
        out = np.stack([u, v, opa, sc[:, 0], sc[:, 1]], axis=1)
        return out


def _interactive_refine_inspector(
    render_bgr: np.ndarray,
    silhouette: np.ndarray,
    norm_mag: np.ndarray,
    depth_error: np.ndarray,
    prune_list: np.ndarray,
    add_mask: np.ndarray = None,
    title_suffix: str = "",
):
    """两个 OpenCV 窗口：左图鼠标显示像素 silhouette/norm_mag/depth_error，满足添加条件的像素上色；右图显示修剪点，悬停显示 opacity/scale。按 q 关闭并继续。"""
    H, W = render_bgr.shape[:2]
    prune_radius = 15

    display_add = render_bgr.copy()
    if add_mask is not None and add_mask.any():
        # 满足添加条件的像素叠一层半透明绿色 (BGR)
        overlay = display_add.copy()
        overlay[add_mask] = [0, 255, 0]
        cv2.addWeighted(overlay, 0.4, display_add, 0.6, 0, dst=display_add)
    display_prune = render_bgr.copy()
    for i in range(prune_list.shape[0]):
        u, v = int(round(prune_list[i, 0])), int(round(prune_list[i, 1]))
        if 0 <= u < W and 0 <= v < H:
            cv2.circle(display_prune, (u, v), 3, (0, 0, 255), -1)

    state = {"mx": -1, "my": -1, "prune_idx": -1}

    def on_mouse_add(event, x, y, _u, _v):
        if event == cv2.EVENT_MOUSEMOVE:
            state["mx"], state["my"] = x, y
            state["prune_idx"] = -1

    def on_mouse_prune(event, x, y, _u, _v):
        if event == cv2.EVENT_MOUSEMOVE and prune_list.shape[0] > 0:
            state["mx"], state["my"] = x, y
            dx = prune_list[:, 0] - x
            dy = prune_list[:, 1] - y
            dist = np.sqrt(dx * dx + dy * dy)
            idx = np.argmin(dist)
            state["prune_idx"] = idx if dist[idx] <= prune_radius else -1

    cv2.namedWindow("add_metrics" + title_suffix)
    cv2.namedWindow("prune_points" + title_suffix)
    cv2.setMouseCallback("add_metrics" + title_suffix, on_mouse_add)
    cv2.setMouseCallback("prune_points" + title_suffix, on_mouse_prune)

    while True:
        frame_add = display_add.copy()
        frame_prune = display_prune.copy()

        mx, my = state["mx"], state["my"]
        if 0 <= mx < W and 0 <= my < H:
            sil = float(silhouette[my, mx])
            nm = float(norm_mag[my, mx])
            de = float(depth_error[my, mx])
            txt = f"({mx},{my}) sil={sil:.4f} norm_mag={nm:.4f} depth_err={de:.4f}"
            cv2.putText(frame_add, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.circle(frame_add, (mx, my), 4, (0, 255, 0), 1)

        idx = state["prune_idx"]
        if idx >= 0:
            row = prune_list[idx]
            u, v, opa, s0, s1 = row[0], row[1], row[2], row[3], row[4]
            txt = f"opacity={opa:.4f} scale0={s0:.6f} scale1={s1:.6f}"
            cv2.putText(frame_prune, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.circle(frame_prune, (int(round(u)), int(round(v))), 8, (0, 255, 0), 2)

        cv2.putText(frame_add, "Green = add condition | Press 'q' to close", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame_prune, "Hover near red dot: opacity/scale | Press 'q' to continue", (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("add_metrics" + title_suffix, frame_add)
        cv2.imshow("prune_points" + title_suffix, frame_prune)
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q"):
            break
    cv2.destroyAllWindows()


def _refine_one_step(gaussian_model, training_frame, opt_params, enable_exposure, enable_sh, global_iter):
    """One optimization step on a single frame. Returns (render_dict, est_w2c, gt_color, gt_depth)."""
    gt_color = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
    gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
    render_settings, est_w2c = training_frame["render_settings"]
    renderer_type = training_frame.get("renderer_type", "2dgs")
    render_dict = render_gaussian_model(gaussian_model, render_settings, est_w2c, renderer_type=renderer_type)
    rendered_color = render_dict["color"].permute(1, 2, 0)
    rendered_depth = render_dict["depth"]
    if enable_exposure and training_frame.get("exposure_ab") is not None:
        rendered_color = torch.clamp(
            rendered_color * torch.exp(training_frame["exposure_ab"][0, 0]) + training_frame["exposure_ab"][0, 1], 0, 1.0)
    reg_loss = isotropic_loss(gaussian_model.get_scaling())
    depth_mask = (gt_depth > 0)
    color_loss = (1.0 - opt_params.lambda_dssim) * l1_loss(
        rendered_color[depth_mask, :], gt_color[depth_mask, :]
    ) + opt_params.lambda_dssim * (1.0 - ssim(rendered_color, gt_color))
    depth_loss = l1_loss(rendered_depth[:, depth_mask], gt_depth[depth_mask])
    total_loss = color_loss + depth_loss + reg_loss
    if renderer_type != "3dgs":
        dist_loss = 1000 * render_dict["rend_dist"].mean()
        rend_normal = render_dict["rend_normal"]
        surf_normal = render_dict["normal"]
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = 0.05 * normal_error.mean()
        total_loss = total_loss + dist_loss + normal_loss
    total_loss.backward()
    if enable_sh and global_iter > 0 and global_iter % 1000 == 0:
        gaussian_model.oneupSHdegree()
    return render_dict, est_w2c, gt_color, gt_depth


def _prune_gaussians(
    gaussian_model: GaussianModel,
    prune_opacity_threshold: float,
    max_scale_ratio_to_mode: float,
    max_distance_from_origin: float = None,
) -> int:
    """Prune low-opacity, oversized, and too-far Gaussians. No add. Returns total number removed."""
    with torch.no_grad():
        n_before = gaussian_model.get_size()
        if n_before == 0:
            return 0
        prune_mask = (gaussian_model.get_opacity() < prune_opacity_threshold).squeeze()
        n_valid = (~prune_mask).sum().item()
        if n_valid > 0:
            gaussian_model.prune_points(prune_mask)
        if max_scale_ratio_to_mode is not None and max_scale_ratio_to_mode > 0:
            prune_oversized_gaussians(gaussian_model, max_scale_ratio=max_scale_ratio_to_mode, verbose=False)
        if max_distance_from_origin is not None and max_distance_from_origin > 0:
            xyz = gaussian_model.get_xyz()
            dist = torch.sqrt((xyz ** 2).sum(dim=1))
            far_mask = dist > max_distance_from_origin
            if far_mask.any():
                gaussian_model.prune_points(far_mask)
        n_after = gaussian_model.get_size()
        n_removed = n_before - n_after
        if n_removed > 0:
            print(f"  [refine] 减少 {n_removed} 个高斯，当前共 {n_after} 个")
        return n_removed


def _add_gaussians_from_frame(
    gaussian_model: GaussianModel,
    render_dict: dict,
    training_frame: dict,
    est_w2c: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    depth_error_median_k: float,
    add_gaussians_max_points: int,
    new_gaussian_scale_max: float,
    add_valid_depth_min: float,
    add_depth_valid_for_error_min: float,
    add_silhouette_threshold: float,
    add_norm_mag_threshold: float,
    max_distance_from_origin: float = None,
) -> int:
    """Add new Gaussians from current frame (alpha/depth/normal mask). Returns number added."""
    with torch.no_grad():
        gt_color_add = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
        gt_depth_add = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
        if gt_depth_add.dim() == 2:
            gt_depth_add = gt_depth_add.unsqueeze(0)
        valid_depth_mask = (gt_depth_add > add_valid_depth_min).squeeze(0)
        silhouette = render_dict["alpha"]
        non_presence_sil = (silhouette < add_silhouette_threshold).squeeze(0)
        render_depth_add = render_dict["depth"].squeeze(0)
        gt_d = gt_depth_add.squeeze(0)
        depth_error = torch.abs(gt_d - render_depth_add) * (gt_d > add_depth_valid_for_error_min)
        valid_err = depth_error[gt_d > add_depth_valid_for_error_min]
        non_presence_depth = depth_error > (depth_error_median_k * valid_err.median() if valid_err.numel() > 0 else 1e6)
        surf_norm = render_dict.get("normal")
        if surf_norm is not None:
            norm_mag = torch.sqrt((surf_norm ** 2).sum(dim=0))
            norm_mask = norm_mag < add_norm_mag_threshold
        else:
            norm_mask = torch.zeros_like(non_presence_sil, dtype=torch.bool)
        non_presence_mask = (
            (non_presence_sil | non_presence_depth | norm_mask).reshape(-1) & valid_depth_mask.reshape(-1)
        )
        if not non_presence_mask.any():
            return 0
        w2c = est_w2c.squeeze().float().cuda()
        new_pt_cld, mean3_sq_dist = backproject_frame_to_pointcloud(
            gt_color_add, gt_d, w2c, fx, fy, cx, cy, non_presence_mask, device="cuda")
        if new_pt_cld.shape[0] == 0:
            return 0
        if max_distance_from_origin is not None and max_distance_from_origin > 0:
            dist = torch.sqrt((new_pt_cld[:, :3] ** 2).sum(dim=1))
            keep = dist <= max_distance_from_origin
            new_pt_cld = new_pt_cld[keep]
            mean3_sq_dist = mean3_sq_dist[keep]
        if new_pt_cld.shape[0] == 0:
            return 0
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
        new_opacities = inverse_sigmoid(torch.tensor(0.5, device="cuda")).expand(new_pt_cld.shape[0], 1)
        scale_val = torch.sqrt(mean3_sq_dist).clamp(min=1e-6, max=new_gaussian_scale_max)
        scale_cols = 3 if not getattr(gaussian_model, "isotropic", True) else 2
        new_scaling = torch.log(scale_val)[..., None].repeat(1, scale_cols)
        gaussian_model.densification_postfix(
            new_xyz=new_pt_cld[:, :3],
            new_rgb=new_rgb,
            new_features_dc=new_features[:, :, 0:1].transpose(1, 2).contiguous(),
            new_features_rest=new_features[:, :, 1:].transpose(1, 2).contiguous(),
            new_opacities=new_opacities,
            new_scaling=new_scaling,
            new_rotation=new_rots,
        )
        return new_pt_cld.shape[0]


def refine_global_map(pt_cloud: o3d.geometry.PointCloud, training_frames, max_iterations: int,
                      export_refine_mesh=True, output_dir=".",
                      len_frames=None, o3d_intrinsic=None, enable_sh=True, enable_exposure=False,
                      max_scale_ratio_to_mode: float = 5.0,
                      add_gaussians_every: int = 0,
                      height=None, width=None, fx=None, fy=None, cx=None, cy=None,
                      add_gaussians_max_points: int = 5000,
                      depth_error_median_k: float = 10.0,
                      prune_opacity_threshold: float = 0.1,
                      new_gaussian_scale_max: float = 0.02,
                      add_valid_depth_min: float = 0.1,
                      add_depth_valid_for_error_min: float = 0.05,
                      add_silhouette_threshold: float = 0.6,
                      add_norm_mag_threshold: float = 0.9,
                      max_distance_from_origin: float = 15.0,
                      refine_mode: str = "shuffle",
                      per_frame_iters: int = 100,
                      prune_add_interval: int = 20,
                      refine_vis_interval: int = 0,
                      refine_vis_save: bool = True,
                      refine_vis_interactive: bool = False,
                      renderer_type: str = "2dgs") -> GaussianModel:
    """Refines a global map. Two modes:
    - shuffle: training_frames is an iterator; max_iterations total steps; random frame order.
    - sequential_per_frame: training_frames is indexable (e.g. RenderFrames); outer loop over
      keyframes in order, inner loop per_frame_iters per keyframe; every prune_add_interval steps
      do prune + add from current frame (mapper-style).
    """
    has_intrinsics = all(x is not None for x in (height, width, fx, fy, cx, cy))
    opt_params = OptimizationParams(ArgumentParser(description="Training script parameters"))
    isotropic = renderer_type != "3dgs"
    gaussian_model = GaussianModel(3, isotropic=isotropic)
    gaussian_model.active_sh_degree = 0
    if pt_cloud is None:
        output_mesh_path = output_dir / "mesh" / "cleaned_mesh.ply"
        if not output_mesh_path.exists():
            raise FileNotFoundError(
                f"Mesh file not found: {output_mesh_path}. "
                "Run reconstruction eval first to generate cleaned_mesh.ply, or use init_from='splats' with merge_submaps."
            )
        output_mesh = o3d.io.read_triangle_mesh(str(output_mesh_path))
        if len(output_mesh.vertices) == 0:
            raise ValueError(
                f"Loaded mesh from {output_mesh_path} has no vertices. "
                "Reconstruction may have produced an empty mesh."
            )
        pcd = o3d.geometry.PointCloud()
        pcd.points = output_mesh.vertices
        pcd.colors = output_mesh.vertex_colors
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        pcd = BasicPointCloud(points=np.asarray(pcd.points), colors=np.asarray(pcd.colors))
        gaussian_model.create_from_pcd(pcd, 1.0)
        gaussian_model.training_setup(opt_params)
    else:
        gaussian_model.training_setup(opt_params)
        gaussian_model.add_points(pt_cloud)

    if max_scale_ratio_to_mode is not None and max_scale_ratio_to_mode > 0:
        prune_oversized_gaussians(gaussian_model, max_scale_ratio=max_scale_ratio_to_mode)
    if max_distance_from_origin is not None and max_distance_from_origin > 0:
        _prune_gaussians(gaussian_model, 0.0, None, max_distance_from_origin=max_distance_from_origin)

    print(f"[refine] 初始共 {gaussian_model.get_size()} 个高斯")
    do_vis = refine_vis_interval > 0
    vis_dir = None
    if do_vis and refine_vis_save and output_dir is not None:
        vis_dir = Path(output_dir) / "refine_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)

    if refine_mode == "sequential_per_frame":
        # Mapper-style: outer = keyframes in order, inner = per_frame_iters; every prune_add_interval: prune + add
        n_frames = len(training_frames) if len_frames is None else len_frames
        total_iters = n_frames * per_frame_iters
        global_iter = 0
        pbar = tqdm(total=total_iters, desc="Refinement (sequential per frame)")
        for frame_idx in range(n_frames):
            training_frame = training_frames[frame_idx]
            for inner_iter in range(per_frame_iters):
                torch.cuda.empty_cache()
                gaussian_model.update_learning_rate(global_iter)
                gaussian_model.optimizer.zero_grad(set_to_none=True)
                render_dict, est_w2c, _, _ = _refine_one_step(
                    gaussian_model, training_frame, opt_params, enable_exposure, enable_sh, global_iter)
                with torch.no_grad():
                    gaussian_model.optimizer.step()
                    gaussian_model.optimizer.zero_grad(set_to_none=True)
                if do_vis and global_iter % refine_vis_interval == 0:
                    save_id = f"f{frame_idx:04d}_i{inner_iter:04d}" if refine_vis_save and vis_dir else None
                    _vis_refine_frame(render_dict, training_frame, save_id=save_id, save_path=str(vis_dir) if vis_dir else None)
                    if refine_vis_interactive and has_intrinsics:
                        with torch.no_grad():
                            sil, nm, de = _compute_pixel_metrics_for_vis(render_dict, training_frame)
                            rgb = render_dict["color"].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                            render_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            prune_list = _compute_prune_projection_for_vis(
                                gaussian_model, prune_opacity_threshold, est_w2c, fx, fy, cx, cy, height, width,
                            )
                            add_mask = _compute_add_mask_for_vis(
                                render_dict, training_frame, depth_error_median_k,
                                add_valid_depth_min, add_depth_valid_for_error_min,
                                add_silhouette_threshold, add_norm_mag_threshold,
                            )
                            _interactive_refine_inspector(render_bgr, sil, nm, de, prune_list, add_mask=add_mask, title_suffix=f"_{frame_idx}_{inner_iter}")
                if inner_iter > 0 and inner_iter % prune_add_interval == 0:
                    _prune_gaussians(gaussian_model, prune_opacity_threshold, max_scale_ratio_to_mode, max_distance_from_origin)
                    # 最后一次迭代只修剪不添加，避免新加高斯没有后续优化
                    if has_intrinsics and inner_iter < per_frame_iters - 1:
                        n_added = _add_gaussians_from_frame(
                            gaussian_model, render_dict, training_frame, est_w2c,
                            fx, fy, cx, cy, depth_error_median_k, add_gaussians_max_points, new_gaussian_scale_max,
                            add_valid_depth_min, add_depth_valid_for_error_min,
                            add_silhouette_threshold, add_norm_mag_threshold,
                            max_distance_from_origin,
                        )
                        if n_added > 0:
                            print(f"  [refine] 新增 {n_added} 个高斯，当前共 {gaussian_model.get_size()} 个")
                global_iter += 1
                pbar.update(1)
        pbar.close()
        print(f"[refine] 结束共 {gaussian_model.get_size()} 个高斯")
    else:
        # Original shuffle mode: iterator, max_iterations
        do_add_gaussians = add_gaussians_every > 0 and has_intrinsics
        iteration = 0
        for iteration in tqdm(range(max_iterations), desc="Refinement"):
            torch.cuda.empty_cache()
            training_frame = next(training_frames)
            gaussian_model.update_learning_rate(iteration)
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            render_dict, est_w2c, _, _ = _refine_one_step(
                gaussian_model, training_frame, opt_params, enable_exposure, enable_sh, iteration)
            with torch.no_grad():
                if do_vis and iteration % refine_vis_interval == 0:
                    save_id = f"iter{iteration:06d}" if refine_vis_save and vis_dir else None
                    _vis_refine_frame(render_dict, training_frame, save_id=save_id, save_path=str(vis_dir) if vis_dir else None)
                    if refine_vis_interactive and has_intrinsics:
                        with torch.no_grad():
                            sil, nm, de = _compute_pixel_metrics_for_vis(render_dict, training_frame)
                            rgb = render_dict["color"].squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
                            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                            render_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            prune_list = _compute_prune_projection_for_vis(
                                gaussian_model, prune_opacity_threshold, est_w2c, fx, fy, cx, cy, height, width,
                            )
                            add_mask = _compute_add_mask_for_vis(
                                render_dict, training_frame, depth_error_median_k,
                                add_valid_depth_min, add_depth_valid_for_error_min,
                                add_silhouette_threshold, add_norm_mag_threshold,
                            )
                            _interactive_refine_inspector(render_bgr, sil, nm, de, prune_list, add_mask=add_mask, title_suffix=f"_iter{iteration}")
                if do_add_gaussians and iteration > 0 and iteration % add_gaussians_every == 0:
                    _prune_gaussians(gaussian_model, prune_opacity_threshold, max_scale_ratio_to_mode, max_distance_from_origin)
                    n_added = _add_gaussians_from_frame(
                        gaussian_model, render_dict, training_frame, est_w2c,
                        fx, fy, cx, cy, depth_error_median_k, add_gaussians_max_points, new_gaussian_scale_max,
                        add_valid_depth_min, add_depth_valid_for_error_min,
                        add_silhouette_threshold, add_norm_mag_threshold,
                        max_distance_from_origin,
                    )
                    if n_added > 0:
                        print(f"  [refine] 新增 {n_added} 个高斯，当前共 {gaussian_model.get_size()} 个")
                gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)
            iteration += 1
        print(f"[refine] 结束共 {gaussian_model.get_size()} 个高斯")
    
    try:
        if export_refine_mesh and len_frames is not None:
            mesh_len = len_frames
            output_dir = output_dir / "mesh" / "refined_mesh.ply"
            scale = 1.0
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=5.0 * scale / 512.0,
                sdf_trunc=0.04 * scale,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            for i in tqdm(range(mesh_len), desc="Integrating mesh"):
                if refine_mode == "sequential_per_frame":
                    training_frame = training_frames[i]
                else:
                    training_frame = next(training_frames)
                gt_color = training_frame["color"].squeeze(0) if training_frame["color"].dim() > 3 else training_frame["color"]
                gt_depth = training_frame["depth"].squeeze(0) if training_frame["depth"].dim() > 2 else training_frame["depth"]
                render_settings, estimate_w2c = training_frame["render_settings"]
                rtype = training_frame.get("renderer_type", "2dgs")
                render_dict = render_gaussian_model(gaussian_model, render_settings, estimate_w2c, renderer_type=rtype)
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
