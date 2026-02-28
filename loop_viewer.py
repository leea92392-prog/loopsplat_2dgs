"""Visualize point clouds of loop-closure submaps to check overlap/registration quality."""
import json
import numpy as np
import open3d as o3d
from pathlib import Path
import torch


def load_submap_point_cloud(submap_path: Path):
    """Load xyz and rgb from submap ckpt, return as numpy arrays."""
    ckpt = torch.load(submap_path, map_location="cpu", weights_only=False)
    gp = ckpt["gaussian_params"]
    xyz = gp["xyz"].numpy()
    rgb = gp.get("rgb", None)
    if rgb is not None:
        rgb = rgb.numpy()
        if rgb.ndim == 3:
            rgb = rgb.reshape(-1, 3)
        rgb = np.clip(rgb, 0, 1).astype(np.float64)
    else:
        rgb = np.ones((len(xyz), 3)) * 0.5
    return xyz, rgb


def apply_transform(xyz: np.ndarray, tsfm: np.ndarray) -> np.ndarray:
    """Apply 4x4 transform to Nx3 points."""
    ones = np.ones((xyz.shape[0], 1))
    xyz_h = np.hstack([xyz, ones])
    return (tsfm @ xyz_h.T).T[:, :3]


def main():
    base = Path("output_miniba/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household—improve")
    submaps_dir = base / "submaps"
    lc_path = base / "loop_closures" / "loop_closure_info_1.json"

    with open(lc_path) as f:
        lc_info = json.load(f)

    loop_edges = lc_info["loop_edges"]
    corrections = {c["submap_id"]: np.array(c["correct_tsfm"]) for c in lc_info["corrections"]}

    # Submap IDs involved in loop closure (1, 2, 35)
    submap_ids = sorted(set(s for edge in loop_edges for s in edge))

    # Colors: submap 1=red, 2=green, 35=blue
    colors = {1: [1, 0, 0], 2: [0, 1, 0], 35: [0, 0, 1]}

    pcds_raw = []
    pcds_corrected = []

    for sid in submap_ids:
        ckpt_path = submaps_dir / f"{sid:06d}.ckpt"
        xyz, rgb = load_submap_point_cloud(ckpt_path)

        # Raw (no PGO correction)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
        pcds_raw.append((pcd, sid))

        # Corrected (apply PGO)
        tsfm = corrections.get(sid, np.eye(4))
        xyz_corr = apply_transform(xyz, tsfm)
        pcd_c = o3d.geometry.PointCloud()
        pcd_c.points = o3d.utility.Vector3dVector(xyz_corr)
        pcd_c.colors = o3d.utility.Vector3dVector(rgb)
        pcds_corrected.append((pcd_c, sid))

    # Option A: Use original rgb (to see scene appearance)
    print("Visualizing RAW (before PGO) - overlapping region should show if drift exists")
    o3d.visualization.draw_geometries(
        [p for p, _ in pcds_raw],
        window_name="Loop submaps: RAW (1=red tint, 2=green, 35=blue via rgb)",
    )

    # Option B: Paint by submap for clearer overlap check
    pcds_raw_tinted = []
    for pcd, sid in pcds_raw:
        pts = np.asarray(pcd.points)
        n = len(pts)
        c = np.tile(colors.get(sid, [0.5, 0.5, 0.5]), (n, 1))
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.colors = o3d.utility.Vector3dVector(c)
        pcds_raw_tinted.append(p)

    print("Visualizing RAW with submap colors: Red=submap1, Green=submap2, Blue=submap35")
    o3d.visualization.draw_geometries(
        pcds_raw_tinted,
        window_name="Loop submaps: by submap color (check overlap)",
    )

    # Corrected
    pcds_corr_tinted = []
    for pcd, sid in pcds_corrected:
        pts = np.asarray(pcd.points)
        n = len(pts)
        c = np.tile(colors.get(sid, [0.5, 0.5, 0.5]), (n, 1))
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts)
        p.colors = o3d.utility.Vector3dVector(c)
        pcds_corr_tinted.append(p)

    print("Visualizing CORRECTED (after PGO)")
    o3d.visualization.draw_geometries(
        pcds_corr_tinted,
        window_name="Loop submaps: after PGO",
    )


if __name__ == "__main__":
    main()