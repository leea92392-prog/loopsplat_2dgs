#!/usr/bin/env python3
"""
Registration method verification script.

Compares GSR (Gaussian Splatting Registration) vs traditional geometric methods
(ICP, FPFH+RANSAC+ICP) for loop-closure submap registration. Reads from existing
SLAM output and does not modify the original SLAM code.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import torch

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.lc import Loop_closure
from src.entities.logger import Logger
from src.utils.io_utils import load_config

# Reuse from loop_viewer
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


def build_submap_lc_info(loop_closer: Loop_closure, submaps_paths: list, checkpoint_path: Path):
    """Build submap_lc_info by loading each submap and extracting kf_desc via NetVLAD."""
    for submap_path in submaps_paths:
        submap_id = int(submap_path.stem)
        submap_dict = torch.load(submap_path, map_location="cpu", weights_only=False)
        kf_ids = submap_dict["submap_keyframes"]
        keyframes_info = {kf_id: {"keyframe_id": kf_id} for kf_id in kf_ids}

        loop_closer.submap_id = submap_id
        loop_closer.update_submaps_info(keyframes_info)


def parse_loop_edges(lc_info: dict) -> list:
    """Extract (source_id, target_id) pairs from loop closure info."""
    if "loop_edges" in lc_info:
        return [tuple(e) for e in lc_info["loop_edges"]]
    if "loop_closure_records" in lc_info:
        return [
            (r["source_submap_id"], r["target_submap_id"])
            for r in lc_info["loop_closure_records"]
        ]
    return []


def visualize_and_save(
    source_xyz: np.ndarray,
    target_xyz: np.ndarray,
    tsfm: np.ndarray,
    method_name: str,
    output_path: Path,
    edge_label: str,
):
    """Create point clouds and optionally save visualization."""
    source_transformed = apply_transform(source_xyz, tsfm)

    pcd_src = o3d.geometry.PointCloud()
    pcd_src.points = o3d.utility.Vector3dVector(source_transformed)
    pcd_src.paint_uniform_color([1, 0, 0])  # Red = transformed source

    pcd_tgt = o3d.geometry.PointCloud()
    pcd_tgt.points = o3d.utility.Vector3dVector(target_xyz)
    pcd_tgt.paint_uniform_color([0, 1, 0])  # Green = target

    combined = pcd_src + pcd_tgt

    # Save as PLY for external viewing
    ply_path = output_path / f"edge_{edge_label}_{method_name}.ply"
    o3d.io.write_point_cloud(str(ply_path), combined)
    print(f"  Saved: {ply_path}")

    # Save individual point clouds for side-by-side
    o3d.io.write_point_cloud(str(output_path / f"edge_{edge_label}_{method_name}_source.ply"), pcd_src)
    o3d.io.write_point_cloud(str(output_path / f"edge_{edge_label}_{method_name}_target.ply"), pcd_tgt)

    return combined


def run_verification(checkpoint_path: Path, save_images: bool, lc_file: str = None):
    """Main verification logic."""
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    config["data"]["output_path"] = str(checkpoint_path)
    dataset = get_dataset(config["dataset_name"])(dataset_config=config)

    estimated_c2w = torch.load(
        checkpoint_path / "estimated_c2w.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(estimated_c2w, torch.Tensor):
        c2ws_est = estimated_c2w
    else:
        c2ws_est = torch.tensor(estimated_c2w, dtype=torch.float32)

    submaps_dir = checkpoint_path / "submaps"
    submaps_paths = sorted(submaps_dir.glob("*.ckpt"), key=lambda p: int(p.stem))

    lc_dir = checkpoint_path / "loop_closures"
    if not lc_dir.exists():
        raise FileNotFoundError(f"Loop closures dir not found: {lc_dir}")

    if lc_file:
        lc_path = checkpoint_path / "loop_closures" / lc_file
        if not lc_path.exists():
            raise FileNotFoundError(f"LC file not found: {lc_path}")
        lc_files = [lc_path]
    else:
        lc_files = sorted(lc_dir.glob("loop_closure_info_*.json"), key=lambda p: int(p.stem.split("_")[-1]))

    if not lc_files:
        raise FileNotFoundError(f"No loop closure info in {lc_dir}")

    logger = Logger(checkpoint_path, use_wandb=False)
    loop_closer = Loop_closure(config, dataset, logger)
    loop_closer.submap_path = submaps_dir
    loop_closer.submap_paths = [str(p) for p in submaps_paths]
    loop_closer.c2ws_est = c2ws_est.cuda() if torch.cuda.is_available() else c2ws_est
    loop_closer.submap_id = len(submaps_paths) - 1

    print("Building submap LC info (NetVLAD descriptors)...")
    build_submap_lc_info(loop_closer, submaps_paths, checkpoint_path)

    output_dir = checkpoint_path / "registration_verification"
    output_dir.mkdir(exist_ok=True, parents=True)

    all_results = []

    for lc_path in lc_files:
        print(f"\nProcessing {lc_path.name}...")
        with open(lc_path) as f:
            lc_info = json.load(f)

        loop_edges = parse_loop_edges(lc_info)
        if not loop_edges:
            print("  No loop edges found, skipping.")
            continue

        for source_id, target_id in loop_edges:
            edge_label = f"{source_id}_{target_id}"
            print(f"\n--- Loop edge ({source_id}, {target_id}) ---")

            try:
                submap_source = loop_closer.submap_loader(source_id, load_gaussian=True)
                submap_target = loop_closer.submap_loader(target_id, load_gaussian=True)
            except Exception as e:
                print(f"  Failed to load submaps: {e}")
                continue

            source_xyz, _ = load_submap_point_cloud(Path(loop_closer.submap_paths[source_id]))
            target_xyz, _ = load_submap_point_cloud(Path(loop_closer.submap_paths[target_id]))

            results = {"edge": edge_label, "source_id": source_id, "target_id": target_id}

            # GT reference (if available)
            try:
                gt_out = loop_closer.pairwise_registration(submap_source, submap_target, method="gt")
                gt_tsfm = gt_out["transformation"]
                results["gt_available"] = True
            except Exception:
                gt_tsfm = np.eye(4)
                results["gt_available"] = False

            # 1. Raw (before alignment)
            raw_tsfm = np.eye(4)
            if save_images:
                visualize_and_save(
                    source_xyz, target_xyz, raw_tsfm, "raw", output_dir, edge_label
                )

            # 2. GSR
            try:
                t0 = time.perf_counter()
                gsr_out = loop_closer.pairwise_registration(
                    submap_source, submap_target, method="gs_reg"
                )
                gsr_time = time.perf_counter() - t0
                gsr_tsfm = gsr_out["transformation"]
                results["gsr"] = {
                    "successful": gsr_out.get("successful", True),
                    "time_s": gsr_time,
                }
                if save_images:
                    visualize_and_save(
                        source_xyz, target_xyz, gsr_tsfm, "gsr", output_dir, edge_label
                    )
                print(f"  GSR: success={results['gsr']['successful']}, time={gsr_time:.2f}s")
            except Exception as e:
                results["gsr"] = {"successful": False, "error": str(e)}
                print(f"  GSR failed: {e}")

            torch.cuda.empty_cache()

            # 3. ICP
            try:
                t0 = time.perf_counter()
                icp_out = loop_closer.pairwise_registration(
                    submap_source, submap_target, method="icp"
                )
                icp_time = time.perf_counter() - t0
                icp_tsfm = icp_out["transformation"]
                results["icp"] = {
                    "time_s": icp_time,
                    "n_points": int(icp_out.get("n_points", 0)),
                }
                if save_images:
                    visualize_and_save(
                        source_xyz, target_xyz, icp_tsfm, "icp", output_dir, edge_label
                    )
                print(f"  ICP: time={icp_time:.2f}s")
            except Exception as e:
                results["icp"] = {"error": str(e)}
                print(f"  ICP failed: {e}")

            # 4. Robust ICP
            try:
                t0 = time.perf_counter()
                robust_out = loop_closer.pairwise_registration(
                    submap_source, submap_target, method="robust_icp"
                )
                robust_time = time.perf_counter() - t0
                robust_tsfm = robust_out["transformation"]
                results["robust_icp"] = {
                    "time_s": robust_time,
                    "fitness": robust_out.get("fitness"),
                    "inlier_rmse": robust_out.get("inlier_rmse"),
                }
                if save_images:
                    visualize_and_save(
                        source_xyz,
                        target_xyz,
                        robust_tsfm,
                        "robust_icp",
                        output_dir,
                        edge_label,
                    )
                print(f"  Robust ICP: time={robust_time:.2f}s, fitness={results['robust_icp'].get('fitness')}")
            except Exception as e:
                results["robust_icp"] = {"error": str(e)}
                print(f"  Robust ICP failed: {e}")

            all_results.append(results)

            del submap_source, submap_target
            torch.cuda.empty_cache()

    summary_path = output_dir / "comparison_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSummary saved to {summary_path}")
    print("Use Open3D or MeshLab to open the PLY files for visual comparison.")
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare GSR vs geometric registration for loop-closure submaps."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="output_miniba/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household—improve",
        help="Path to SLAM output directory",
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        default=True,
        help="Save PLY point clouds for visualization (default: True)",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Disable saving PLY files",
    )
    parser.add_argument(
        "--lc_file",
        type=str,
        default=None,
        help="Specific loop closure JSON file (e.g. loop_closure_info_1.json)",
    )
    args = parser.parse_args()

    save_images = args.save_images and not args.no_save
    run_verification(
        checkpoint_path=Path(args.checkpoint_path),
        save_images=save_images,
        lc_file=args.lc_file,
    )


if __name__ == "__main__":
    main()
