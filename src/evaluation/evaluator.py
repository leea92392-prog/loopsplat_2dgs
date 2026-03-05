""" This module is responsible for evaluating rendering, trajectory and reconstruction metrics"""
import traceback
from argparse import ArgumentParser
from copy import deepcopy
from itertools import cycle
from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import torchvision
import json
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision.utils import save_image
from tqdm import tqdm

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.evaluation.evaluate_merged_map import (RenderFrames, merge_submaps,
                                                refine_global_map,
                                                render_frames_collate_fn)
from src.evaluation.evaluate_reconstruction import evaluate_reconstruction, clean_mesh
from src.evaluation.evaluate_trajectory import evaluate_trajectory
from src.utils.io_utils import load_config, save_dict_to_json, log_metrics_to_wandb
from src.utils.mapper_utils import calc_psnr
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, setup_seed, 
                             torch2np, filter_depth_outliers)
from src.utils.vis_utils import show_render_result
from src.utils.utils import depth_to_normal

class Evaluator(object):

    def __init__(self, checkpoint_path, config_path, config=None, save_render=True) -> None:
        if config is None:
            self.config = load_config(config_path)
        else:
            self.config = config
        setup_seed(self.config["seed"])

        self.checkpoint_path = Path(checkpoint_path)
        self.use_wandb = self.config["use_wandb"]
        self.device = "cuda"
        self.dataset = get_dataset(self.config["dataset_name"])(dataset_config=self.config)
        self.scene_name = self.config["data"]["scene_name"]
        self.dataset_name = self.config["dataset_name"]
        self.gt_poses = np.array(self.dataset.poses)
        self.fx, self.fy = self.dataset.intrinsics[0, 0], self.dataset.intrinsics[1, 1]
        self.cx, self.cy = self.dataset.intrinsics[0, 2], self.dataset.intrinsics[1, 2]
        self.width, self.height = self.dataset.width, self.dataset.height
        self.save_render = save_render
        if self.save_render:
            self.render_path = self.checkpoint_path / "rendered_imgs"
            self.render_path.mkdir(exist_ok=True, parents=True)

        self.estimated_c2w = torch2np(torch.load(self.checkpoint_path / "estimated_c2w.ckpt", map_location=self.device))
        self.init_c2w = None
        if (self.checkpoint_path / "init_c2w.ckpt").exists():
            self.init_c2w = torch2np(torch.load(self.checkpoint_path / "init_c2w.ckpt", map_location=self.device))
        self.submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*')))
        self.exposures_ab = None
        if (self.checkpoint_path / "exposures_ab.ckpt").exists():
            self.exposures_ab = torch2np(torch.load(self.checkpoint_path / "exposures_ab.ckpt", map_location=self.device))
            print(f"Loaded trained exposures paramters for scene {self.scene_name}")

    def run_trajectory_eval(self):
        """ Evaluates the estimated trajectory """
        print("Running trajectory evaluation...")
        evaluate_trajectory(self.estimated_c2w,self.init_c2w ,self.gt_poses, self.checkpoint_path)

    def run_rendering_eval(self):
        """ Renders the submaps and global splats and evaluates the PSNR, LPIPS, SSIM and depth L1 metrics."""
        print("Running rendering evaluation...")
        psnr, lpips, ssim, depth_l1 = [], [], [], []
        color_transform = torchvision.transforms.ToTensor()
        lpips_model = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(self.device)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))

        submaps_paths = sorted(
            list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:

                _, gt_color, gt_depth, _,_ = self.dataset[keyframe_id]
                gt_color = color_transform(gt_color).to(self.device)
                gt_depth = np2torch(gt_depth).to(self.device)

                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_settings,est_w2c = get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c)
                #计算真实深度图的法向量图
                gt_normal = depth_to_normal(render_settings,gt_depth.unsqueeze(0))
                
                render_dict = render_gaussian_model(
                    gaussian_model, render_settings, est_w2c)
                show_render_result(render_rgb=render_dict["color"], render_depth=render_dict["depth"],
                               gt_depth=gt_depth,gt_rgb=gt_color,render_normal=render_dict["normal"],gt_normal=gt_normal,
                               save_id=keyframe_id,save_path=self.render_path)
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                if self.save_render:
                    torchvision.utils.save_image(
                        rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                mse_loss = torch.nn.functional.mse_loss(
                    rendered_color, gt_color)
                psnr_value = (-10. * torch.log10(mse_loss)).item()
                lpips_value = lpips_model(
                    rendered_color[None], gt_color[None]).item()
                ssim_value = ms_ssim(
                    rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                depth_l1_value = torch.abs(
                    (rendered_depth - gt_depth)).mean().item()

                psnr.append(psnr_value)
                lpips.append(lpips_value)
                ssim.append(ssim_value)
                depth_l1.append(depth_l1_value)

        num_frames = len(psnr)
        metrics = {
            "psnr": sum(psnr) / num_frames,
            "lpips": sum(lpips) / num_frames,
            "ssim": sum(ssim) / num_frames,
            "depth_l1_train_view": sum(depth_l1) / num_frames,
            "num_renders": num_frames
        }
        save_dict_to_json(metrics, "rendering_metrics.json",
                          directory=self.checkpoint_path)

        x = list(range(len(psnr)))
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].plot(x, psnr, label="PSNR")
        axs[0].legend()
        axs[0].set_title("PSNR")
        axs[1].plot(x, ssim, label="SSIM")
        axs[1].legend()
        axs[1].set_title("SSIM")
        axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
        axs[2].legend()
        axs[2].set_title("Depth L1 Render")
        plt.tight_layout()
        plt.savefig(str(self.checkpoint_path /
                    "rendering_metrics.png"), dpi=300)
        print(metrics)

    def run_reconstruction_eval(self):
        """ Reconstructs the mesh, evaluates it, render novel view depth maps from it, and evaluates them as well """
        print("Running reconstruction evaluation...")

        (self.checkpoint_path / "mesh").mkdir(exist_ok=True, parents=True)
        opt_settings = OptimizationParams(ArgumentParser(
            description="Training script parameters"))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        scale = 1.0
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            # voxel_length=10.0 * scale / 512.0,
            # sdf_trunc=0.08 * scale,
            voxel_length=0.01,
            sdf_trunc=0.05,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        trajectory_points = []
        submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            gaussian_model = GaussianModel()
            gaussian_model.training_setup(opt_settings)
            gaussian_model.restore_from_params(
                submap["gaussian_params"], opt_settings)

            for keyframe_id in submap["submap_keyframes"]:
                estimate_c2w = self.estimated_c2w[keyframe_id]
                estimate_w2c = np.linalg.inv(estimate_c2w)
                render_settings,est_w2c = get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c)
                render_dict = render_gaussian_model(
                    gaussian_model, render_settings, est_w2c)
                rendered_color, rendered_depth = render_dict["color"].detach(
                ), render_dict["depth"][0].detach()
                rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)

                rendered_color = (
                    torch2np(rendered_color.permute(1, 2, 0)) * 255).astype(np.uint8)
                rendered_depth = torch2np(rendered_depth)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(rendered_color)),
                    o3d.geometry.Image(rendered_depth),
                    depth_scale=scale,
                    # depth_trunc=30,
                    depth_trunc=3.0,
                    convert_rgb_to_intensity=False)
                volume.integrate(rgbd, intrinsic, estimate_w2c)
                trajectory_points.append(estimate_c2w[:3, 3])
            # 关键：释放当前子图的 GPU 内存  
            del gaussian_model  
            del submap  
            torch.cuda.empty_cache()  
        print("Extracting mesh from TSDF volume...")
        o3d_mesh = volume.extract_triangle_mesh()
        compensate_vector = (-0.0 * scale / 512.0, 2.5 *
                             scale / 512.0, -2.5 * scale / 512.0)
        o3d_mesh = o3d_mesh.translate(compensate_vector)
        o3d_mesh = clean_mesh(o3d_mesh)
        file_name = self.checkpoint_path / "mesh" / "cleaned_mesh.ply"
        o3d.io.write_triangle_mesh(str(file_name), o3d_mesh)
        print(f"Reconstructed mesh saved to {file_name}")

        trajectory_pcd = o3d.geometry.PointCloud()
        trajectory_pcd.points = o3d.utility.Vector3dVector(trajectory_points)
        o3d.visualization.draw_geometries([o3d_mesh])

        o3d_pcd = volume.extract_point_cloud()
        o3d.io.write_point_cloud(str(self.checkpoint_path / "mesh" / "reconstructed_point_cloud.ply"), o3d_pcd)
        print("Reconstructed point cloud saved.")
        # 可视化点云
        o3d.visualization.draw_geometries([o3d_pcd])
        # if self.config["dataset_name"] == "replica":
        #     evaluate_reconstruction(file_name,
        #                             f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_mesh_cull_virt_cams.ply",
        #                             f"data/Replica-SLAM/cull_replica/{self.scene_name}/gt_pc_unseen.npy",
        #                             self.checkpoint_path)
    


    # def run_mesh_vis(self):
    #     """ vislize the mesh"""
    #     print("Running vis...")
    #     trajectory_points = []
    #     submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
    #     for submap_path in tqdm(submaps_paths):
    #         submap = torch.load(submap_path, map_location=self.device)
    #         for keyframe_id in submap["submap_keyframes"]:
    #             estimate_c2w = self.estimated_c2w[keyframe_id]
    #             estimate_w2c = np.linalg.inv(estimate_c2w)
    #             trajectory_points.append(estimate_c2w[:3, 3])
    #     file_name = self.checkpoint_path / "mesh" / "cleaned_mesh.ply"
    #     o3d_mesh = o3d.io.read_triangle_mesh(str(file_name))
    #     trajectory_lines = []
    #     for i in range(len(trajectory_points) - 1):
    #         trajectory_lines.append([i, i + 1])
    #     trajectory_line_set = o3d.geometry.LineSet()
    #     trajectory_line_set.points = o3d.utility.Vector3dVector(trajectory_points)
    #     trajectory_line_set.lines = o3d.utility.Vector2iVector(trajectory_lines)
    #     # 设置轨迹的颜色
    #     colors = [[0, 1, 0] for _ in range(len(trajectory_lines))]  # 绿色
    #     trajectory_line_set.colors = o3d.utility.Vector3dVector(colors)
    #     # 定义回调函数
    #     def save_image(vis):
    #         global image_counter
    #         screenshots_dir = self.checkpoint_path / "mesh11"
    #         if not os.path.exists(screenshots_dir):
    #             os.makedirs(screenshots_dir)
    #         image_path = screenshots_dir/f"visualization_{image_counter}.png"
    #         vis.capture_screen_image(str(image_path))
    #         print(f"Saved {image_path}")
    #         image_counter += 1
    #         return False
    #     global image_counter
    #     image_counter = 0
    #     # 可视化重建的网格和轨迹线段
    #     # 从网格采样点云（均匀采样，例如 100000 个点）
    #     o3d_pcd = o3d_mesh.sample_points_uniformly(number_of_points=60000)
    #     # 保存点云
    #     o3d.io.write_point_cloud(str(self.checkpoint_path / "mesh" / "reconstructed_pcd.ply"), o3d_pcd)
    #     print("Reconstructed point cloud saved.")
    #     # 可视化
    #     o3d.visualization.draw_geometries([o3d_pcd])
    #     vis = o3d.visualization.VisualizerWithKeyCallback()
    #     vis.create_window()
    #     vis.add_geometry(o3d_mesh)
    #     opt = vis.get_render_option()
    #     opt.background_color = np.asarray([1, 1, 1])
    #     # opt.line_width = 5.0
    #     vis.add_geometry(trajectory_line_set)
    #     # 注册按键回调函数
    #     vis.register_key_callback(ord("S"), save_image)
    #     vis.run()
    #     vis.destroy_window()

    def run_mesh_vis(self):
        """ visualize the mesh with trajectory """
        print("Running visualization...")
        
        # 收集轨迹点
        trajectory_points = []
        submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
        for submap_path in tqdm(submaps_paths):
            submap = torch.load(submap_path, map_location=self.device)
            for keyframe_id in submap["submap_keyframes"]:
                estimate_c2w = self.estimated_c2w[keyframe_id]
                trajectory_points.append(estimate_c2w[:3, 3])
        
        # 读取网格
        file_name = self.checkpoint_path / "mesh" / "cleaned_mesh.ply"
        o3d_mesh = o3d.io.read_triangle_mesh(str(file_name))
        
        # ==== 关键修改：创建圆柱体轨迹 ====
        trajectory_cylinders = o3d.geometry.TriangleMesh()
        
        # 轨迹颜色 (绿色)
        trajectory_color = [0, 1, 0]  # RGB 绿色
        
        # 轨迹半径 (控制线宽)
        trajectory_radius = 0.01  # 增大此值使轨迹更粗
        
        for i in range(len(trajectory_points) - 1):
            start = trajectory_points[i]
            end = trajectory_points[i + 1]
            segment_length = np.linalg.norm(end - start)
            
            # 创建两点之间的圆柱体
            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=trajectory_radius,
                height=segment_length,
                resolution=20,  # 圆柱体侧面细分程度
                split=1
            )
            
            # 计算方向向量
            direction = end - start
            direction_normalized = direction / segment_length
            
            # 计算旋转使圆柱体朝向正确方向
            # 默认圆柱体沿Y轴方向，需要旋转到实际方向
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(z_axis, direction_normalized)
            
            if np.linalg.norm(rotation_axis) > 1e-6:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
                rotation_angle = np.arccos(np.dot(z_axis, direction_normalized))
                rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                    rotation_axis * rotation_angle
                )
            else:
                # 方向平行于Z轴时不需要旋转
                rotation_matrix = np.eye(3)
            
            # 应用旋转和平移
            cylinder.rotate(rotation_matrix, center=[0, 0, 0])
            cylinder.translate((start + end) / 2)
            
            # 设置颜色
            cylinder.paint_uniform_color(trajectory_color)
            
            # 添加到轨迹集合
            trajectory_cylinders += cylinder
        
        # 回调函数和计数器
        global image_counter
        image_counter = 0
        
        def save_image(vis):
            global image_counter
            screenshots_dir = self.checkpoint_path / "mesh11"
            if not os.path.exists(screenshots_dir):
                os.makedirs(screenshots_dir)
            image_path = screenshots_dir/f"visualization_{image_counter}.png"
            vis.capture_screen_image(str(image_path))
            print(f"Saved {image_path}")
            image_counter += 1
            return False
        
        # 创建可视化器
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        
        # 添加网格和轨迹
        vis.add_geometry(o3d_mesh)
        vis.add_geometry(trajectory_cylinders)
        
        # 设置背景颜色为白色
        render_opt = vis.get_render_option()
        render_opt.background_color = np.asarray([1, 1, 1])  # 白色背景
        
        # 注册按键回调函数
        vis.register_key_callback(ord("S"), save_image)
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
        
        # 从网格采样点云（均匀采样）
        o3d_pcd = o3d_mesh.sample_points_uniformly(number_of_points=60000)
        o3d.io.write_point_cloud(str(self.checkpoint_path / "mesh" / "reconstructed_pcd.ply"), o3d_pcd)
        print("Reconstructed point cloud saved.")
        
        # 可视化点云
        o3d.visualization.draw_geometries([o3d_pcd])

    def run_global_map_eval(self, init_from='mesh'):
        """ Merges the map, evaluates it over training and novel views 
        
        Args:
            init_from (str, optional): 'mesh' or 'splats'. Initialization method for the global splats. Defaults to mesh vertices reconstructed before.
        """
        print("Running global map evaluation...")

        render_frames_dataset = RenderFrames(
            self.dataset, self.estimated_c2w, self.height, self.width, self.fx, self.fy, self.exposures_ab)
        refine_cfg = self.config.get("refine_global_map", {})
        refine_mode = refine_cfg.get("refine_mode", "shuffle")

        if refine_mode == "sequential_per_frame":
            # Mapper-style: keyframes in order; per_frame_iters per keyframe; prune+add every prune_add_interval
            training_frames = render_frames_dataset
            len_frames = len(render_frames_dataset)
        else:
            training_frames = DataLoader(
                render_frames_dataset, batch_size=1, shuffle=True, collate_fn=render_frames_collate_fn)
            len_frames = len(training_frames)
            training_frames = cycle(training_frames)

        merge_radius = self.config.get("lc", {}).get("merge_radius", 0.01)
        merged_cloud = merge_submaps(self.submaps_paths, radius=merge_radius) if init_from == 'splats' else None

        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        add_gaussians_every = int(refine_cfg.get("add_gaussians_every", 500))
        per_frame_iters = int(refine_cfg.get("per_frame_iters", 100))
        prune_add_interval = int(refine_cfg.get("prune_add_interval", 20))

        refined_merged_gaussian_model = refine_global_map(
            merged_cloud, training_frames, max_iterations=9000, export_refine_mesh=False,
            output_dir=self.checkpoint_path, len_frames=len_frames,
            o3d_intrinsic=intrinsic,
            add_gaussians_every=add_gaussians_every,
            height=self.height, width=self.width, fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy,
            add_gaussians_max_points=int(refine_cfg.get("add_gaussians_max_points", 5000)),
            depth_error_median_k=float(refine_cfg.get("depth_error_median_k", 10.0)),
            prune_opacity_threshold=float(refine_cfg.get("prune_opacity_threshold", 0.1)),
            max_scale_ratio_to_mode=float(refine_cfg.get("max_scale_ratio_to_mode", 5.0)),
            new_gaussian_scale_max=float(refine_cfg.get("new_gaussian_scale_max", 0.02)),
            add_valid_depth_min=float(refine_cfg.get("add_valid_depth_min", 0.1)),
            add_depth_valid_for_error_min=float(refine_cfg.get("add_depth_valid_for_error_min", 0.05)),
            add_silhouette_threshold=float(refine_cfg.get("add_silhouette_threshold", 0.6)),
            add_norm_mag_threshold=float(refine_cfg.get("add_norm_mag_threshold", 0.9)),
            refine_mode=refine_mode,
            per_frame_iters=per_frame_iters,
            prune_add_interval=prune_add_interval,
            refine_vis_interval=int(refine_cfg.get("refine_vis_interval", 0)),
            refine_vis_save=bool(refine_cfg.get("refine_vis_save", True)),
            refine_vis_interactive=bool(refine_cfg.get("refine_vis_interactive", False)),
        )
        ply_path = self.checkpoint_path / \
            f"{self.config['data']['scene_name']}_global_splats.ply"
        refined_merged_gaussian_model.save_ply(ply_path)
        print(f"Refined global splats saved to {ply_path}")

        if self.config["dataset_name"] == "scannetpp":
            # "NVS evaluation only supported for scannetpp"

            eval_config = deepcopy(self.config)
            print(f"✨ Eval NVS for scene {self.config['data']['scene_name']}...")
            (self.checkpoint_path / "nvs_eval").mkdir(exist_ok=True, parents=True)
            eval_config["data"]["use_train_split"] = False
            test_set = get_dataset(eval_config["dataset_name"])({**eval_config["data"], **eval_config["cam"]})
            test_poses = torch.stack([torch.from_numpy(test_set[i][3]) for i in range(len(test_set))], dim=0)
            test_frames = RenderFrames(test_set, test_poses, self.height, self.width, self.fx, self.fy)

            psnr_list = []
            for i in tqdm(range(len(test_set))):
                gt_color, _ = (
                    test_frames[i]["color"],
                    test_frames[i]["depth"])
                render_settings, est_w2c = test_frames[i]["render_settings"]
                render_dict = render_gaussian_model(refined_merged_gaussian_model, render_settings, est_w2c)
                rendered_color, _ = (render_dict["color"].permute(1, 2, 0), render_dict["depth"],)
                rendered_color = torch.clip(rendered_color, 0, 1)
                save_image(rendered_color.permute(2, 0, 1), self.checkpoint_path / f"nvs_eval/{i:04d}.jpg")
                psnr = calc_psnr(gt_color, rendered_color)
                psnr_list.append(psnr.item())
            print(f"PSNR List: {psnr_list}")
            print(f"Avg. NVS PSNR: {np.array(psnr_list).mean()}")
            with open(self.checkpoint_path / 'nvs_eval' / "results.json", "w") as f:
                data = {"avg_nvs_psnr": np.mean(psnr_list)}
                json.dump(data, f, indent=4)
        
        else: # evaluate rendering performance on the global submap
            print("Running rendering evaluation on global map ...")
            psnr, lpips, ssim, depth_l1 = [], [], [], []
            color_transform = torchvision.transforms.ToTensor()
            lpips_model = LearnedPerceptualImagePatchSimilarity(
                net_type='alex', normalize=True).to(self.device)
            
            submaps_paths = sorted(list((self.checkpoint_path / "submaps").glob('*.ckpt')))
            for submap_path in tqdm(submaps_paths):
                submap = torch.load(submap_path, map_location=self.device)

                for keyframe_id in submap["submap_keyframes"]:

                    _, gt_color, gt_depth, _, _ = self.dataset[keyframe_id]
                    gt_color = color_transform(gt_color).to(self.device)
                    gt_depth = np2torch(gt_depth).to(self.device)

                    estimate_c2w = self.estimated_c2w[keyframe_id]
                    estimate_w2c = np.linalg.inv(estimate_c2w)
                    render_settings, est_w2c = get_render_settings(self.width, self.height, self.dataset.intrinsics, estimate_w2c)
                    render_dict = render_gaussian_model(
                        refined_merged_gaussian_model, render_settings, est_w2c)
                    rendered_color, rendered_depth = render_dict["color"].detach(
                    ), render_dict["depth"][0].detach()
                    rendered_color = torch.clamp(rendered_color, min=0.0, max=1.0)
                    if self.save_render:
                        torchvision.utils.save_image(
                            rendered_color, self.render_path / f"{keyframe_id:05d}.png")

                    mse_loss = torch.nn.functional.mse_loss(
                        rendered_color, gt_color)
                    psnr_value = (-10. * torch.log10(mse_loss)).item()
                    lpips_value = lpips_model(
                        rendered_color[None], gt_color[None]).item()
                    ssim_value = ms_ssim(
                        rendered_color[None], gt_color[None], data_range=1.0, size_average=True).item()
                    depth_l1_value = torch.abs(
                        (rendered_depth - gt_depth)).mean().item()

                    psnr.append(psnr_value)
                    lpips.append(lpips_value)
                    ssim.append(ssim_value)
                    depth_l1.append(depth_l1_value)

            num_frames = len(psnr)
            metrics = {
                "psnr": sum(psnr) / num_frames,
                "lpips": sum(lpips) / num_frames,
                "ssim": sum(ssim) / num_frames,
                "depth_l1_train_view": sum(depth_l1) / num_frames,
                "num_renders": num_frames
            }
            save_dict_to_json(metrics, "rendering_metrics_global.json",
                            directory=self.checkpoint_path)

            x = list(range(len(psnr)))
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].plot(x, psnr, label="PSNR")
            axs[0].legend()
            axs[0].set_title("PSNR")
            axs[1].plot(x, ssim, label="SSIM")
            axs[1].legend()
            axs[1].set_title("SSIM")
            axs[2].plot(x, depth_l1, label="Depth L1 (Train view)")
            axs[2].legend()
            axs[2].set_title("Depth L1 Render")
            plt.tight_layout()
            plt.savefig(str(self.checkpoint_path /
                        "rendering_metrics_global.png"), dpi=300)
            print(metrics)

    def run(self):
        """ Runs the general evaluation flow """

        print("Starting evaluation...🍺")

        try:
            self.run_trajectory_eval()
        except Exception:
            print("Could not run trajectory eval")
            traceback.print_exc()

        try:
            self.run_rendering_eval()
        except Exception:
            print("Could not run rendering eval")
            traceback.print_exc()

        try:
            self.run_reconstruction_eval()
        except Exception:
            print("Could not run reconstruction eval")
            traceback.print_exc()

        try:
            self.run_global_map_eval()
        except Exception:
            print("Could not run global map eval")
            traceback.print_exc()

        # if self.use_wandb: 
        #     evals = ["rendering_metrics.json", "reconstruction_metrics.json", "ate_aligned.json", "nvs_eval/results.json"]
        #     log_metrics_to_wandb(evals, self.checkpoint_path, "Evaluation")
