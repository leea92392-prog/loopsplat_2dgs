""" This module includes the Mapper class, which is responsible scene mapping: Paragraph 3.2  """
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision

from src.entities.arguments import OptimizationParams
from src.entities.datasets import TUM_RGBD, BaseDataset, ScanNet
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.losses import isotropic_loss, l1_loss, ssim
from src.utils.mapper_utils import (calc_psnr, compute_camera_frustum_corners,
                                    compute_frustum_point_ids,
                                    compute_new_points_ids,
                                    compute_opt_views_distribution,
                                    create_point_cloud, geometric_edge_mask,
                                    sample_pixels_based_on_gradient)
from src.utils.gaussian_model_utils import RGB2SH
from src.utils.utils import (get_render_settings, np2ptcloud, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *  # noqa - needed for debugging


class Mapper(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Sets up the mapper parameters
        Args:
            config: configuration of the mapper
            dataset: The dataset object used for extracting camera parameters and reading the data
            logger: The logger object used for logging the mapping process and saving visualizations
        """
        self.config = config
        self.logger = logger
        self.dataset = dataset
        self.iterations = config["iterations"]
        self.new_submap_iterations = config["new_submap_iterations"]
        self.new_submap_points_num = config["new_submap_points_num"]
        self.new_submap_gradient_points_num = config["new_submap_gradient_points_num"]
        self.new_frame_sample_size = config["new_frame_sample_size"]
        self.new_points_radius = config["new_points_radius"]
        self.alpha_thre = config["alpha_thre"]
        self.pruning_thre = config["pruning_thre"]
        self.current_view_opt_iterations = config["current_view_opt_iterations"]
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.keyframes = []
    
    #计算添加高斯的掩模
    def compute_seeding_mask(self, gaussian_model: GaussianModel, keyframe: dict, new_submap: bool) -> np.ndarray:
        """
        Computes a binary mask to identify regions within a keyframe where new Gaussian models should be seeded
        based on alpha masks or color gradient
        Args:
            gaussian_model: The current submap
            keyframe (dict): Keyframe dict containing color, depth, and render settings
            new_submap (bool): A boolean indicating whether the seeding is occurring in current submap or a new submap
        Returns:
            np.ndarray: A binary mask of shpae (H, W) indicates regions suitable for seeding new 3D Gaussian models
        """
        #是否添加高斯的掩模
        seeding_mask = None
        #为什么如果是新的子图要用边缘掩模，只在边缘处添加高斯？？
        if new_submap:
            color_for_mask = (torch2np(keyframe["color"].permute(1, 2, 0)) * 255).astype(np.uint8)
            seeding_mask = geometric_edge_mask(color_for_mask, RGB=True)
        #如果不是新的子图，就用alpha掩模和深度误差掩模
        else:
            render_dict = render_gaussian_model(gaussian_model, keyframe["render_settings"],keyframe["est_w2c"])
            alpha_mask = (render_dict["alpha"] < self.alpha_thre)
            gt_depth_tensor = keyframe["depth"][None]
            depth_error = torch.abs(gt_depth_tensor - render_dict["depth"]) * (gt_depth_tensor > 0)
            depth_error_mask = (render_dict["depth"] > gt_depth_tensor) * (depth_error > 40 * depth_error.median())
            seeding_mask = alpha_mask | depth_error_mask
            seeding_mask = torch2np(seeding_mask[0])
        return seeding_mask
    
    #从当前帧的图像和深度图中创建点云（并变换到世界系），并按照规则采样
    def seed_new_gaussians(self, gt_color: np.ndarray, gt_depth: np.ndarray, intrinsics: np.ndarray,
                           estimate_c2w: np.ndarray, seeding_mask: np.ndarray, is_new_submap: bool) -> np.ndarray:
        """
        Seeds means for the new 3D Gaussian based on ground truth color and depth, camera intrinsics,
        estimated camera-to-world transformation, a seeding mask, and a flag indicating whether this is a new submap.
        Args:
            gt_color: The ground truth color image as a numpy array with shape (H, W, 3).
            gt_depth: The ground truth depth map as a numpy array with shape (H, W).
            intrinsics: The camera intrinsics matrix as a numpy array with shape (3, 3).
            estimate_c2w: The estimated camera-to-world transformation matrix as a numpy array with shape (4, 4).
            seeding_mask: A binary mask indicating where to seed new Gaussians, with shape (H, W).
            is_new_submap: Flag indicating whether the seeding is for a new submap (True) or an existing submap (False).
        Returns:
            np.ndarray: An array of 3D points where new Gaussians will be initialized, with shape (N, 3)

        """
        #获得当前帧图像在世界坐标系下的点云，形状为【num，6】，且在第一个维度上是有序的，与图像的像素顺序一致，也与下面展平的深度图一致
        pts = create_point_cloud(gt_color,  gt_depth, intrinsics, estimate_c2w)
        #深度图展平为一维数组
        flat_gt_depth = gt_depth.flatten()
        non_zero_depth_mask = flat_gt_depth > 0.#一个一维的深度掩模  # need filter if zero depth pixels in gt_depth
        valid_ids = np.flatnonzero(seeding_mask)#把这个掩模也展平为一维数组，找到非零的索引
        if is_new_submap:
            if self.new_submap_points_num < 0:#不用看，config里面没有设置小于零的情况
                uniform_ids = np.arange(pts.shape[0])
            else:
                #从点云里随即采样一些点的索引
                uniform_ids = np.random.choice(pts.shape[0], self.new_submap_points_num, replace=False)
            #基于梯度的采样
            gradient_ids = sample_pixels_based_on_gradient(gt_color, self.new_submap_gradient_points_num)
            #合并采样点，将uniform_ids和gradient_ids合并
            combined_ids = np.concatenate((uniform_ids, gradient_ids))
            combined_ids = np.concatenate((combined_ids, valid_ids))
            #去重
            sample_ids = np.unique(combined_ids)
        else:
            if self.new_frame_sample_size < 0 or len(valid_ids) < self.new_frame_sample_size:
                sample_ids = valid_ids
            else:
                sample_ids = np.random.choice(valid_ids, size=self.new_frame_sample_size, replace=False)
        sample_ids = sample_ids[non_zero_depth_mask[sample_ids]]
        return pts[sample_ids, :].astype(np.float32)

    #优化子地图
    def optimize_submap(self, keyframes: list, gaussian_model: GaussianModel, iterations: int = 100) -> dict:
        """
        Optimizes the submap by refining the parameters of the 3D Gaussian based on the observations
        from keyframes observing the submap.
        Args:
            keyframes: A list of tuples consisting of frame id and keyframe dictionary
            gaussian_model: An instance of the GaussianModel class representing the initial state
                of the Gaussian model to be optimized.
            iterations: The number of iterations to perform the optimization process. Defaults to 100.
        Returns:
            losses_dict: Dictionary with the optimization statistics
        """

        iteration = 0
        losses_dict = {}

        current_frame_iters = self.current_view_opt_iterations * iterations
        distribution = compute_opt_views_distribution(len(keyframes), iterations, current_frame_iters)
        start_time = time.time()
        while iteration < iterations + 1:
            gaussian_model.optimizer.zero_grad(set_to_none=True)
            keyframe_id = np.random.choice(np.arange(len(keyframes)), p=distribution)

            frame_id, keyframe = keyframes[keyframe_id]
            render_pkg = render_gaussian_model(gaussian_model, keyframe["render_settings"], keyframe["est_w2c"])

            image, depth = render_pkg["color"], render_pkg["depth"]
            # show_render_result(render_rgb=image, render_depth=depth,
            #                    gt_depth=keyframe["depth"],gt_rgb=keyframe["color"],render_normal=render_pkg["normal"])
            if keyframe["exposure_ab"] is not None:
                image = torch.clamp(image * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
            gt_image = keyframe["color"]
            gt_depth = keyframe["depth"]

            mask = (gt_depth > 0) & (~torch.isnan(depth)).squeeze(0)
            #颜色损失
            color_loss = (1.0 - self.opt.lambda_dssim) * l1_loss(
                image[:, mask], gt_image[:, mask]) + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            #深度损失
            depth_loss = l1_loss(depth[:, mask], gt_depth[mask])
            #正则化损失
            reg_loss = isotropic_loss(gaussian_model.get_scaling())
            rend_dist = render_pkg["rend_dist"]
            dist_loss = 1000*rend_dist.mean()
            rend_normal  = render_pkg['rend_normal']
            surf_normal = render_pkg['normal']
            normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
            normal_loss = 0.05*(normal_error).mean()
            total_loss = color_loss + depth_loss + reg_loss
            #print("color_loss:",color_loss.item(),"depth_loss:",depth_loss.item(),"isotropic_loss:",reg_loss.item(),"dist_loss:",dist_loss.item(),"normal_loss",normal_loss.item())
            if self.config['use_normal_reg']:
                total_loss +=   normal_loss
            if self.config['use_dist_reg']:
                total_loss +=dist_loss
            total_loss.backward()

            losses_dict[frame_id] = {"color_loss": color_loss.item(),
                                     "depth_loss": depth_loss.item(),
                                     "total_loss": total_loss.item()}

            with torch.no_grad():

                if iteration == iterations // 2 or iteration == iterations:
                    prune_mask = (gaussian_model.get_opacity()
                                  < self.pruning_thre).squeeze()
                    gaussian_model.prune_points(prune_mask)

                # Optimizer step
                if iteration < iterations:
                    gaussian_model.optimizer.step()
                gaussian_model.optimizer.zero_grad(set_to_none=True)

            iteration += 1
        optimization_time = time.time() - start_time
        losses_dict["optimization_time"] = optimization_time
        losses_dict["optimization_iter_time"] = optimization_time / iterations
        return losses_dict

    #向子地图中添加新的高斯
    # def grow_submap(self, gt_depth: np.ndarray, estimate_c2w: np.ndarray, gaussian_model: GaussianModel,
        #                 pts: np.ndarray, filter_cloud: bool) -> int:
        #     """
        #     Expands the submap by integrating new points from the current keyframe
        #     Args:
        #         gt_depth: The ground truth depth map for the current keyframe, as a 2D numpy array.
        #         estimate_c2w: The estimated camera-to-world transformation matrix for the current keyframe of shape (4x4)
        #         gaussian_model (GaussianModel): The Gaussian model representing the current state of the submap.
        #         pts: The current set of 3D points in the keyframe of shape (N, 6)
        #         filter_cloud: A boolean flag indicating whether to apply filtering to the point cloud to remove
        #             outliers or noise before integrating it into the map.
        #     Returns:
        #         int: The number of points added to the submap
        #     """
        #     gaussian_points = gaussian_model.get_xyz()
        #     #计算当前帧的相机视锥体的角点，并变换到世界系
        #     camera_frustum_corners = compute_camera_frustum_corners(gt_depth, estimate_c2w, self.dataset.intrinsics)
        #     #计算在视锥体内的点的索引
        #     reused_pts_ids = compute_frustum_point_ids(
        #         gaussian_points, np2torch(camera_frustum_corners), device="cuda")
        #     #输入原有的视锥体内的点的索引和当前帧的点云，对于每一个新的点云通过计算其阈值范围内有无邻近点，来决定是否添加到视锥体内
        #     new_pts_ids = compute_new_points_ids(gaussian_points[reused_pts_ids], np2torch(pts[:, :3]).contiguous(),
        #                                          radius=self.new_points_radius, device="cuda")
        #     new_pts_ids = torch2np(new_pts_ids)
        #     point_rgb = pts[:, 3:]
        #     print("rgb max:", point_rgb.max(), "rgb min:", point_rgb.min(), "rgb mean:", point_rgb.mean())
        #     # if new_pts_ids.shape[0] > 0:
        #     #     cloud_to_add = np2ptcloud(pts[new_pts_ids, :3], pts[new_pts_ids, 3:] / 255.0)
        #     #     if filter_cloud:
        #     #         cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
        #     #     #真正添加高斯的地方
        #     #     gaussian_model.add_points(cloud_to_add)
        #     if new_pts_ids.shape[0] > 0:
        #         cloud_to_add = np2ptcloud(pts[:, :3], pts[:, 3:] / 255.0)
        #         if filter_cloud:
        #             cloud_to_add, _ = cloud_to_add.remove_statistical_outlier(nb_neighbors=40, std_ratio=2.0)
        #         #真正添加高斯的地方
        #         gaussian_model.add_points(cloud_to_add)
        #     gaussian_model._features_dc.requires_grad = True
        #     gaussian_model._features_rest.requires_grad = True
        #     print("Gaussian model size", gaussian_model.get_size())
        #     return new_pts_ids.shape[0]

    def get_pointcloud(
        self,
        color,
        depth,
        w2c,
        transform_pts=True,
        mask=None,
        compute_mean_sq_dist=False,
        mean_sq_dist_method="projective",
        add_noise=False,
        use_median=False,
        noise_scale=0.5,
    ):
        """
        Compute a new Gaussian for each empty pixel in the given frame.
        """
        width, height = color.shape[2], color.shape[1]
        FX = self.dataset.fx
        FY = self.dataset.fy
        CX = self.dataset.cx
        CY = self.dataset.cy

        # Compute indices of pixels
        x_grid, y_grid = torch.meshgrid(
            torch.arange(width).cuda().float(),
            torch.arange(height).cuda().float(),
            indexing="xy",
        )
        xx = (x_grid - CX) / FX
        yy = (y_grid - CY) / FY
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)
        depth_z = depth.reshape(-1)

        if add_noise and torch.any(mask):
            if use_median:
                if torch.any(~mask):
                    median_z = torch.median(depth_z[~mask])
                    std_z = torch.sqrt(torch.var(depth_z[~mask]))
                else:
                    median_z = 1
                    std_z = 1
                depth_z = torch.normal(
                    median_z * torch.ones_like(depth_z), noise_scale * std_z
                )
            else:
                std_z = torch.sqrt(torch.var(depth_z[mask]))
                depth_z = torch.normal(depth_z, noise_scale * std_z)

        # Initialize point cloud
        pts_cam = torch.stack((xx * depth_z, yy * depth_z, depth_z), dim=-1)
        if transform_pts:
            # Transform points to world frame
            pix_ones = torch.ones(height * width, 1).cuda().float()
            pts4 = torch.cat((pts_cam, pix_ones), dim=1)
            c2w = torch.inverse(w2c)
            pts = (c2w @ pts4.T).T[:, :3]
        else:
            pts = pts_cam

        # Compute mean squared distance for initializing the scale of the Gaussians
        if compute_mean_sq_dist:
            scale_gaussian = depth_z / ((FX + FY) / 2)
            mean3_sq_dist = scale_gaussian**2
           
        # Colorize point cloud
        cols = torch.permute(color, (1, 2, 0)).reshape(
            -1, 3
        )  # (C, H, W) -> (H, W, C) -> (H * W, C)
        point_cld = torch.cat((pts, cols), -1)

        # Select points based on mask
        if mask is not None:
            point_cld = point_cld[mask]
            if compute_mean_sq_dist:
                mean3_sq_dist = mean3_sq_dist[mask]

        if compute_mean_sq_dist:
            return point_cld, mean3_sq_dist
        else:
            return point_cld

    def grow_submap(self, keyframe: dict,gaussian_model,is_new_submap) -> int:
        """
        Expands the submap by integrating new points from the current keyframe
        Args:
            keyframe (dict): Keyframe dict containing color, depth, and render settings
        Returns:
            int: The number of points added to the submap
        """
        gt_color = keyframe["color"]
        gt_depth = keyframe["depth"]
        estimate_w2c = keyframe["est_w2c"]
        #新的子图，直接添加所有的点
        if is_new_submap:
            non_presence_mask = torch.ones(
                    gt_depth.shape, dtype=bool, device="cuda"
                ).reshape(-1)
            render_mask = torch.zeros(
                    gt_depth.shape, dtype=bool, device="cuda"
                ).reshape(-1)
        #添加新观察到的点
        else:
            result = render_gaussian_model(gaussian_model, keyframe["render_settings"],keyframe["est_w2c"])
            silhouette = result["alpha"] 
            non_presence_sil_mask = silhouette < 0.6
            #深度误差掩模
            render_depth = result["depth"]
            depth_error = torch.abs(gt_depth - render_depth) * (gt_depth > 0)
            non_presence_depth_mask = depth_error > 10 * depth_error.median()
            #法向量掩模
            surf_norm = result["normal"]
            norm_magnitude = torch.sqrt(torch.sum(surf_norm ** 2, dim=0))
            norm_mask = norm_magnitude < 0.9
            # Determine non-presence mask
            non_presence_mask = non_presence_sil_mask | non_presence_depth_mask | norm_mask
            non_presence_mask = non_presence_mask.reshape(-1)
        valid_depth_mask = gt_depth > 0
        non_presence_mask = non_presence_mask & valid_depth_mask.reshape(-1)
        #创建点云
        new_pt_cld, mean3_sq_dist = self.get_pointcloud(
                gt_color,
                gt_depth,
                estimate_w2c,
                mask=non_presence_mask,
                compute_mean_sq_dist=True,
            )
        #添加高斯
        new_rgb = new_pt_cld[:, 3:6].float().cuda()
        fused_color = RGB2SH(new_pt_cld[:, 3:6].float().cuda())
        new_features = (torch.zeros( 
                (fused_color.shape[0] , 3 , (gaussian_model.max_sh_degree + 1) ** 2 , )).float().cuda()
            )
        new_features[:, :3, 0] = fused_color
        new_features[:, 3:, 1:] = 0.0
        # Identity rotation
        new_rots = torch.zeros((new_pt_cld.shape[0], 4), device="cuda")
        new_rots[:, 0] = 1
        # 0.5 opacity
        new_opacities = torch.zeros(
            (new_pt_cld.shape[0], 1), dtype=torch.float, device="cuda"
        )
        # Spherical pixel-wide scaling
        new_scaling = torch.log(torch.sqrt(mean3_sq_dist))[..., None].repeat(1, 2)
        gaussian_model.densification_postfix(
            new_xyz=new_pt_cld[:, :3],
            new_features_dc=new_features[:, :, 0:1].transpose(1, 2).contiguous(),
            new_features_rest=new_features[:, :, 1:].transpose(1, 2).contiguous(),
            new_opacities=new_opacities,
            new_scaling=new_scaling,
            new_rotation=new_rots,
            new_rgb=new_rgb,
            )
        return new_pt_cld.shape[0]
    # def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool, exposure_ab=None) -> dict:
        #     """ Calls out the mapping process described in paragraph 3.2
        #     The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        #     Args:
        #         frame_id: current keyframe id
        #         estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
        #         gaussian_model (GaussianModel): The current Gaussian model of the submap
        #         is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        #     Returns:
        #         opt_dict: Dictionary with statistics about the optimization process
        #     """
        #     #读取当前帧的图像和深度图（numpy数组的格式）
        #     _, gt_color, gt_depth, _ = self.dataset[frame_id]
        #     estimate_w2c = np.linalg.inv(estimate_c2w)
        #     #将图像转换为张量
        #     color_transform = torchvision.transforms.ToTensor()
        #     render_settings,est_w2c=get_render_settings(
        #             self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)
        #     keyframe = {
        #         "color": color_transform(gt_color).cuda(),
        #         "depth": np2torch(gt_depth, device="cuda"),
        #         "render_settings": render_settings,
        #         "est_w2c": est_w2c,
        #         "exposure_ab": exposure_ab}
        #     #计算添加新高斯的掩模
        #     #如果是空白子图，就用边缘掩模，只在边缘处添加高斯（？？？）；如果不是空白子图，就用alpha掩模和深度误差掩模
        #     seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        #     #从新的图像帧和深度帧中创建点云，并按照规则采样
        #     pts = self.seed_new_gaussians(
        #         gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)
        #     #如果是TUM_RGBD或者ScanNet数据集，并且不是新的子地图，filter_cloud被设置为True
        #     filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap
        #     #在这里会向gaussian_model中添加新的高斯
        #     new_pts_num = self.grow_submap(gt_depth, estimate_c2w, gaussian_model, pts, filter_cloud)

        #     max_iterations = self.iterations
        #     if is_new_submap:
        #         max_iterations = self.new_submap_iterations
        #     start_time = time.time()
        #     #优化子地图
        #     opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)
        #     optimization_time = time.time() - start_time
        #     print("Optimization time: ", optimization_time)

        #     self.keyframes.append((frame_id, keyframe))

        #     # Visualise the mapping for the current frame
        #     with torch.no_grad():
        #         render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"],keyframe["est_w2c"])
        #         image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
        #         if keyframe["exposure_ab"] is not None:
        #             image_vis = torch.clamp(image_vis * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
        #         psnr_value = calc_psnr(image_vis, keyframe["color"]).item()
        #         opt_dict["psnr_render"] = psnr_value
        #         print(f"PSNR this frame: {psnr_value}")
        #         self.logger.vis_mapping_iteration(
        #             frame_id, max_iterations,
        #             image_vis.clone().detach().permute(1, 2, 0),
        #             depth_vis.clone().detach().permute(1, 2, 0),
        #             keyframe["color"].permute(1, 2, 0),
        #             keyframe["depth"].unsqueeze(-1),
        #             seeding_mask=seeding_mask)

        #     # Log the mapping numbers for the current frame
        #     self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
        #                                       optimization_time/max_iterations, opt_dict)
        #     return opt_dict


    def map(self, frame_id: int, estimate_c2w: np.ndarray, gaussian_model: GaussianModel, is_new_submap: bool, exposure_ab=None) -> dict:
        """ Calls out the mapping process described in paragraph 3.2
        The process goes as follows: seed new gaussians -> add to the submap -> optimize the submap
        Args:
            frame_id: current keyframe id
            estimate_c2w (np.ndarray): The estimated camera-to-world transformation matrix of shape (4x4)
            gaussian_model (GaussianModel): The current Gaussian model of the submap
            is_new_submap (bool): A boolean flag indicating whether the current frame initiates a new submap
        Returns:
            opt_dict: Dictionary with statistics about the optimization process
        """
        #读取当前帧的图像和深度图（numpy数组的格式）
        _, gt_color, gt_depth,_,_ = self.dataset[frame_id]
        estimate_w2c = np.linalg.inv(estimate_c2w)
        #将图像转换为张量
        color_transform = torchvision.transforms.ToTensor()
        render_settings,est_w2c=get_render_settings(
                self.dataset.width, self.dataset.height, self.dataset.intrinsics, estimate_w2c)
        keyframe = {
            "color": color_transform(gt_color).cuda(),
            "depth": np2torch(gt_depth, device="cuda"),
            "render_settings": render_settings,
            "est_w2c": est_w2c,
            "exposure_ab": exposure_ab}
        # #计算添加新高斯的掩模
        # #如果是空白子图，就用边缘掩模，只在边缘处添加高斯（？？？）；如果不是空白子图，就用alpha掩模和深度误差掩模
        # seeding_mask = self.compute_seeding_mask(gaussian_model, keyframe, is_new_submap)
        # #从新的图像帧和深度帧中创建点云，并按照规则采样
        # pts = self.seed_new_gaussians(
        #     gt_color, gt_depth, self.dataset.intrinsics, estimate_c2w, seeding_mask, is_new_submap)
        # #如果是TUM_RGBD或者ScanNet数据集，并且不是新的子地图，filter_cloud被设置为True
        # filter_cloud = isinstance(self.dataset, (TUM_RGBD, ScanNet)) and not is_new_submap
        #在这里会向gaussian_model中添加新的高斯
        new_pts_num = self.grow_submap(keyframe, gaussian_model,is_new_submap)

        max_iterations = self.iterations
        if is_new_submap:
            max_iterations = self.new_submap_iterations
        start_time = time.time()
        #优化子地图
        opt_dict = self.optimize_submap([(frame_id, keyframe)] + self.keyframes, gaussian_model, max_iterations)
        optimization_time = time.time() - start_time
        print("Optimization time: ", optimization_time)

        self.keyframes.append((frame_id, keyframe))

        # Visualise the mapping for the current frame
        with torch.no_grad():
            render_pkg_vis = render_gaussian_model(gaussian_model, keyframe["render_settings"],keyframe["est_w2c"])
            image_vis, depth_vis = render_pkg_vis["color"], render_pkg_vis["depth"]
            if keyframe["exposure_ab"] is not None:
                image_vis = torch.clamp(image_vis * torch.exp(keyframe["exposure_ab"][0]) + keyframe["exposure_ab"][1], 0, 1.)
            psnr_value = calc_psnr(image_vis, keyframe["color"]).item()
            opt_dict["psnr_render"] = psnr_value
            print(f"PSNR this frame: {psnr_value}")
            self.logger.vis_mapping_iteration(
                frame_id, max_iterations,
                image_vis.clone().detach().permute(1, 2, 0),
                depth_vis.clone().detach().permute(1, 2, 0),
                keyframe["color"].permute(1, 2, 0),
                keyframe["depth"].unsqueeze(-1),
                )

        # Log the mapping numbers for the current frame
        self.logger.log_mapping_iteration(frame_id, new_pts_num, gaussian_model.get_size(),
                                          optimization_time/max_iterations, opt_dict)
        return opt_dict