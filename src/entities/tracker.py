""" This module includes the Mapper class, which is responsible scene mapping: Paper Section 3.4  """
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from scipy.spatial.transform import Rotation as R

from src.entities.arguments import OptimizationParams
from src.entities.losses import l1_loss
from src.entities.gaussian_model import GaussianModel
from src.entities.logger import Logger
from src.entities.datasets import BaseDataset
from src.entities.visual_odometer import VisualOdometer
from src.utils.gaussian_model_utils import build_rotation
from src.utils.tracker_utils import (compute_camera_opt_params,
                                     extrapolate_poses, multiply_quaternions,
                                     transformation_to_quaternion)
from src.utils.pose_utils import euler_matrix
from src.utils.utils import (get_render_settings, np2torch,
                             render_gaussian_model, torch2np)
from src.utils.vis_utils import *
from src.entities.pose_utils_adapter import PoseUtilsAdapter  
class Tracker(object):
    def __init__(self, config: dict, dataset: BaseDataset, logger: Logger) -> None:
        """ Initializes the Tracker with a given configuration, dataset, and logger.
        Args:
            config: Configuration dictionary specifying hyperparameters and operational settings.
            dataset: The dataset object providing access to the sequence of frames.
            logger: Logger object for logging the tracking process.
        """
        self.dataset = dataset
        self.logger = logger
        self.config = config
        self.filter_alpha = self.config["filter_alpha"]
        self.filter_outlier_depth = self.config["filter_outlier_depth"]
        self.alpha_thre = self.config["alpha_thre"]
        self.soft_alpha = self.config["soft_alpha"]
        self.mask_invalid_depth_in_color_loss = self.config["mask_invalid_depth"]
        self.w_color_loss = self.config["w_color_loss"]
        self.transform = torchvision.transforms.ToTensor()
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))
        self.frame_depth_loss = []
        self.frame_color_loss = []
        self.odometry_type = self.config["odometry_type"]
        self.help_camera_initialization = self.config["help_camera_initialization"]
        self.use_imu = self.config["use_imu"]
        self.init_err_ratio = self.config["init_err_ratio"]
        self.enable_exposure = self.config["enable_exposure"]
        self.odometer = VisualOdometer(self.dataset.intrinsics, self.config["odometer_method"])
        self.use_pose_utils = self.config.get("use_pose_utils", False)
        if self.use_imu:
            self.tstamps = self.dataset.tstamps
            self.tf = {}
            self.tf["c2i"] = self.dataset.get_c2i_tf()
        self.pose_utils_adapter = PoseUtilsAdapter(config["pose_utils"], dataset)  



    def compute_losses(self, gaussian_model: GaussianModel, render_settings: dict,est_w2c,
                       gt_color: torch.Tensor, gt_depth: torch.Tensor, depth_mask: torch.Tensor,
                       exposure_ab=None) -> tuple:
        """ Computes the tracking losses with respect to ground truth color and depth.
        Args:
            gaussian_model: The current state of the Gaussian model of the scene.
            render_settings: Dictionary containing rendering settings such as image dimensions and camera intrinsics.
            opt_cam_rot: Optimizable tensor representing the camera's rotation.
            opt_cam_trans: Optimizable tensor representing the camera's translation.
            gt_color: Ground truth color image tensor.
            gt_depth: Ground truth depth image tensor.
            depth_mask: Binary mask indicating valid depth values in the ground truth depth image.
        Returns:
            A tuple containing losses and renders
        """
        render_dict = render_gaussian_model(gaussian_model, render_settings,est_w2c,)
        rendered_color, rendered_depth = render_dict["color"], render_dict["depth"]
        #show_render_result(render_rgb=rendered_color, render_depth=rendered_depth,render_normal=render_dict["normal"])
        if self.enable_exposure:
            rendered_color = torch.clamp(torch.exp(exposure_ab[0]) * rendered_color + exposure_ab[1], 0, 1.)
        alpha_mask = render_dict["alpha"] > 0.9

        tracking_mask = torch.ones_like(alpha_mask).bool()
        tracking_mask &= depth_mask
        depth_err = torch.abs(rendered_depth - gt_depth) * depth_mask

        if self.filter_alpha:
            tracking_mask &= alpha_mask
        if self.filter_outlier_depth and torch.median(depth_err) > 0:
            tracking_mask &= depth_err < 50 * torch.median(depth_err)

        color_loss = l1_loss(rendered_color, gt_color, agg="none")
        depth_loss = l1_loss(rendered_depth, gt_depth, agg="none") * tracking_mask

        if self.soft_alpha:
            alpha = render_dict["alpha"] ** 3
            color_loss *= alpha
            depth_loss *= alpha
            if self.mask_invalid_depth_in_color_loss:
                color_loss *= tracking_mask
        else:
            color_loss *= tracking_mask

        color_loss = color_loss.sum()
        depth_loss = depth_loss.sum()

        return color_loss, depth_loss, rendered_color, rendered_depth, alpha_mask

    def quad2rotation(self, q):
        if not isinstance(q, torch.Tensor):
            q = torch.tensor(q).cuda()
        norm = torch.sqrt(
            q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]
        )
        q = q / norm
        rot = torch.zeros((3, 3)).to(q)
        r = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        rot[0, 0] = 1 - 2 * (y * y + z * z)
        rot[0, 1] = 2 * (x * y - r * z)
        rot[0, 2] = 2 * (x * z + r * y)
        rot[1, 0] = 2 * (x * y + r * z)
        rot[1, 1] = 1 - 2 * (x * x + z * z)
        rot[1, 2] = 2 * (y * z - r * x)
        rot[2, 0] = 2 * (x * z - r * y)
        rot[2, 1] = 2 * (y * z + r * x)
        rot[2, 2] = 1 - 2 * (x * x + y * y)
        return rot
    
    def propagate_imu(self,camm1, camm2, imu_meas_list, c2i, dt_cam, dt_imu):
        """
        Propagate camera pose based on IMU measurements

        Args:
            c2w (tensor): pose at idx-1
            c2 (tensor): pose at idx-2
            imu_meas_list (tensor): IMU measurements from last timestep to current timestep
            c2i (tensor): camera-to-imu homogeneous transformation matrix
            dt_cam: time between camera frames
            dt_imu: time between imu measurements
        Returns:
            cam (tensor): propagated pose
        """


        # Transform camera frame to IMU frame
        c2wm1 = torch.from_numpy(camm1).cuda()
        c2wm2 = torch.from_numpy(camm2).cuda()
        i2wm1 = c2wm1 @ torch.inverse(c2i)
        i2wm2 = c2wm2 @ torch.inverse(c2i)

        i2w = i2wm1.clone()  # Tracked IMU pose

        # Get the linear velocity using the constant velocity model
        rel_T = torch.inverse(i2wm2) @ i2wm1  # rel transform from i(now-1)->i(now-2)
        lin_vel = rel_T[:3, 3] / dt_cam #从i(now-1)到i(now-2)的相对速度
        G = torch.tensor([0.0, -9.80665, 0.0])
        #G = torch.tensor([0.0, 0.0, -9.80665])
        # Then, do IMU preintegration
        #根据IMU的测量计算
        for imu_meas in imu_meas_list:
            lin_accel = imu_meas[25:28]
            ang_vel = imu_meas[13:16]

            # Remove the gravity component
            lin_accel -= i2w[:3, :3].T @ G.to(i2w)

            # Preintegrate
            change_in_position = lin_vel * dt_imu + 0.5 * lin_accel * dt_imu * dt_imu
            change_in_orientation = ang_vel * dt_imu

            # Propagate pose
            delta = euler_matrix(*change_in_orientation, axes="sxyz")
            #delta = torch.eye(4).to(i2w)
            delta[0:3, 3] = change_in_position  # delta is i(now)->i(prev)
            i2w = i2w @ delta

            # TODO: perform covariance propagation

        # Transform back to camera frame
        c2w = i2w @ c2i
        return c2w.detach().cpu().numpy() 
    
    def track(self, frame_id: int, gaussian_model: GaussianModel, prev_c2ws: np.ndarray) -> np.ndarray:
        """
        Updates the camera pose estimation for the current frame based on the provided image and depth, using either ground truth poses,
        constant speed assumption, or visual odometry.
        Args:
            frame_id: Index of the current frame being processed.
            gaussian_model: The current Gaussian model of the scene.
            prev_c2ws: Array containing the camera-to-world transformation matrices for the frames (0, i - 2, i - 1)
        Returns:
            The updated camera-to-world transformation matrix for the current frame.
        """
        #depth是numpy float32格式，intrinsics是np.array格式
        _, image, depth, gt_c2w,imu_meas = self.dataset[frame_id]
        intrinsics = self.dataset.intrinsics  
        if self.use_pose_utils:  
            # 使用pose_utils进行位姿估计  
            estimated_c2w = self.pose_utils_adapter.estimate_pose(
                frame_id, image, depth, intrinsics
            )
            print(f"estimated_c2w (frame {frame_id}):\n{estimated_c2w}")
            print(f"gt_c2w (frame {frame_id}):\n{gt_c2w}")
            return estimated_c2w, None  
        if (self.help_camera_initialization or self.odometry_type == "odometer") and self.odometer.last_rgbd is None:
            _, last_image, last_depth, _ = self.dataset[frame_id - 1]
            self.odometer.update_last_rgbd(last_image, last_depth)
        if self.odometry_type == "gt":
            return gt_c2w, None
        elif self.odometry_type == "const_speed":
            init_c2w = extrapolate_poses(prev_c2ws[1:])
        elif self.odometry_type == "odometer":
            odometer_rel = self.odometer.estimate_rel_pose(image, depth)
            init_c2w = prev_c2ws[-1] @ odometer_rel
        elif self.odometry_type == "previous":
            init_c2w = prev_c2ws[-1]
        if self.use_imu and imu_meas is not None:
            init_c2w = self.propagate_imu(
                        prev_c2ws[-1],
                        prev_c2ws[-2],
                        imu_meas,
                        self.tf["c2i"],
                        self.tstamps[frame_id - 1] - self.tstamps[frame_id - 2],
                        # 1 / self.cfg["cam"]["fps"] * self.cfg["stride"],
                        1 / 100.0,
                    )
        exposure_ab = None
        init_w2c = np.linalg.inv(init_c2w)
        camera_T = np2torch(init_w2c, "cuda")[:3,3].requires_grad_(True)
        camera_q = np2torch(R.from_matrix(init_w2c[:3,:3]).as_quat(canonical=True)[[3, 0, 1, 2 ]], "cuda").requires_grad_(True)        
        pose_optimizer = torch.optim.Adam(
            [
                {
                    "params": [camera_T],
                    "lr": self.config["cam_trans_lr"],
                },
                {
                    "params": [camera_q],
                    "lr": self.config["cam_rot_lr"],
                },
            ]
        )
        est_w2c = torch.eye(4, device="cuda").float()
        est_w2c[:3,:3] = self.quad2rotation(camera_q)
        est_w2c[:3,3] = camera_T
        render_settings,_ = get_render_settings(
            self.dataset.width, self.dataset.height, self.dataset.intrinsics)
        gt_color = self.transform(image).cuda()
        gt_depth = np2torch(depth, "cuda")
        depth_mask = gt_depth > 0.0
        gt_trans = np2torch(gt_c2w[:3, 3])
        gt_quat = np2torch(R.from_matrix(gt_c2w[:3, :3]).as_quat(canonical=True)[[3, 0, 1, 2]])
        num_iters = self.config["iterations"]
        current_min_loss = float("inf")

        print(f"\nTracking frame {frame_id}")
        # Initial loss check
        color_loss, depth_loss, _, _, _ = self.compute_losses(gaussian_model, render_settings,est_w2c, gt_color, gt_depth, depth_mask, )
        #如果初始损失过大，增加迭代次数
        if len(self.frame_color_loss) > 0 and (
            color_loss.item() > self.init_err_ratio * np.median(self.frame_color_loss)
            or depth_loss.item() > self.init_err_ratio * np.median(self.frame_depth_loss)
        ):
            num_iters *= 2
            print(f"Higher initial loss, increasing num_iters to {num_iters}")
        for iter in range(num_iters):
            color_loss, depth_loss, _, _, _, = self.compute_losses(
                gaussian_model, render_settings, est_w2c, gt_color, gt_depth, depth_mask)

            total_loss = (self.w_color_loss * color_loss + (1 - self.w_color_loss) * depth_loss)
            total_loss.backward()
            pose_optimizer.step()
            #gaussian_model.scheduler.step(total_loss, epoch=iter)
            pose_optimizer.zero_grad(set_to_none=True)
            est_w2c = torch.eye(4, device="cuda").float()
            est_w2c[:3,:3] = self.quad2rotation(camera_q)
            est_w2c[:3,3] = camera_T
            with torch.no_grad():
                if total_loss.item() < current_min_loss:
                    current_min_loss = total_loss.item()
                    candidate_q = camera_q.clone().detach()
                    candidate_T = camera_T.clone().detach()
                if iter == num_iters - 1:
                    curr_w2c = torch.eye(4, device="cuda").float()
                    curr_w2c[:3,:3] = self.quad2rotation(candidate_q)
                    curr_w2c[:3,3] = candidate_T   
                else:
                    curr_w2c = torch.eye(4, device="cuda").float()
                    curr_w2c[:3,:3] = self.quad2rotation(camera_q)
                    curr_w2c[:3,3] = camera_T                
                if iter == num_iters - 1:
                    cur_c2w = torch.inverse(curr_w2c)
                    cur_cam = transformation_to_quaternion(cur_c2w)
                    if (gt_quat * cur_cam[:4]).sum() < 0:  # for logging purpose
                        gt_quat *= -1
                    self.frame_color_loss.append(color_loss.item())
                    self.frame_depth_loss.append(depth_loss.item())
                    self.logger.log_tracking_iteration(
                        frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                        wandb_output=True, print_output=True)
                # elif iter % 20 == 0:
                #     self.logger.log_tracking_iteration(
                #         frame_id, cur_cam, gt_quat, gt_trans, total_loss, color_loss, depth_loss, iter, num_iters,
                #         wandb_output=False, print_output=True)

        final_c2w = cur_c2w
        final_c2w[-1, :] = torch.tensor([0., 0., 0., 1.], dtype=final_c2w.dtype, device=final_c2w.device)
        return torch2np(final_c2w), exposure_ab