""" This module includes the Gaussian-SLAM class, which is responsible for controlling Mapper and Tracker
    It also decides when to start a new submap and when to update the estimated camera poses.
"""
import os
import pprint
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import roma

from src.entities.arguments import OptimizationParams
from src.entities.datasets import get_dataset
from src.entities.gaussian_model import GaussianModel
from src.entities.mapper import Mapper
from src.entities.tracker import Tracker
from src.entities.lc import Loop_closure
from src.entities.logger import Logger
from src.utils.io_utils import save_dict_to_ckpt, save_dict_to_yaml
from src.utils.mapper_utils import exceeds_motion_thresholds 
from src.utils.utils import np2torch, setup_seed, torch2np
from src.utils.vis_utils import *  # noqa - needed for debugging


class GaussianSLAM(object):

    def __init__(self, config: dict) -> None:

        self._setup_output_path(config)
        self.device = "cuda"
        self.config = config

        self.scene_name = config["data"]["scene_name"]
        self.dataset_name = config["dataset_name"]
        #self.dataset[i]会调用__getitem__方法，返回第i帧的数据，包括（idx，rgb，depth，pose），读进的是numpy数组格式
        self.dataset = get_dataset(config["dataset_name"])(dataset_config=self.config)

        n_frames = len(self.dataset)
        frame_ids = list(range(n_frames))
        #选择特定的帧ID作为mapping的帧，这里的加号不是运算符，而是要把n_frames-1加入到frame_ids中
        self.mapping_frame_ids = frame_ids[::config["mapping"]["map_every"]] + [n_frames - 1]
        #待估计的相机位姿
        self.estimated_c2ws = torch.empty(len(self.dataset), 4, 4)
        self.estimated_c2ws[0] = torch.from_numpy(self.dataset[0][3])
        #？？？
        self.exposures_ab = torch.zeros(len(self.dataset), 2)

        save_dict_to_yaml(config, "config.yaml", directory=self.output_path)
        #根据移动距离和旋转角度判断是否开始新的子地图
        self.submap_using_motion_heuristic = config["mapping"]["submap_using_motion_heuristic"]

        self.keyframes_info = {}
        self.opt = OptimizationParams(ArgumentParser(description="Training script parameters"))

        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids = [0]
        else:
            self.new_submap_frame_ids = frame_ids[::config["mapping"]["new_submap_every"]] + [n_frames - 1]
            self.new_submap_frame_ids.pop(0)
        if "rot_thre" in config:
            self.rot_thre = config["rot_thre"]
            self.trans_thre = config["trans_thre"]
        else:
            self.rot_thre = 50
            self.trans_thre = 0.5
        self.logger = Logger(self.output_path, config["use_wandb"])
        self.mapper = Mapper(config["mapping"], self.dataset, self.logger)
        self.tracker = Tracker(config["tracking"], self.dataset, self.logger)
        self.enable_exposure = self.tracker.enable_exposure
        self.loop_closer = Loop_closure(config, self.dataset, self.logger)
        self.loop_closer.submap_path = self.output_path / "submaps"
        
        print('Tracking config')
        pprint.PrettyPrinter().pprint(config["tracking"])
        print('Mapping config')
        pprint.PrettyPrinter().pprint(config["mapping"])
        print('Loop closure config')
        pprint.PrettyPrinter().pprint(config["lc"])
        

    def _setup_output_path(self, config: dict) -> None:
        """ Sets up the output path for saving results based on the provided configuration. If the output path is not
        specified in the configuration, it creates a new directory with a timestamp.
        Args:
            config: A dictionary containing the experiment configuration including data and output path information.
        """
        if "output_path" not in config["data"]:
            output_path = Path(config["data"]["output_path"])
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = output_path / self.timestamp
        else:
            self.output_path = Path(config["data"]["output_path"])
        self.output_path.mkdir(exist_ok=True, parents=True)
        
        os.makedirs(self.output_path / "mapping_vis", exist_ok=True)
        os.makedirs(self.output_path / "tracking_vis", exist_ok=True)

    #判断是否开始新的子地图
    def should_start_new_submap(self, frame_id: int) -> bool:
        """ Determines whether a new submap should be started based on the motion heuristic or specific frame IDs.
        Args:
            frame_id: The ID of the current frame being processed.
        Returns:
            A boolean indicating whether to start a new submap.
        """
        if self.submap_using_motion_heuristic:
            if exceeds_motion_thresholds(
                self.estimated_c2ws[frame_id], self.estimated_c2ws[self.new_submap_frame_ids[-1]],
                    rot_thre=self.rot_thre, trans_thre=self.trans_thre):
                print(f"\nNew submap at {frame_id}")
                return True
        elif frame_id in self.new_submap_frame_ids:
            return True
        return False
    #保存当前的高斯模型作为历史子地图
    def save_current_submap(self, gaussian_model: GaussianModel):
        """Saving the current submap's checkpoint and resetting the Gaussian model

        Args:
            gaussian_model (GaussianModel): The current GaussianModel instance to capture and reset for the new submap.
        """
        
        gaussian_params = gaussian_model.capture_dict()
        submap_ckpt_name = str(self.submap_id).zfill(6)
        #保存子地图的高斯参数和子地图内的关键帧信息
        submap_ckpt = {
            "gaussian_params": gaussian_params,
            "submap_keyframes": sorted(list(self.keyframes_info.keys()))
        }
        #保存字典到文件
        save_dict_to_ckpt(
            submap_ckpt, f"{submap_ckpt_name}.ckpt", directory=self.output_path / "submaps")
    
    def start_new_submap(self, frame_id: int, gaussian_model: GaussianModel) -> None:
        """ Initializes a new submap.
        This function updates the submap count and optionally marks the current frame ID for new submap initiation.
        Args:
            frame_id: The ID of the current frame at which the new submap is started.
            gaussian_model: The current GaussianModel instance to capture and reset for the new submap.
        Returns:
            A new, reset GaussianModel instance for the new submap.
        """
        
        gaussian_model = GaussianModel(0)
        gaussian_model.training_setup(self.opt)
        self.mapper.keyframes = []
        self.keyframes_info = {}
        if self.submap_using_motion_heuristic:
            self.new_submap_frame_ids.append(frame_id)
        self.mapping_frame_ids.append(frame_id) if frame_id not in self.mapping_frame_ids else self.mapping_frame_ids
        self.submap_id += 1
        self.loop_closer.submap_id += 1
        return gaussian_model
    
    #经过位姿图优化后，更新子图的高斯模型
    def rigid_transform_gaussians(self, gaussian_params, tsfm_matrix):
        '''
        Apply a rigid transformation to the Gaussian parameters.
        
        Args:
            gaussian_params (dict): Dictionary containing Gaussian parameters.
            tsfm_matrix (torch.Tensor): 4x4 rigid transformation matrix.
            
        Returns:
            dict: Updated Gaussian parameters after applying the transformation.
        '''
        # Transform Gaussian centers (xyz)
        tsfm_matrix = torch.from_numpy(tsfm_matrix).float()
        xyz = gaussian_params['xyz']
        pts_ones = torch.ones((xyz.shape[0], 1))
        pts_homo = torch.cat([xyz, pts_ones], dim=1)
        transformed_xyz = (tsfm_matrix @ pts_homo.T).T[:, :3]
        gaussian_params['xyz'] = transformed_xyz

        # Rotate covariance matrix (rotation)
        rotation = gaussian_params['rotation']
        cur_rot = roma.unitquat_to_rotmat(rotation)
        rot_mat = tsfm_matrix[:3, :3].unsqueeze(0)  # Adding batch dimension
        new_rot = rot_mat @ cur_rot
        new_quat = roma.rotmat_to_unitquat(new_rot)
        gaussian_params['rotation'] = new_quat.squeeze()

        return gaussian_params
    
    #经过位姿图优化后，更新关键帧的位姿
    def update_keyframe_poses(self, lc_output, submaps_kf_ids, cur_frame_id):
        '''
        Update the keyframe poses using the correction from pgo, currently update the frame range that covered by the keyframes.
        
        '''
        for correction in lc_output:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']
            submap_kf_ids = submaps_kf_ids[submap_id]
            min_id, max_id = min(submap_kf_ids), max(submap_kf_ids)
            self.estimated_c2ws[min_id:max_id + 1] = torch.from_numpy(correct_tsfm).float() @ self.estimated_c2ws[min_id:max_id + 1]
        
        # last tracked frame is based on last submap, update it as well
        self.estimated_c2ws[cur_frame_id] = torch.from_numpy(lc_output[-1]['correct_tsfm']).float() @ self.estimated_c2ws[cur_frame_id]
        
        
    def apply_correction_to_submaps(self, correction_list):
        submaps_kf_ids= {}
        for correction in correction_list:
            submap_id = correction['submap_id']
            correct_tsfm = correction['correct_tsfm']

            submap_ckpt_name = str(submap_id).zfill(6) + ".ckpt"
            submap_ckpt = torch.load(self.output_path / "submaps" / submap_ckpt_name)
            submaps_kf_ids[submap_id] = submap_ckpt["submap_keyframes"]

            gaussian_params = submap_ckpt["gaussian_params"]
            updated_gaussian_params = self.rigid_transform_gaussians(
                gaussian_params, correct_tsfm)

            submap_ckpt["gaussian_params"] = updated_gaussian_params
            torch.save(submap_ckpt, self.output_path / "submaps" / submap_ckpt_name)
        return submaps_kf_ids
    
    def run(self) -> None:
        """ Starts the main program flow for Gaussian-SLAM, including tracking and mapping. """
        setup_seed(self.config["seed"])
        #默认第一个创建子图
        gaussian_model = GaussianModel(0)#0阶球谐函数
        gaussian_model.training_setup(self.opt)
        self.submap_id = 0

        for frame_id in range(len(self.dataset)):
            print("Processing frame", frame_id,"/",len(self.dataset))
            #执行mini——ba进行位姿估计,第一帧返回的是GT位姿
            estimated_c2w, exposure_ab = self.tracker.track(
                frame_id, gaussian_model,
                torch2np(self.estimated_c2ws[torch.tensor([0, frame_id - 2, frame_id - 1])]))
            exposure_ab = exposure_ab if self.enable_exposure else None
            self.estimated_c2ws[frame_id] = np2torch(estimated_c2w)

            # Reinitialize gaussian model for new segment
            #判断是否开始新的子地图
            if self.should_start_new_submap(frame_id):
                # first save current submap and its keyframe info
                self.save_current_submap(gaussian_model)
                
                # update submap infomation for loop closer
                #在回环中更新子地图信息，主要是提取子地图中每个关键帧rgb图像的描述符，将它们拼接成一个大的描述符，
                #记录自相似程度，记录当前子图在回环中的id
                self.loop_closer.update_submaps_info(self.keyframes_info)
                
                # # apply loop closure
                # lc_output = self.loop_closer.loop_closure(self.estimated_c2ws)
                
                # if len(lc_output) > 0:
                #     submaps_kf_ids = self.apply_correction_to_submaps(lc_output)#校正子地图的高斯参数
                #     self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)#校正关键帧的位姿
                
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
                
                gaussian_model = self.start_new_submap(frame_id, gaussian_model)
            #建图关键帧
            if (frame_id in self.mapping_frame_ids):
                print("\nMapping frame", frame_id)
                print("It's in submap", self.submap_id)
                gaussian_model.training_setup(self.opt, exposure_ab) 
                estimate_c2w = torch2np(self.estimated_c2ws[frame_id])
                #检查新的子地图的关键帧信息是否为空，一开始是空的，还没有高斯被添加
                new_submap = not bool(self.keyframes_info)
                #使用当前帧信息更新子地图
                #这个opt_dict是这一帧的优化损失、优化时间、场景的重建结果
                opt_dict = self.mapper.map(
                    frame_id, estimate_c2w, gaussian_model, new_submap, exposure_ab)
                # Keyframes info update
                #frame_id是键，keyframe_id，opt_dict是对应的值
                self.keyframes_info[frame_id] = {
                    "keyframe_id": frame_id, 
                    "opt_dict": opt_dict,
                }
                if self.enable_exposure:
                    self.keyframes_info[frame_id]["exposure_a"] = exposure_ab[0].item()
                    self.keyframes_info[frame_id]["exposure_b"] = exposure_ab[1].item()
            # if frame_id == len(self.dataset) - 1 and self.config['lc']['final']:
            #     self.save_current_submap(gaussian_model)
            #     print("\n Final loop closure ...")
            #     self.loop_closer.update_submaps_info(self.keyframes_info)
            #     lc_output = self.loop_closer.loop_closure(self.estimated_c2ws)
            #     if len(lc_output) > 0:
            #         submaps_kf_ids = self.apply_correction_to_submaps(lc_output)
            #         self.update_keyframe_poses(lc_output, submaps_kf_ids, frame_id)
            if frame_id == len(self.dataset) - 1: 
                self.save_current_submap(gaussian_model)
                save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)

            if self.enable_exposure:
                self.exposures_ab[frame_id] = torch.tensor([exposure_ab[0].item(), exposure_ab[1].item()])
        save_dict_to_ckpt(self.estimated_c2ws[:frame_id + 1], "estimated_c2w.ckpt", directory=self.output_path)
        if self.enable_exposure:
            save_dict_to_ckpt(self.exposures_ab, "exposures_ab.ckpt", directory=self.output_path)
