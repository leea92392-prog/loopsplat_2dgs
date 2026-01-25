import numpy as np  
import torch  
from src.pose_utils.feature_detector import Detector  
from src.pose_utils.matcher import Matcher  
from src.pose_utils.pose_initializer import PoseInitializer  
from src.pose_utils.utils import depth2points  
  
class PoseUtilsAdapter:  
    def __init__(self, config, dataset):  
        self.dataset = dataset  
        top_k = config.get('top_k', 1000)  # 默认值1000  
        self.feature_detector = Detector(top_k, dataset.width, dataset.height)  
        fundmat_samples = config.get('fundmat_samples', 1000)  # RANSAC采样数  
        max_error = config.get('max_error', 1.0)  # 最大误差阈值  
        self.matcher = Matcher(fundmat_samples, max_error)           
        # 初始化pose_initializer，适配loopsplat_2dgs参数  
        args = type('Args', (), {  
            'num_pts_miniba_incr': config.get('num_pts_miniba_incr', 500),  
            'min_num_inliers': config.get('min_num_inliers', 30),  
            'pnpransac_samples': config.get('pnpransac_samples', 1000),  
            'iters_miniba_incr': config.get('iters_miniba_incr', 10)  
        })()  
          
        self.pose_initializer = PoseInitializer(  
            dataset.width, dataset.height, None, self.matcher, 4.0, args  
        )  
          
        # 关键帧历史  
        self.keyframes = []  
      
    def estimate_pose(self, frame_id, image, depth, intrinsics):  
        """估计当前帧位姿"""  
        # 提取特征  
        desc_kpts = self.feature_detector(image)  
          
        if frame_id < 2:  
            # 第一帧使用GT位姿  
            estimated_pose = self.dataset[frame_id][3]
            # 保存关键帧信息
            self.keyframes.append({  
                'frame_id': frame_id,  
                'desc_kpts': desc_kpts,  
                'pose': estimated_pose,  
                'pts3d': depth2points(desc_kpts.kpts, depth, intrinsics[0,0],   
                                    torch.tensor([(self.dataset.width-1)/2, (self.dataset.height-1)/2])),  
                'conf': desc_kpts.pts_conf  
            })
            return estimated_pose  
          
        # 使用pose_utils进行位姿估计  
        estimated_pose = self.pose_initializer.initialize_from_depth(  
            desc_kpts, depth, intrinsics, self.keyframes  
        )  
          
        if estimated_pose is not None:  
            # 保存关键帧信息  
            self.keyframes.append({  
                'frame_id': frame_id,  
                'desc_kpts': desc_kpts,  
                'pose': estimated_pose,  
                'pts3d': depth2points(desc_kpts.kpts, depth, intrinsics[0,0],   
                                    torch.tensor([(self.dataset.width-1)/2, (self.dataset.height-1)/2])),  
                'conf': desc_kpts.pts_conf  
            })  
              
            # 保持关键帧数量在合理范围  
            if len(self.keyframes) > 10:  
                self.keyframes = self.keyframes[-10:]  
                  
            return estimated_pose  
        else:  
            # 追踪失败，使用上一帧位姿  
            return self.keyframes[-1]['pose'] if self.keyframes else self.dataset[frame_id][3]