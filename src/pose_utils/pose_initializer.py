#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math

from src.pose_utils.feature_detector import DescribedKeypoints
from src.pose_utils.mini_ba import MiniBA
from src.pose_utils.utils import fov2focal, depth2points, sixD2mtx
# from scene.keyframe import Keyframe
from src.pose_utils.ransac import RANSACEstimator, EstimatorType

class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher
        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda').float()
        # self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        # self.num_kpts = args.num_kpts
        # self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        # self.num_pts_miniba_incr = args.num_pts_miniba_incr
        # self.min_num_inliers = args.min_num_inliers
        # # Initialize the focal length
        # if args.init_focal > 0:
        #     self.f_init = args.init_focal
        # elif args.init_fov > 0:
        #     self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        # else:
        #     self.f_init = 0.7 * width
        # # Initialize MiniBA models
        # self.miniba_bootstrap = MiniBA(
        #     1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
        #     make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        # self.miniba_rebooting = MiniBA(
        #     1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
        #     make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        # self.miniBA_incr = MiniBA(
        #     1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
        #     make_cuda_graph=True, iters=args.iters_miniba_incr)
        # self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)
        # 直接使用深度图的焦距，不需要bootstrap  
        self.num_kpts = 1000
        self.num_pts_miniba_incr = args.num_pts_miniba_incr  
        self.min_num_inliers = args.min_num_inliers  
        # 只初始化增量式MiniBA  
        self.miniBA_incr = MiniBA(  
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,  
            make_cuda_graph=True, iters=args.iters_miniba_incr)  
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)  

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    
    # @torch.no_grad()
    # def initialize_incremental(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int, is_test: bool, curr_img):
    #     """
    #     Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
    #     """
        
    #     # Match the current frame with previous keyframes
    #     xyz = []
    #     uvs = []
    #     confs = []
    #     match_indices = []
    #     for keyframe in keyframes:
    #         matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)

    #         mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
    #         xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
    #         uvs.append(matches.kpts[mask])
    #         confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
    #         match_indices.append(matches.idx[mask])

    #     xyz = torch.cat(xyz, dim=0)
    #     uvs = torch.cat(uvs, dim=0)
    #     confs = torch.cat(confs, dim=0)
    #     match_indices = torch.cat(match_indices, dim=0)

    #     # Subsample the points if there are too many
    #     if len(xyz) > self.num_pts_pnpransac:
    #         selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
    #         xyz = xyz[selected_indices]
    #         uvs = uvs[selected_indices]
    #         confs = confs[selected_indices]
    #         match_indices = match_indices[selected_indices]

    #     # Estimate an initial camera pose and inliers using PnP RANSAC
    #     Rs6D_init = keyframes[0].rW2C
    #     ts_init = keyframes[0].tW2C
    #     Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)

    #     xyz = xyz[inliers]
    #     uvs = uvs[inliers]
    #     confs = confs[inliers]
    #     match_indices = match_indices[inliers]

    #     # Subsample the points if there are too many
    #     if len(xyz) >= self.num_pts_miniba_incr:
    #         selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
    #         xyz_ba = xyz[selected_indices]
    #         uvs_ba = uvs[selected_indices]
    #     elif len(xyz) < self.num_pts_miniba_incr:
    #         xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
    #         uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

    #     # Run the initialization
    #     Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
    #     Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
    #     Rt = torch.eye(4, device="cuda")
    #     Rt[:3, :3] = sixD2mtx(Rs6D)[0]
    #     Rt[:3, 3] = ts[0]

    #     # Check if we have sufficiently many inliers
    #     if is_test or mask.sum() > self.min_num_inliers:
    #         # Return the pose of the current frame
    #         return Rt
    #     else:
    #         print("Too few inliers for pose initialization")
    #         # Remove matches as we prevent the current frame from being registered
    #         for keyframe in keyframes:
    #             keyframe.desc_kpts.matches.pop(index, None)
    #         return None
    @torch.no_grad()    
    def initialize_from_depth(self, curr_desc_kpts, depth, intrinsics, cur_frame_id, prev_keyframes, xyz=None):    
        """直接从深度图和关键帧进行位姿估计，跳过bootstrap  
        
        Args:  
            curr_desc_kpts: 当前帧的特征点  
            depth: 深度图  
            intrinsics: 相机内参  
            cur_frame_id: 当前帧ID  
            prev_keyframes: 历史关键帧列表  
            xyz: 可选的预计算3D点，如果为None则重新计算  
        """  
        if depth.dtype != torch.float32:    
            depth = depth.float()    
        if curr_desc_kpts.kpts.dtype != torch.float32:    
            curr_desc_kpts.kpts = curr_desc_kpts.kpts.float()    


        # 与之前的关键帧匹配  
        matched_xyz = []  
        matched_uvs = []  
        confs = []  
          
        for keyframe in prev_keyframes[-3:]:  # 使用最近2个关键帧  
            matches = self.matcher(curr_desc_kpts, keyframe['desc_kpts'],   
                                 remove_outliers=True, update_kpts_flag="all", kID=cur_frame_id, kID_other=keyframe['frame_id'])  
            print(f"Frame {cur_frame_id} matched with Keyframe {keyframe['frame_id']}: {len(matches.kpts)} inliers.")
            if len(matches.kpts) > self.min_num_inliers:  
                matched_xyz.append(keyframe['pts3d'][matches.idx_other])  
                matched_uvs.append(matches.kpts)  
                confs.append(keyframe['conf'][matches.idx_other])  
        if not matched_xyz:  
            return None  
              
        # PnP RANSAC + MiniBA优化  
        xyz = torch.cat(matched_xyz, dim=0)  
        uvs = torch.cat(matched_uvs, dim=0)  
        confs = torch.cat(confs, dim=0)  
        print(f"Total matched points for PnP RANSAC: {len(xyz)}")
        print(f"Total matched uvs for PnP RANSAC: {len(uvs)}")
        # 使用上关键帧的位姿作为初始化  
        # 这个位姿是世界到相机的变换矩阵
        init_pose = prev_keyframes[-1]['pose']  
        Rs6D_init = torch.from_numpy(init_pose[:3, :2]).cuda().float()  
        ts_init = torch.from_numpy(init_pose[:3, 3]).cuda().float()  
        
        # 检查是否有异常值  
        if torch.isnan(xyz).any() or torch.isinf(xyz).any():  
            print("WARNING: xyz contains NaN or Inf!")  
        if torch.isnan(uvs).any() or torch.isinf(uvs).any():  
            print("WARNING: uvs contains NaN or Inf!")  
        Rt, inliers = self.PnPRANSAC(uvs, xyz, intrinsics[0,0], self.centre, Rs6D_init, ts_init, None)  
        print("PnP RANSAC Rt:\n", Rt.cpu().numpy())
        if len(inliers) < self.min_num_inliers:  
            return None  
        # MiniBA细化  
        xyz_inliers = xyz[inliers]  
        uvs_inliers = uvs[inliers]  
        print("MiniBA input xyz points: ", xyz_inliers.shape)
        print("MiniBA input uv points: ", uvs_inliers.shape)
        if len(xyz_inliers) == 0 or len(uvs_inliers) == 0:  
            print("Warning: Empty inliers after PnP RANSAC")  
            return None  
  
        if len(xyz_inliers) != len(uvs_inliers):  
            print(f"Warning: Mismatched tensor lengths: xyz={len(xyz_inliers)}, uvs={len(uvs_inliers)}")  
            min_len = min(len(xyz_inliers), len(uvs_inliers))  
            xyz_inliers = xyz_inliers[:min_len]  
            uvs_inliers = uvs_inliers[:min_len]    
        # 参考原生实现: 处理点数不足的情况  
        if len(xyz_inliers) >= self.num_pts_miniba_incr:  
            # 随机采样到固定数量  
            selected_indices = torch.topk(  
                torch.rand_like(xyz_inliers[..., 0]),   
                self.num_pts_miniba_incr,   
                dim=0,   
                largest=False  
            )[1]  
            xyz_ba = xyz_inliers[selected_indices]  
            uvs_ba = uvs_inliers[selected_indices]  
        else:  
            # 填充到固定大小  
            num_padding = self.num_pts_miniba_incr - len(xyz_inliers)  
            xyz_ba = torch.cat([  
                xyz_inliers,   
                torch.zeros(num_padding, 3, device="cuda")  
            ], dim=0)  
            uvs_ba = torch.cat([  
                uvs_inliers,   
                -torch.ones(num_padding, 2, device="cuda")  # -1 标记为无效  
            ], dim=0)  
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(  
            Rs6D, ts, intrinsics[0,0],   
            xyz_ba, self.centre, uvs_ba.view(-1) )  
          
        # 构建最终位姿  
        final_pose = torch.eye(4, device="cuda")  
        final_pose[:3, :3] = sixD2mtx(Rs6D)[0]  
        final_pose[:3, 3] = ts[0]  
        final_pose = torch.inverse(final_pose)  # miniBA输出的是w2c，需要取逆得到c2w
        return final_pose.cpu().numpy()