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
import torch.nn as nn
import torch.nn.functional as F
import os

from src.pose_utils.matcher import Matches

class DescribedKeypoints():
    """
    A class to store 2D keypoints, their descriptors, their matches to other images, and their estimated 3D positions.
    """
    @torch.no_grad()
    def __init__(self, kpts, feats):
        self.kpts = kpts # [N, 2] 关键点的 2D 像素坐标
        self.feats = feats # [N, D] 关键点的描述子
        self.valid = feats.abs().sum(dim=-1) > 0 # [N] 有效关键点掩码
        self.nvalid = self.valid.sum() #
        self.has_pt3d = torch.zeros(kpts.shape[0], dtype=torch.bool, device=kpts.device) # [N] 是否已三角化为 3D 点
        self.pts_conf = torch.ones(kpts.shape[0], dtype=torch.float, device=kpts.device) # [N] 3D 点置信度
        self.pts3d = torch.zeros(kpts.shape[0], 3, dtype=torch.float, device=kpts.device) # [N, 3] 3D 点坐标
        self.depth = torch.zeros(kpts.shape[0], dtype=torch.float, device=kpts.device) # [N] 关键点深度
        self.matches = {} # Dict[int, Matches] 与其他图像的匹配结果，键为其他图像的 ID

    @torch.no_grad()
    def update_matches(self, other_id, matches, swap=False):
        if swap:
            matches = Matches(matches.kpts_other, matches.kpts, matches.idx_other, matches.idx)
        self.matches[other_id] = matches
 
    @torch.no_grad()
    def update_3D_pts(self, pts3D, depth, conf, idx):
        self.has_pt3d[idx] = True
        self.depth[idx] = depth
        self.pts3d[idx] = pts3D
        self.pts_conf[idx] = conf

    @torch.no_grad()
    def to(self, device):
        self.kpts = self.kpts.to(device)
        self.feats = self.feats.to(device)
        self.valid = self.valid.to(device)
        self.has_pt3d = self.has_pt3d.to(device)
        self.pts_conf = self.pts_conf.to(device)
        self.pts3d = self.pts3d.to(device)
        self.depth = self.depth.to(device)
        for key in self.matches:
            self.matches[key].kpts = self.matches[key].kpts.to(device)
            self.matches[key].kpts_other = self.matches[key].kpts_other.to(device)
            self.matches[key].idx = self.matches[key].idx.to(device)
            self.matches[key].idx_other = self.matches[key].idx_other.to(device)

# From https://github.com/verlab/accelerated_features
# XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
class InterpolateSparse2d(nn.Module):
    """ Efficiently interpolate tensor at given sparse 2D positions. """ 
    def __init__(self, mode = 'bicubic', align_corners = False): 
        super().__init__()
        self.mode = mode
        self.align_corners = align_corners

    def normgrid(self, x, H, W):
        """ Normalize coords to [-1,1]. """
        return 2. * (x/(torch.tensor([W-1, H-1], device = x.device, dtype = x.dtype))) - 1.

    def forward(self, x, pos, H, W):
        """
        Input
            x: [B, C, H, W] feature tensor
            pos: [B, N, 2] tensor of positions
            H, W: int, original resolution of input 2d positions -- used in normalization [-1,1]

        Returns
            [B, N, C] sampled channels at 2d positions
        """
        grid = self.normgrid(pos, H, W).unsqueeze(-2).to(x.dtype)
        x = F.grid_sample(x, grid, mode = self.mode , align_corners = False)
        return x.permute(0,2,3,1).squeeze(-2)
    
class Detector():
    @torch.no_grad()
    def __init__(self, top_k, width, height):
        cache_path = f"models/cache/xfeat_{width}_{height}_{top_k}.pt"
        valid_height = max(32, (height // 32) * 32)  
        valid_width = max(32, (width // 32) * 32)  
        dummy_img = torch.randn(1, 3, valid_height, valid_width).cuda().to(torch.half)
        if os.path.exists(cache_path):
            extractor = torch.jit.load(cache_path)
        else:
            print(f"Compiling feature extractor")
            extractor = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=top_k)
            extractor = extractor.cuda().eval().to(torch.half)

            ## Overriding the functions to run at fixed size
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            #图像张量预处理，尺寸调整为32的倍数（XFeat网络的要求）
            def preprocess_tensor(x):
                H, W = x.shape[-2:]
                _H, _W = (H//32) * 32, (W//32) * 32
                rh, rw = H/_H, W/_W

                x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
                return x, rh, rw
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            # 非极大值抑制，从热力图中提取局部最大值作为关键点候选
            def NMS(x, threshold = 0.05, kernel_size = 5, nvalid = int(1.5 * top_k)):
                B, _, H, W = x.shape
                pad=kernel_size//2
                local_max = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=pad)(x)
                xTot = x * (x == local_max) * (x > threshold)
                xOut, pos1d = torch.topk(xTot.view(-1), nvalid)
                pos2d = torch.zeros((B, nvalid, 2), dtype=torch.long, device=x.device)
                pos2d[..., 0] = pos1d % W
                pos2d[..., 1] = pos1d // W
                return xOut, pos2d
            
            # Adapted from https://github.com/verlab/accelerated_features to enable jit tracing
            # XFeat: Accelerated Features for Lightweight Image Matching, Under Apache-2.0 license
            # 提取特征点并计算描述子的主函数
            def detectAndCompute(x):
                top_k = extractor.top_k
                detection_threshold = extractor.detection_threshold
                x, rh1, rw1 = extractor.preprocess_tensor(x)

                B, _, _H1, _W1 = x.shape
                
                M1, K1, H1 = extractor.net(x) # M1：描述子张量，K1：关键点热力图，H1：可靠性热力图
                M1 = F.normalize(M1, dim=1)

                #Convert logits to heatmap and extract kpts
                #转换后的 K1h 是一个归一化的热力图,值域在 [0, 1] 之间,表示每个像素位置是关键点的概率。
                K1h = extractor.get_kpts_heatmap(K1) 
                #在K1h热力图上进行非极大值抑制，提取关键点位置
                xOut, mkpts = extractor.NMS(K1h, threshold=detection_threshold, kernel_size=5)

                #Compute reliability scores
                _nearest = InterpolateSparse2d('nearest')
                _bilinear = InterpolateSparse2d('bilinear')
                scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
                scores[torch.all(mkpts == 0, dim=-1)] = -1

                #Select top-k features
                # idxs = torch.argsort(-scores)
                idxs = torch.topk(scores, top_k)[1]
                mkpts_x  = torch.gather(mkpts[...,0], -1, idxs)
                mkpts_y  = torch.gather(mkpts[...,1], -1, idxs)
                xOut = torch.gather(xOut[None], -1, idxs)
                mkpts = torch.cat([mkpts_x[...,None], mkpts_y[...,None]], dim=-1)
                scores = torch.gather(scores, -1, idxs)
                scores *= xOut > 0

                #Interpolate descriptors at kpts positions
                # 使用双线性插值在关键点位置提取特征描述子(原因是M1的空间分辨率较低，所得到的描述子位置可能不是整数像素位置)
                feats = extractor.interpolator(M1, mkpts, H = _H1, W = _W1)

                #L2-Normalize
                feats = F.normalize(feats, dim=-1)

                #Correct kpt scale
                mkpts = mkpts.float() * torch.tensor([rw1,rh1], device=mkpts.device, dtype=torch.float).view(1, 1, -1)

                return mkpts[0], feats[0] * (scores[0][..., None] > 0)
            # 重写了模型的三个关键函数
            extractor.preprocess_tensor = preprocess_tensor
            extractor.NMS = NMS
            extractor.forward = detectAndCompute
            # 重新编译模型
            extractor = torch.jit.trace(extractor, [dummy_img])
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.jit.save(extractor, cache_path)
            extractor = torch.jit.load(cache_path)

        self.extractor = extractor

    @torch.no_grad()
    def __call__(self, image):
        return DescribedKeypoints(*(self.extractor(image[None].half())))
    