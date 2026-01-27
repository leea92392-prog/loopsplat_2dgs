import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import json
import imageio
import trimesh
from typing import List, Union
import warnings

class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_config: dict):
        self.dataset_path = Path(dataset_config["data"]["input_path"])
        self.frame_limit = dataset_config.get("frame_limit", -1)
        self.dataset_config = dataset_config
        self.height = dataset_config["cam"]["H"]
        self.width = dataset_config["cam"]["W"]
        self.fx = dataset_config["cam"]["fx"]
        self.fy = dataset_config["cam"]["fy"]
        self.cx = dataset_config["cam"]["cx"]
        self.cy = dataset_config["cam"]["cy"]

        self.depth_scale = dataset_config["cam"]["depth_scale"]
        self.distortion = np.array(
            dataset_config["cam"]['distortion']) if 'distortion' in dataset_config["cam"] else None
        self.crop_edge = dataset_config["cam"]['crop_edge'] if 'crop_edge' in dataset_config["cam"] else 0
        if self.crop_edge:
            self.height -= 2 * self.crop_edge
            self.width -= 2 * self.crop_edge
            self.cx -= self.crop_edge
            self.cy -= self.crop_edge

        self.fovx = 2 * math.atan(self.width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.height / (2 * self.fy))
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.color_paths = []
        self.depth_paths = []

    def __len__(self):
        return len(self.color_paths) if self.frame_limit < 0 else int(self.frame_limit)


class Replica(BaseDataset):

    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(
            list((self.dataset_path / "results").glob("frame*.jpg")))
        self.depth_paths = sorted(
            list((self.dataset_path / "results").glob("depth*.png")))
        self.load_poses(self.dataset_path / "traj.txt")
        self.start = dataset_config["start_idx"]
        self.end = dataset_config["early_stop"]
        self.stride = dataset_config["stride"]
        self.n_img = len(self.color_paths)
        if self.end < 0:
            self.end = self.n_img
        self.color_paths = self.color_paths[self.start : self.end : self.stride]
        self.depth_paths = self.depth_paths[self.start : self.end : self.stride]
        self.poses = self.poses[self.start : self.end : self.stride]
        print(f"Loaded {len(self.color_paths)} frames")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            self.poses.append(c2w.astype(np.float32))

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index], None


class TUM_RGBD(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            self.dataset_path, frame_rate=32)
        self.start = dataset_config["start_idx"]
        self.end = dataset_config["early_stop"]
        self.stride = dataset_config["stride"]
        self.n_img = len(self.color_paths)
        if self.end < 0:
            self.end = self.n_img
        self.color_paths = self.color_paths[self.start : self.end : self.stride]
        self.depth_paths = self.depth_paths[self.start : self.end : self.stride]
        self.poses = self.poses[self.start : self.end : self.stride]

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))
            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt) and (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))
        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths = [], [], []
        inv_pose = None
        for ix in indicies:
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            if inv_pose is None:
                inv_pose = np.linalg.inv(c2w)
                c2w = np.eye(4)
            else:
                c2w = inv_pose@c2w
            poses += [c2w.astype(np.float32)]

        return images, depths, poses

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index], None


class ScanNet(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths = sorted(list(
            (self.dataset_path / "rgb").glob("*.png")), key=lambda x: int(os.path.basename(x)[-9:-4]))
        self.depth_paths = sorted(list(
            (self.dataset_path / "depth").glob("*.TIFF")), key=lambda x: int(os.path.basename(x)[-10:-5]))
        self.n_img = len(self.color_paths)
        self.load_poses(self.dataset_path / "gt_pose.txt")

    def load_poses(self, path):
        self.poses = []
        pose_data = np.loadtxt(path, delimiter=" ", dtype=np.unicode_, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)
        for i in range(self.n_img):
            quat = pose_vecs[i][4:]
            trans = pose_vecs[i][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans
            pose = T
            self.poses.append(pose)

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.dataset_config["W"], self.dataset_config["H"]))

        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        return index, color_data, depth_data, self.poses[index], None


class ScanNetPP(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.use_train_split = dataset_config["use_train_split"]
        self.train_test_split = json.load(open(f"{self.dataset_path}/dslr/train_test_lists.json", "r"))
        if self.use_train_split:
            self.image_names = self.train_test_split["train"]
        else:
            self.image_names = self.train_test_split["test"]
        self.load_data()

    def load_data(self):
        self.poses = []
        cams_path = self.dataset_path / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        cams_metadata = json.load(open(str(cams_path), "r"))
        frames_key = "frames" if self.use_train_split else "test_frames"
        frames_metadata = cams_metadata[frames_key]
        frame2idx = {frame["file_path"]: index for index, frame in enumerate(frames_metadata)}
        P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(np.float32)
        for image_name in self.image_names:
            frame_metadata = frames_metadata[frame2idx[image_name]]
            # if self.ignore_bad and frame_metadata['is_bad']:
            #     continue
            color_path = str(self.dataset_path / "dslr" / "undistorted_images" / image_name)
            depth_path = str(self.dataset_path / "dslr" / "undistorted_depths" / image_name.replace('.JPG', '.png'))
            self.color_paths.append(color_path)
            self.depth_paths.append(depth_path)
            c2w = np.array(frame_metadata["transform_matrix"]).astype(np.float32)
            c2w = P @ c2w @ P.T
            self.poses.append(c2w)

    def __len__(self):
        if self.use_train_split:
            return len(self.image_names) if self.frame_limit < 0 else int(self.frame_limit)
        else:
            return len(self.image_names)

    def __getitem__(self, index):

        color_data = np.asarray(imageio.imread(self.color_paths[index]), dtype=float)
        color_data = cv2.resize(color_data, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        color_data = color_data.astype(np.uint8)

        depth_data = np.asarray(imageio.imread(self.depth_paths[index]), dtype=np.int64)
        depth_data = cv2.resize(depth_data.astype(float), (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        return index, color_data, depth_data, self.poses[index],None

class UTMM(BaseDataset):
    def __init__(self, dataset_config: dict):
        super().__init__(dataset_config)
        self.color_paths, self.depth_paths, self.poses, self.imus, self.tstamps = self.loadtum(
            self.dataset_path, frame_rate=32)
        self.start = dataset_config["start_idx"]
        self.end = dataset_config["early_stop"]
        self.stride = dataset_config["stride"]
        self.n_img = len(self.color_paths)
        self.desired_height = dataset_config["desired_height"]
        self.desired_width = dataset_config["desired_width"]
        self.height_downsample_ratio = float(self.desired_height) / self.height
        self.width_downsample_ratio = float(self.desired_width) / self.width
        K = np.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K = self.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        self.fx , self.fy, self.cx, self.cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        self.fovx = 2 * math.atan(self.desired_width / (2 * self.fx))
        self.fovy = 2 * math.atan(self.desired_height / (2 * self.fy))
        self.width = self.desired_width
        self.height = self.desired_height
        self.intrinsics = np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        if self.end < 0:
            self.end = self.n_img
        self.color_paths = self.color_paths[self.start : self.end : self.stride]
        self.depth_paths = self.depth_paths[self.start : self.end : self.stride]
        self.poses = self.poses[self.start : self.end : self.stride]
        self.tstamps = self.tstamps[self.start : self.end : self.stride]
        self.use_imu = dataset_config["tracking"]["use_imu"]
        if self.use_imu:
            concat_imus = []
            idx = 0
            while idx < self.end:
                cat_entry = torch.empty(0)
                for i in range(self.stride):
                    if idx >= self.end:
                        break
                    imu_entry = self.imus[idx]
                    cat_entry = torch.cat([cat_entry, imu_entry], dim=0)
                    idx += 1
                concat_imus += [cat_entry]

            self.imus = concat_imus
        self.i2c = np.array([-0.01104814029330714,0.22456530118066098,-0.17573647511484858, -0.5, 0.5, -0.5, -0.5]).astype(np.float32)
    
    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        return np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose,tstamp_imu=None,max_dt=0.12):
        """ pair images, depths, and poses """
        associations = []
        lstart = 0
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None and tstamp_imu is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))
            elif tstamp_imu is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))
            else:  # imu and pose provided
                assert (
                    tstamp_pose is not None and tstamp_imu is not None
                ), "Both IMU and pose must be provided"
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))
                lend = np.argmin(np.abs(tstamp_imu - t))
                l = np.arange(lstart, lend + 1, step=1)
                if (
                    (np.abs(tstamp_depth[j] - t) < max_dt)
                    and (np.abs(tstamp_pose[k] - t) < max_dt)
                    and (np.abs(tstamp_imu[lend] - t) < max_dt)
                ):
                    associations.append((i, j, k, l))
                    lstart = lend + 1

        return associations

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')
        imu_list = os.path.join(datapath, "imu.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)
        imu_data = self.parse_list(imu_list)
        imu_vecs = imu_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        tstamp_imu = imu_data[:, 0].astype(np.float64)

        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose,tstamp_imu)

        images, poses, depths,imu_meas, tstamp = [], [], [],[], []
        inv_pose = None

        # for i, j, k, l in associations:
        #     images += [os.path.join(datapath, image_data[i, 1])]
        #     depths += [os.path.join(datapath, depth_data[j, 1])]
        #     c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
        #     if inv_pose is None:
        #         inv_pose = np.linalg.inv(c2w)
        #         c2w = np.eye(4)
        #     else:
        #         c2w = inv_pose @ c2w
        #     poses += [c2w.astype(np.float32)]
        #     imu_meas += [torch.from_numpy(imu_vecs[l, :]).float()]
        #     tstamp += [tstamp_image[i]]
        for i, j, k, l in associations:
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]
            c2w = self.pose_matrix_from_quaternion_pose(pose_vecs[k])
            poses += [c2w.astype(np.float32)]
            imu_meas += [torch.from_numpy(imu_vecs[l, :]).float()]
            tstamp += [tstamp_image[i]]
        return images, depths, poses, imu_meas, tstamp
    
    def pose_matrix_from_quaternion_pose(self, pvec):
        from scipy.spatial.transform import Rotation

        r2w = np.eye(4)
        r2w[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()

        # Convert from robot frame to camera optical frame
        # z forward, x right, y down
        c2r = np.eye(4)
        c2r[:3, :3] = Rotation.from_matrix(
            [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        ).as_matrix()
        r2w = r2w @ c2r  # r2w * c2r
        r2w[:3, 3] = pvec[:3]
        return r2w

    # def pose_matrix_from_quaternion_pose(self, pvec):
    #     """convert 4x4 pose matrix to (t, q)"""
    #     from scipy.spatial.transform import Rotation

    #     r2w = np.eye(4)
    #     r2w[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
    #     r2w[:3, 3] = pvec[:3]
    #     # Convert from robot frame to camera optical frame
    #     # z forward, x right, y down
    #     c2r = np.eye(4)
    #     c2r[:3, :3] = Rotation.from_matrix(
    #         [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    #     ).as_matrix()
    #     c2w = r2w @ c2r  # r2w * c2r
    #     return c2w

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose
    
    def scale_intrinsics(
        self,
        intrinsics: Union[np.ndarray, torch.Tensor],
        h_ratio: Union[float, int],
        w_ratio: Union[float, int],
    ):
        r"""Scales the intrinsics appropriately for resized frames where
        :math:`h_\text{ratio} = h_\text{new} / h_\text{old}` and :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

        Args:
            intrinsics (numpy.ndarray or torch.Tensor): Intrinsics matrix of original frame
            h_ratio (float or int): Ratio of new frame's height to old frame's height
                :math:`h_\text{ratio} = h_\text{new} / h_\text{old}`
            w_ratio (float or int): Ratio of new frame's width to old frame's width
                :math:`w_\text{ratio} = w_\text{new} / w_\text{old}`

        Returns:
            numpy.ndarray or torch.Tensor: Intrinsics matrix scaled approprately for new frame size

        Shape:
            - intrinsics: :math:`(*, 3, 3)` or :math:`(*, 4, 4)`
            - Output: Matches `intrinsics` shape, :math:`(*, 3, 3)` or :math:`(*, 4, 4)`

        """
        if isinstance(intrinsics, np.ndarray):
            scaled_intrinsics = intrinsics.astype(np.float32).copy()
        elif torch.is_tensor(intrinsics):
            scaled_intrinsics = intrinsics.to(torch.float).clone()
        else:
            raise TypeError("Unsupported input intrinsics type {}".format(type(intrinsics)))
        if not (intrinsics.shape[-2:] == (3, 3) or intrinsics.shape[-2:] == (4, 4)):
            raise ValueError(
                "intrinsics must have shape (*, 3, 3) or (*, 4, 4), but had shape {} instead".format(
                    intrinsics.shape
                )
            )
        if (intrinsics[..., -1, -1] != 1).any() or (intrinsics[..., 2, 2] != 1).any():
            warnings.warn(
                "Incorrect intrinsics: intrinsics[..., -1, -1] and intrinsics[..., 2, 2] should be 1."
            )

        scaled_intrinsics[..., 0, 0] *= w_ratio  # fx
        scaled_intrinsics[..., 1, 1] *= h_ratio  # fy
        scaled_intrinsics[..., 0, 2] *= w_ratio  # cx
        scaled_intrinsics[..., 1, 2] *= h_ratio  # cy
        return scaled_intrinsics
    
    def get_c2i_tf(self):
        """Get the transformation matrix from camera optical frame to IMU frame"""
        # tf_list = os.path.join(self.dataset_path, "tf.txt")

        # tf_data = self.parse_list(tf_list).astype(np.float64)
        tf_data = self.i2c
        # Convert translation+quaternion to homogeneous matrix
        i2c = self.pose_matrix_from_quaternion(tf_data)
        c2i = np.linalg.inv(i2c)

        return torch.from_numpy(c2i).float().to("cuda")    

    def __getitem__(self, index):
        color_data = cv2.imread(str(self.color_paths[index]))
        if self.distortion is not None:
            color_data = cv2.undistort(
                color_data, self.intrinsics, self.distortion)
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = cv2.resize(color_data, (self.desired_width, self.desired_height))
        depth_data = cv2.imread(
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)
        depth_data = cv2.resize(depth_data, (self.desired_width, self.desired_height), interpolation=cv2.INTER_NEAREST)
        depth_data = depth_data.astype(np.float32) / self.depth_scale
        edge = self.crop_edge
        # if edge > 0:
        #     color_data = color_data[edge:-edge, edge:-edge]
        #     depth_data = depth_data[edge:-edge, edge:-edge]
        # Interpolate depth values for splatting
        if self.use_imu:
            imu = self.imus[index].to("cuda").type(torch.float)
            return index, color_data, depth_data,self.poses[index], imu
        else:
            return index, color_data, depth_data,self.poses[index], None
class ZED720(BaseDataset):  
    def __init__(self, dataset_config: dict):  
        super().__init__(dataset_config)  
          
        # 加载场景目录下的images和depth目录  
        scene_path = self.dataset_path  
        self.color_paths = sorted(  
            list((scene_path / "images").glob("*")),  
            key=lambda x: int(x.stem)  
        )  
        self.depth_paths = sorted(  
            list((scene_path / "depth").glob("*")),  
            key=lambda x: int(x.stem)  
        )  
          
        # 读取配置参数  
        self.start = dataset_config["start_idx"]  
        self.end = dataset_config["early_stop"]  
        self.stride = dataset_config["stride"]  
        self.n_img = len(self.color_paths)  
          
        # 处理 early_stop 参数  
        if self.end < 0:  
            self.end = self.n_img  
          
        # 应用 start, end, stride 切片  
        self.color_paths = self.color_paths[self.start : self.end : self.stride]  
        self.depth_paths = self.depth_paths[self.start : self.end : self.stride]  
          
        # 初始化位姿为单位阵  
        n_frames = len(self.color_paths)  
        self.poses = [np.eye(4, dtype=np.float32) for _ in range(n_frames)]  
          
        print(f"Loaded {len(self.color_paths)} frames from ZED720 dataset")  
      
    def __getitem__(self, index):  
        color_data = cv2.imread(str(self.color_paths[index]))  
        if self.distortion is not None:  
            color_data = cv2.undistort(  
                color_data, self.intrinsics, self.distortion)  
        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)  
          
        depth_data = cv2.imread(  
            str(self.depth_paths[index]), cv2.IMREAD_UNCHANGED)  
        depth_data = depth_data.astype(np.float32) / self.depth_scale  
          
        edge = self.crop_edge  
        if edge > 0:  
            color_data = color_data[edge:-edge, edge:-edge]  
            depth_data = depth_data[edge:-edge, edge:-edge]  
          
        return index, color_data, depth_data, self.poses[index], None

def get_dataset(dataset_name: str):
    if dataset_name == "replica":
        return Replica
    elif dataset_name == "tum_rgbd":
        return TUM_RGBD
    elif dataset_name == "scan_net":
        return ScanNet
    elif dataset_name == "scannetpp":
        return ScanNetPP
    elif dataset_name == "utmm":
        return UTMM
    elif dataset_name == "zed720":
        return ZED720
    raise NotImplementedError(f"Dataset {dataset_name} not implemented")
