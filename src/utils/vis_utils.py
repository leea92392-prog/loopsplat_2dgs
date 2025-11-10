from collections import OrderedDict
from copy import deepcopy
from typing import List, Union
import os
import numpy as np
import open3d as o3d
import cv2
from matplotlib import colors

COLORS_ANSI = OrderedDict({
    "blue": "\033[94m",
    "orange": "\033[93m",
    "green": "\033[92m",
    "red": "\033[91m",
    "purple": "\033[95m",
    "brown": "\033[93m",  # No exact match, using yellow
    "pink": "\033[95m",
    "gray": "\033[90m",
    "olive": "\033[93m",  # No exact match, using yellow
    "cyan": "\033[96m",
    "end": "\033[0m",  # Reset color
})


COLORS_MATPLOTLIB = OrderedDict({
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'yellow-green': '#bcbd22',
    'cyan': '#17becf'
})


COLORS_MATPLOTLIB_RGB = OrderedDict({
    'blue': np.array([31, 119, 180]) / 255.0,
    'orange': np.array([255, 127,  14]) / 255.0,
    'green': np.array([44, 160,  44]) / 255.0,
    'red': np.array([214,  39,  40]) / 255.0,
    'purple': np.array([148, 103, 189]) / 255.0,
    'brown': np.array([140,  86,  75]) / 255.0,
    'pink': np.array([227, 119, 194]) / 255.0,
    'gray': np.array([127, 127, 127]) / 255.0,
    'yellow-green': np.array([188, 189,  34]) / 255.0,
    'cyan': np.array([23, 190, 207]) / 255.0
})


def get_color(color_name: str):
    """ Returns the RGB values of a given color name as a normalized numpy array.
    Args:
        color_name: The name of the color. Can be any color name from CSS4_COLORS.
    Returns:
        A numpy array representing the RGB values of the specified color, normalized to the range [0, 1].
    """
    if color_name == "custom_yellow":
        return np.asarray([255.0, 204.0, 102.0]) / 255.0
    if color_name == "custom_blue":
        return np.asarray([102.0, 153.0, 255.0]) / 255.0
    assert color_name in colors.CSS4_COLORS
    return np.asarray(colors.to_rgb(colors.CSS4_COLORS[color_name]))


def plot_ptcloud(point_clouds: Union[List, o3d.geometry.PointCloud], show_frame: bool = True):
    """ Visualizes one or more point clouds, optionally showing the coordinate frame.
    Args:
        point_clouds: A single point cloud or a list of point clouds to be visualized.
        show_frame: If True, displays the coordinate frame in the visualization. Defaults to True.
    """
    # rotate down up
    if not isinstance(point_clouds, list):
        point_clouds = [point_clouds]
    if show_frame:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        point_clouds = point_clouds + [mesh_frame]
    o3d.visualization.draw_geometries(point_clouds)


def draw_registration_result_original_color(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                                            transformation: np.ndarray):
    """ Visualizes the result of a point cloud registration, keeping the original color of the source point cloud.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
    """
    source_temp = deepcopy(source)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target])


def draw_registration_result(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud,
                             transformation: np.ndarray, source_color: str = "blue", target_color: str = "orange"):
    """ Visualizes the result of a point cloud registration, coloring the source and target point clouds.
    Args:
        source: The source point cloud.
        target: The target point cloud.
        transformation: The transformation matrix applied to the source point cloud.
        source_color: The color to apply to the source point cloud. Defaults to "blue".
        target_color: The color to apply to the target point cloud. Defaults to "orange".
    """
    source_temp = deepcopy(source)
    source_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[source_color])

    target_temp = deepcopy(target)
    target_temp.paint_uniform_color(COLORS_MATPLOTLIB_RGB[target_color])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def show_render_result(gt_rgb=None, gt_depth=None,est_depth=None,est_depth_scaled=None,render_rgb=None,render_depth=None,
                       render_normal=None,gt_normal=None,render_alpha=None ,save_id=None,save_path=None):
    if gt_rgb is not None:
        gt_image_show = gt_rgb.detach().cpu().numpy().transpose(1, 2, 0)
        gt_image_show = gt_image_show/np.max(gt_image_show)
        gt_image_show = (gt_image_show*255).astype(np.uint8)
        gt_image_show = cv2.cvtColor(gt_image_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", gt_image_show)
    if gt_depth is not None:
        depth_show = gt_depth.detach().cpu().numpy()*5000
        depth_show = depth_show.astype(np.uint16)
        depth_image_8bit =  cv2.normalize(depth_show, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 16位转8位
        gt_depth_color_map = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
        cv2.imshow("depth", gt_depth_color_map)
    if est_depth is not None:
        est_depth_show = (1/est_depth).cpu().numpy()
        est_depth_show = est_depth_show/np.max(est_depth_show)
        est_depth_show = (est_depth_show*255).astype(np.uint8)
        cv2.imshow("est_depth", est_depth_show)
    if est_depth_scaled is not None:
        est_depth_scaled_show = est_depth_scaled.cpu().numpy()
        est_depth_scaled_show = est_depth_scaled_show/np.max(est_depth_scaled_show)
        est_depth_scaled_show = (est_depth_scaled_show*255).astype(np.uint8)
        cv2.imshow("est_depth_scaled", est_depth_scaled_show)
    if render_rgb is not None:
        image_show = render_rgb.detach().cpu().numpy().transpose(1, 2, 0)
        image_show = image_show/np.max(image_show)
        image_show = (image_show*255).astype(np.uint8)
        image_show = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
        cv2.imshow("render_rgb", image_show)
    if render_depth is not None:
        depth_show = render_depth.detach().cpu().numpy() * 5000
        depth_show = depth_show.astype(np.uint16)
        depth_show = np.squeeze(depth_show)  # 去除维度为1的维度
        depth_image_8bit =  cv2.normalize(depth_show, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # 16位转8位
        render_depth_color_map = cv2.applyColorMap(depth_image_8bit, cv2.COLORMAP_JET)
        cv2.imshow("render_depth", render_depth_color_map)
    if render_normal is not None:
        normal_show = render_normal.detach().cpu().numpy().transpose(1, 2, 0)
        rgb_norms = np.zeros((normal_show.shape[0], normal_show.shape[1], 3), dtype=np.uint8)
        rgb_norms[..., 0] = ((normal_show[..., 0] + 1) * 127.5).astype(np.uint8)
        rgb_norms[..., 1] = ((normal_show[..., 1] + 1) * 127.5).astype(np.uint8)
        rgb_norms[..., 2] = ((normal_show[..., 2] + 1) * 127.5).astype(np.uint8)
        rgb_norms = cv2.cvtColor(rgb_norms, cv2.COLOR_RGB2BGR)
        cv2.imshow("render_normal", rgb_norms)
    if gt_normal is not None:
        gt_normal_show = gt_normal.detach().cpu().numpy()
        gt_rgb_norms = np.zeros((gt_normal_show.shape[0], gt_normal_show.shape[1], 3), dtype=np.uint8)
        gt_rgb_norms[..., 0] = ((gt_normal_show[..., 0] + 1) * 127.5).astype(np.uint8)
        gt_rgb_norms[..., 1] = ((gt_normal_show[..., 1] + 1) * 127.5).astype(np.uint8)
        gt_rgb_norms[..., 2] = ((gt_normal_show[..., 2] + 1) * 127.5).astype(np.uint8)
        gt_rgb_norms = cv2.cvtColor(gt_rgb_norms, cv2.COLOR_RGB2BGR)
        cv2.imshow("gt_normal", gt_rgb_norms)
    if render_alpha is not None:
        alpha_show = render_alpha.detach().cpu().numpy()
        alpha_show = alpha_show/np.max(alpha_show)
        alpha_show = (alpha_show*255).astype(np.uint8)
        alpha_show = np.squeeze(alpha_show)
        cv2.imshow("render_alpha", alpha_show)
    if save_id is not None:
        cv2.imwrite(os.path.join(save_path, f"gt_rgb_{save_id}.png"), gt_image_show)
        cv2.imwrite(os.path.join(save_path, f"render_depth_{save_id}.png"), render_depth_color_map)
        cv2.imwrite(os.path.join(save_path, f"gt_depth_{save_id}.png"), gt_depth_color_map)
        cv2.imwrite(os.path.join(save_path, f"render_norm_{save_id}.png"), rgb_norms)
    cv2.waitKey(1)
    return
