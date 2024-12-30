import open3d as o3d
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
mesh_path='/home/lee/A-LOAM-Container/LoopSplat/output/TUM_RGBD/rgbd_dataset_freiburg1_desk/rgbd_dataset_freiburg1_desk_global_splats.ply'
def create_ellipsoid(center, scales, quaternion, resolution=20):
    """
    创建一个椭球，位置为 center，缩放为 scales，旋转由 quaternion 提供
    """
    # 创建一个单位球体
    ellipsoid = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=resolution)
    
    # 缩放椭球
    ellipsoid.scale(1.0, center=(0, 0, 0))  # 先恢复为单位球
    ellipsoid.scale(scales[0], center=(1, 0, 0))  # 沿 X 轴缩放
    ellipsoid.scale(scales[1], center=(0, 1, 0))  # 沿 Y 轴缩放
    ellipsoid.scale(scales[2], center=(0, 0, 1))  # 沿 Z 轴缩放

    # 应用旋转矩阵
    rot_matrix = R.from_quat(quaternion).as_matrix()
    ellipsoid.rotate(rot_matrix, center=(0, 0, 0))
    
    # 移动到目标中心
    ellipsoid.translate(center)
    
    return ellipsoid

def parse_ply_custom_attributes(ply_file):
    """
    使用 plyfile 读取 PLY 文件，并提取自定义属性
    """
    # 使用 PlyData 读取 PLY 文件
    plydata = PlyData.read(ply_file)

    # 读取顶点部分
    vertex_data = plydata['vertex'].data

    # 提取顶点的坐标 (x, y, z)
    points = np.vstack([vertex_data['x'], vertex_data['y'], vertex_data['z']]).T

    # 提取缩放 (scale_0, scale_1, scale_2)
    scales = np.vstack([vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2']]).T

    # 提取旋转的四元数 (rot_0, rot_1, rot_2, rot_3)
    quaternions = np.vstack([vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3']]).T

    return points, scales, quaternions

def load_ply_and_create_ellipsoids(ply_file):
    points, scales, quaternions = parse_ply_custom_attributes(ply_file)
    # 创建 Open3D 可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_options = vis.get_render_option()
    render_options.background_color = np.array([0.1, 0.1, 0.1])  # 设置为灰色背景（可以根据需要更改）
    # 对每个点生成一个椭球
    for i in tqdm(range(len(points)), desc="Loading point cloud"):
        center = points[i]
        scale = scales[i]
        quaternion = quaternions[i]

        # 创建椭球
        ellipsoid = create_ellipsoid(center, scale, quaternion)

        # 将椭球添加到可视化器
        vis.add_geometry(ellipsoid)
    # 启动可视化器
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    # 调用函数，加载 PLY 文件并生成椭球
    load_ply_and_create_ellipsoids(mesh_path)