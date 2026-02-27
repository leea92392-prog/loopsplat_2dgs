import argparse
from pathlib import Path
from src.evaluation.evaluator import Evaluator


def get_args():
    parser = argparse.ArgumentParser(description='Arguments to compute the mesh')
    parser.add_argument('--checkpoint_path', type=str, help='SLAM checkpoint path', default="output/slam/full_experiment/")
    parser.add_argument('--config_path', type=str, help='Config path', default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    #args.checkpoint_path = "/media/lee/Data/2DGS_SLAM_output/2DGS_SLAM_outout/2dgs-slam/TUM_RGBD/rgbd_dataset_freiburg1_desk_final"
    #args.checkpoint_path = "/home/lee/A-LOAM-Container/LoopSplat_2dgs/output/TUM_RGBD/rgbd_dataset_freiburg1_desk_ablation_1_loop"
    #args.checkpoint_path = "/home/lee/A-LOAM-Container/LoopSplat_2dgs/output/TUM_RGBD/rgbd_dataset_freiburg3_long_office_household_final_loop_record2"
    #args.checkpoint_path = "/home/lee/A-LOAM-Container/LoopSplat_2dgs/output/TUM_RGBD/rgbd_dataset_freiburg1_desk_depthlab"
    #args.checkpoint_path = "/home/wujie/ws_0122/loopsplat_2dgs/output/zed720/short_desk_trackingtest"
    
    #args.checkpoint_path = "/home/lee/A-LOAM-Container/LoopSplat_2dgs/output/UTMM/ego-centric-1_loop"
    # args.checkpoint_path = "/home/wujie/ws_0122/loopsplat_2dgs/output/TUM_RGBD/rgbd_dataset_freiburg1_desk_mini_ba_with_render-pose_opt"
    args.checkpoint_path = "/home/wujie/ws_0122/loopsplat_2dgs/output/zed720/work_desk_trackingtest3"
    
    if args.config_path == "":
        args.config_path = Path(args.checkpoint_path) / "config.yaml"
    evaluator = Evaluator(Path(args.checkpoint_path), Path(args.config_path))
    evaluator.run()
