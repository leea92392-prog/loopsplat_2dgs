#!/usr/bin/env python3
"""
Launch the Gaussian Viewer for LoopSplat output.

Usage:
    python run_viewer.py <output_dir>              # Local mode
    python run_viewer.py server <output_dir>       # Server mode (for remote viewing)
    python run_viewer.py client                    # Client mode (connect to server)

The output_dir should contain:
    - config.yaml
    - estimated_c2w.ckpt
    - submaps/*.ckpt  (or <scene_name>_global_splats.ply from evaluation)
"""
import sys
import runpy
from pathlib import Path

# Add project root and graphdecoviewer to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
GRAPHDECO_SRC = PROJECT_ROOT / "thirdparty" / "graphdecoviewer" / "src"
if GRAPHDECO_SRC.exists():
    sys.path.insert(0, str(GRAPHDECO_SRC))
if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        print("\nExample:")
        print("  python run_viewer.py output/TUM_RGBD/rgbd_dataset_freiburg1_desk")
        sys.exit(1)

    # Map: "python run_viewer.py <output_dir>" -> "local <output_dir>"
    if argv[0] not in ("local", "server", "client"):
        argv = ["local", argv[0]] + argv[1:]

    sys.argv = [sys.argv[0]] + argv

    gaussianviewer_path = PROJECT_ROOT / "gaussianviewer.py"
    runpy.run_path(str(gaussianviewer_path), run_name="__main__")
