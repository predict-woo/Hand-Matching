import cv2
import numpy as np
from pathlib import Path
from glob import glob
import os


def load_rgb_depth_cam_ext_int(root_path, subject, cam_id, index=None):
    """
    Load RGB, depth, hand poses, camera extrinsics, and intrinsics.

    Args:
        root_path (str): Root directory path
        subject (str): Subject identifier
        cam_id (str): Camera identifier
        index (int, optional): Specific frame index to load

    Returns:
        tuple: Paths to RGB, depth, hand poses, camera extrinsics and intrinsics
    """
    rgb_paths = sorted(
        glob(os.path.join(root_path, subject, "*/*", cam_id, "rgb/*.png"))
    )
    depth_paths = [path.replace("rgb", "depth") for path in rgb_paths]
    hand_paths = [
        path.replace("rgb", "hand_pose").replace("png", "txt") for path in rgb_paths
    ]
    cam_ext_paths = [
        path.replace("rgb", "cam_pose").replace("png", "txt") for path in rgb_paths
    ]
    cam_int_paths = [path.split("rgb")[0] + "cam_intrinsics.txt" for path in rgb_paths]

    if index is not None:
        return (
            rgb_paths[index],
            depth_paths[index],
            hand_paths[index],
            cam_ext_paths[index],
            cam_int_paths[index],
        )
    return rgb_paths, depth_paths, hand_paths, cam_ext_paths, cam_int_paths


def rgb_path_to_rest(rgb_path_str):
    """
    Derives paths for depth, camera intrinsics, and extrinsics from an RGB image path.

    Assumes a directory structure like: .../camX/{rgb, depth, cam_pose}/...
    and .../camX/cam_intrinsics.txt

    Args:
        rgb_path_str (str): The path to the RGB image file.

    Returns:
        tuple[Path, Path, Path]: Paths to the depth image, camera intrinsics file,
                                 and camera extrinsics file.
    """
    rgb_path = Path(rgb_path_str)
    base_dir = rgb_path.parent.parent  # Go up from 'rgb' dir to 'camX' dir

    depth_path = base_dir / "depth" / rgb_path.name
    cam_int_path = base_dir / "cam_intrinsics.txt"
    cam_ext_path = base_dir / "cam_pose" / rgb_path.with_suffix(".txt").name
    return depth_path, cam_int_path, cam_ext_path


def load_from_rgb_path(rgb_path_str):
    """
    Loads depth, RGB image, camera intrinsics, and camera extrinsics given an RGB image path.

    Args:
        rgb_path_str (str): The path to the RGB image file.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            - depth (np.ndarray): Depth image (H, W).
            - rgb (np.ndarray): RGB image (H, W, 3).
            - cam_int (np.ndarray): Camera intrinsic matrix (3, 3).
            - cam_ext (np.ndarray): Camera extrinsic matrix (4, 4).
    """
    depth_path, cam_int_path, cam_ext_path = rgb_path_to_rest(rgb_path_str)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH)
    rgb = cv2.imread(rgb_path_str)
    cam_int = np.loadtxt(str(cam_int_path))
    cam_ext = np.loadtxt(str(cam_ext_path)).reshape(4, 4)
    return depth, rgb, cam_int, cam_ext
