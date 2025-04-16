# Standard library imports
import os
import pickle
from glob import glob
import copy  # Add import for deepcopy

# Third-party imports
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
import open3d as o3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path


def pcd2dense(
    pcd: o3d.geometry.PointCloud,
    faces: np.ndarray,
    vert_count: int,
    mesh_base_color=(1.0, 1.0, 0.9),
    viewable_indices=None,
) -> o3d.geometry.PointCloud:
    faces_left = faces[:, [0, 2, 1]]
    faces_right = faces
    vertex_colors = np.array([mesh_base_color] * len(pcd.points))
    both_hands_faces = np.concatenate(
        [faces_right, faces_left + len(pcd.points) // 2], axis=0
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pcd.points)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    if viewable_indices is not None:
        # Filter faces to only include those where all vertices are viewable
        viewable_indices_set = set(viewable_indices)
        filtered_faces = []
        for face in both_hands_faces:
            if all(vertex in viewable_indices_set for vertex in face):
                filtered_faces.append(face)
        mesh.triangles = o3d.utility.Vector3iVector(np.array(filtered_faces))
    else:
        mesh.triangles = o3d.utility.Vector3iVector(both_hands_faces)

    # save mesh
    o3d.io.write_triangle_mesh(
        f"viewable_mesh.ply",
        mesh,
        write_triangle_uvs=True,
    )

    res_pcd: o3d.geometry.PointCloud = mesh.sample_points_uniformly(
        number_of_points=vert_count
    )
    return res_pcd


def cam2pixel(cam_coord, f, c):
    """
    Convert 3D camera coordinates to pixel coordinates.

    Args:
        cam_coord (np.ndarray): Camera coordinates (N, 3)
        f (list): Focal lengths [fx, fy]
        c (list): Principal points [cx, cy]

    Returns:
        np.ndarray: Pixel coordinates (N, 3) with z preserved
    """
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def cam2world(cam_coord, R, t):
    """
    Convert camera coordinates to world coordinates.

    Args:
        cam_coord (np.ndarray): Camera coordinates (N, 3)
        R (np.ndarray): Rotation matrix (3, 3)
        t (np.ndarray): Translation vector (3,)

    Returns:
        np.ndarray: World coordinates (N, 3)
    """
    world_coord = np.dot(
        np.linalg.inv(R), (cam_coord - t.reshape(1, 3)).transpose(1, 0)
    ).transpose(1, 0)
    return world_coord


def world2cam(world_coord, R, t):
    """
    Convert world coordinates to camera coordinates.

    Args:
        world_coord (np.ndarray): World coordinates (N, 3)
        R (np.ndarray): Rotation matrix (3, 3)
        t (np.ndarray): Translation vector (3,)

    Returns:
        np.ndarray: Camera coordinates (N, 3)
    """
    cam_coord = np.dot(R, world_coord.transpose(1, 0)).transpose(1, 0) + t.reshape(1, 3)
    return cam_coord


def hand2hand(points, R, t):
    """
    Transform hand coordinates from one frame to another.

    Args:
        points (np.ndarray): Hand coordinates (N, 3)
        R (np.ndarray): Rotation matrix (3, 3)
        t (np.ndarray): Translation vector (3, 1)

    Returns:
        np.ndarray: Transformed hand coordinates (N, 3)
    """
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    RT = np.hstack([R, t])
    camera_points_h = (RT @ points_h.T).T
    camera_points = camera_points_h[:, :3]
    return camera_points


def rotation2rotation(A, B):
    """
    Find the rotation X such that XA = B.

    Args:
        A (np.ndarray): Source rotation matrix (3, 3)
        B (np.ndarray): Target rotation matrix (3, 3)

    Returns:
        np.ndarray: Rotation matrix X
    """
    try:
        A_inv = A.T
        X = np.dot(B, A_inv)  # X = B * A-1 ~~ XA = B
        return X
    except Exception as e:
        print(f"Error: {e}")
        return None


def points2image(points, colors, RT, K, image_width, image_height):
    """
    Project 3D points onto a 2D image.

    Args:
        points (np.ndarray): 3D points (N, 3)
        colors (np.ndarray): Colors for each point (N, 3)
        RT (np.ndarray): Camera extrinsic matrix (3, 4) or (4, 4)
        K (np.ndarray): Camera intrinsic matrix (3, 3)
        image_width (int): Width of the output image
        image_height (int): Height of the output image

    Returns:
        np.ndarray: RGB image
    """
    # Convert to homogeneous coordinates
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    camera_points_h = (RT @ points_h.T).T
    camera_points = camera_points_h[:, :3]

    # Project to image plane
    projected_points = (K @ camera_points.T).T
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    image_width, image_height = int(image_width), int(image_height)
    rgb_map = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Sort points by depth for proper occlusion
    depths = camera_points[:, 2]
    indices = np.argsort(depths)[::-1]

    # Draw points on the image
    for i in indices:
        point = projected_points[i]
        x = int(round(point[0]))
        y = int(round(point[1]))
        z = depths[i]

        if 0 <= x < image_width and 0 <= y < image_height and z > 0:
            color = colors[i].astype(np.uint8).tolist()
            cv2.circle(rgb_map, (x, y), 0, color, -1)

    return rgb_map


def depth2points(depth, fx, fy, cx, cy):
    """
    Convert a depth map to 3D points.

    Args:
        depth (np.ndarray): Depth map (H, W)
        fx, fy (float): Focal lengths
        cx, cy (float): Principal point

    Returns:
        np.ndarray: 3D points (H, W, 3)
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.dstack((x, y, z))


def exo2ego(
    exo_rgb_path,
    exo_depth_path,
    exo_cam_int_path,
    exo_cam_ext_path,
    exo_hand_path,
    ego_cam_int_path,
    ego_cam_ext_path,
    ego_hand_path,
):
    """
    Transform exo-view data to ego-view.

    Args:
        exo_rgb_path (str): Path to exo RGB image
        exo_depth_path (str): Path to exo depth map
        exo_cam_int_path (str): Path to exo camera intrinsics
        exo_cam_ext_path (str): Path to exo camera extrinsics
        exo_hand_path (str): Path to exo hand pose
        ego_cam_int_path (str): Path to ego camera intrinsics
        ego_cam_ext_path (str): Path to ego camera extrinsics
        ego_hand_path (str): Path to ego hand pose

    Returns:
        np.ndarray: Ego RGB prediction
    """
    # Load exo data
    exo_cam_int = np.loadtxt(exo_cam_int_path)
    exo_fx, exo_fy, exo_cx, exo_cy = exo_cam_int[:4]
    exo_cam_ext = np.loadtxt(exo_cam_ext_path).reshape(4, 4)
    exo_depth = cv2.imread(exo_depth_path, cv2.IMREAD_ANYDEPTH)

    # Convert depth to points
    points = depth2points(exo_depth, exo_fx, exo_fy, exo_cx, exo_cy).reshape(-1, 3)
    colors = cv2.imread(exo_rgb_path).reshape(-1, 3)

    # Load ego camera parameters
    ego_cam_ext = np.loadtxt(ego_cam_ext_path).reshape(4, 4)
    ego_cam_ext_inv = np.linalg.inv(ego_cam_ext)

    # Transform points from exo to ego frame
    points /= 1000  # Convert mm to meters
    points_one = np.ones_like(points[:, 0:1])
    points_h = np.hstack([points, points_one])
    translated_points = ego_cam_ext_inv @ exo_cam_ext @ points_h.T
    translated_points = translated_points.T[:, :3]

    # Project points to ego image
    X = np.eye(4)[:3, :]  # Identity transformation in camera frame
    ego_cam_int = np.loadtxt(ego_cam_int_path)
    ego_fx, ego_fy, ego_cx, ego_cy, ego_w, ego_h = ego_cam_int
    ego_K = np.array([[ego_fx, 0, ego_cx], [0, ego_fy, ego_cy], [0, 0, 1]])
    ego_rgb_pred = points2image(translated_points, colors, X, ego_K, ego_w, ego_h)

    return ego_rgb_pred


def exo2ego_force(
    exo_rgb_path,
    exo_depth_path,
    exo_cam_int_path,
    exo_cam_ext_path,
    exo_hand_path,
    ego_cam_int_path,
    ego_cam_ext_path,
    ego_hand_path,
    transformation_chain,
):

    # get pointcloud from exos
    exo_cam_int = np.loadtxt(exo_cam_int_path)
    exo_fx, exo_fy, exo_cx, exo_cy = exo_cam_int[:4]
    exo_cam_ext = np.loadtxt(exo_cam_ext_path).reshape(4, 4)
    exo_depth = cv2.imread(exo_depth_path, cv2.IMREAD_ANYDEPTH)
    points = depth2points(exo_depth, exo_fx, exo_fy, exo_cx, exo_cy).reshape(-1, 3)
    colors = cv2.imread(exo_rgb_path).reshape(-1, 3)

    # project ego with X
    ego_cam_ext = np.loadtxt(ego_cam_ext_path).reshape(4, 4)
    exo_R = exo_cam_ext[:3, :3]
    ego_R = ego_cam_ext[:3, :3]
    exo_t = exo_cam_ext[:3, 3:]
    ego_t = ego_cam_ext[:3, 3:]
    points /= 1000
    ego_cam_ext_inv = np.linalg.inv(ego_cam_ext)
    points_one = np.ones_like(points[:, 0])
    points_one = np.expand_dims(points_one, axis=1)
    points = np.hstack([points, points_one])

    # translated_points = ego_cam_ext_inv @ exo_cam_ext @ points.T
    translated_points = transformation_chain @ points.T

    translated_points = translated_points.T[:, :3]
    X = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    ego_cam_int = np.loadtxt(ego_cam_int_path)
    ego_fx, ego_fy, ego_cx, ego_cy, ego_w, ego_h = ego_cam_int
    ego_K = np.array([[ego_fx, 0, ego_cx], [0, ego_fy, ego_cy], [0, 0, 1]])
    ego_rgb_pred = points2image(translated_points, colors, X, ego_K, ego_w, ego_h)

    return ego_rgb_pred


def depth2pcd(depth, rgb, cam_int, cam_ext, mask=None):
    """
    Convert depth map and RGB image to a colored point cloud.

    Args:
        depth (np.ndarray): Depth map (H, W)
        rgb (np.ndarray): RGB image (H, W, 3)
        cam_int (np.ndarray): Camera intrinsic parameters
        cam_ext (np.ndarray): Camera extrinsic parameters
        mask (np.ndarray, optional): Boolean mask for points to exclude

    Returns:
        o3d.geometry.PointCloud: Colored point cloud
    """
    fx, fy, cx, cy = cam_int[:4]
    points = depth2points(depth, fx, fy, cx, cy).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    # Apply mask if provided
    if mask is not None:
        mask = mask.astype(bool).reshape(-1)
        points = points[~mask]
        colors = colors[~mask]

    # Convert RGB values to range [0, 1]
    colors = colors / 255.0

    # Convert depth from mm to meters
    points /= 1000

    # Create and return point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
