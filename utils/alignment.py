import copy
import numpy as np
import open3d as o3d
from scipy.optimize import minimize_scalar
from hamer.utils.renderer import cam_crop_to_full

from typing import TypedDict, Any
import torch
import copy


class FocalLengthOptArgs(TypedDict):
    """
    cam_bbox: bounding box of the hand in the image
    box_center: center of the bounding box
    box_size: size of the bounding box
    img_size: size of the image
    focal_length: focal length of the camera
    ret_verts: vertices of the hand
    is_right: whether the hand is right
    """

    cam_bbox: Any
    box_center: Any
    box_size: Any
    img_size: Any
    focal_length: float
    ret_verts: torch.Tensor
    is_right: torch.Tensor


def transform_verts_focal_length(
    focal_length_opt_args: FocalLengthOptArgs,
):
    focal_length_opt_args = copy.deepcopy(focal_length_opt_args)
    ret_verts = focal_length_opt_args.pop("ret_verts")
    is_right = focal_length_opt_args.pop("is_right")
    pred_cam_t = cam_crop_to_full(**focal_length_opt_args).detach().cpu().numpy()
    verts_t = ret_verts.detach().cpu().numpy() + pred_cam_t[:, None, :]
    mano_verts = np.concatenate(
        verts_t if is_right[0] else verts_t[::-1], axis=0
    )  # make sure right hand comes first
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mano_verts)
    return pcd


def compute_alignment_loss(
    focal_length,
    ego_focal_length_opt_args: FocalLengthOptArgs,
    exo_focal_length_opt_args: FocalLengthOptArgs,
    weights=None,
    return_details=False,
):
    ego_focal_length_opt_args["focal_length"] = focal_length
    exo_focal_length_opt_args["focal_length"] = focal_length
    ego_pcd = transform_verts_focal_length(ego_focal_length_opt_args)
    exo_pcd = transform_verts_focal_length(exo_focal_length_opt_args)
    if weights is not None:
        transformation, loss, _ = weighted_umeyama_alignment(ego_pcd, exo_pcd, weights)
    else:
        transformation, loss, _ = umeyama_alignment(ego_pcd, exo_pcd)
    if return_details:
        return loss, transformation, ego_pcd, exo_pcd
    return loss


def optimize_focal_length(
    ego_focal_length_opt_args: FocalLengthOptArgs,
    exo_focal_length_opt_args: FocalLengthOptArgs,
    weights=None,
):
    result = minimize_scalar(
        compute_alignment_loss,
        bounds=(1, 5000),
        args=(ego_focal_length_opt_args, exo_focal_length_opt_args, weights),
        method="bounded",
    )

    loss, transformation, ego_pcd, exo_pcd = compute_alignment_loss(
        result.x,
        ego_focal_length_opt_args,
        exo_focal_length_opt_args,
        weights=weights,
        return_details=True,
    )

    if result.success:
        return result.x, loss, transformation, ego_pcd, exo_pcd
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def weighted_umeyama_alignment(source_pcd, target_pcd, weights):
    """
    Weighted Umeyama algorithm to find optimal transformation between two point clouds,
    where each point pair can have different importance in the alignment.

    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud
        target_pcd (o3d.geometry.PointCloud): Target point cloud
        weights (np.ndarray): Weights for each point (shape: [N,]), where N is number of points.
                            Larger weights mean those points have more influence on the alignment.

    Returns:
        tuple: (transformation, loss, transformed_source_pcd)
            - transformation (np.ndarray): 4×4 transformation matrix
            - loss (float): Weighted mean squared error between the aligned source and target points
            - transformed_source_pcd (o3d.geometry.PointCloud): The transformed source point cloud
    """
    # Make copies to avoid modifying the originals
    source_pcd = copy.deepcopy(source_pcd)
    target_pcd = copy.deepcopy(target_pcd)

    # Get points as numpy arrays
    source = np.asarray(source_pcd.points)
    target = np.asarray(target_pcd.points)

    # Normalize weights to sum to 1
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)

    # Get number of points and dimensions
    n, d = source.shape

    # Compute weighted centroids
    source_centroid = np.average(source, axis=0, weights=weights)
    target_centroid = np.average(target, axis=0, weights=weights)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute weighted covariance matrix
    # Multiply each point by sqrt(weight) to incorporate weights into the covariance
    weighted_source_centered = source_centered * np.sqrt(weights)[:, np.newaxis]
    weighted_target_centered = target_centered * np.sqrt(weights)[:, np.newaxis]
    cov = weighted_target_centered.T @ weighted_source_centered

    # SVD of covariance matrix
    u, s_vals, vh = np.linalg.svd(cov)

    # Handle reflection case
    reflection_matrix = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection_matrix[d - 1, d - 1] = -1

    # Calculate rotation
    R = u @ reflection_matrix @ vh

    # Calculate weighted scaling
    weighted_var_source = np.sum(weights[:, np.newaxis] * (source_centered**2))
    s = 1.0 if weighted_var_source == 0 else np.sum(s_vals) / weighted_var_source

    # Calculate translation
    t = target_centroid - s * R @ source_centroid

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t

    # Apply transformation to create the transformed source point cloud
    transformed_source_pcd = copy.deepcopy(source_pcd)
    transformed_source_pcd.transform(transformation)

    # Compute weighted alignment error (loss) as weighted mean squared error
    transformed_points = np.asarray(transformed_source_pcd.points)
    squared_errors = np.sum((target - transformed_points) ** 2, axis=1)
    loss = np.average(squared_errors, weights=weights)

    return transformation, loss, transformed_source_pcd


def umeyama_alignment(source_pcd, target_pcd):
    """
    Umeyama algorithm to find optimal transformation between two point clouds.

    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud
        target_pcd (o3d.geometry.PointCloud): Target point cloud

    Returns:
        tuple: (transformation, loss, transformed_source_pcd)
            - transformation (np.ndarray): 4×4 transformation matrix
            - loss (float): Mean squared error between the aligned source and target points
            - transformed_source_pcd (o3d.geometry.PointCloud): The transformed source point cloud
    """
    # Make copies to avoid modifying the originals
    source_pcd = copy.deepcopy(source_pcd)
    target_pcd = copy.deepcopy(target_pcd)

    # Get points as numpy arrays
    source = np.asarray(source_pcd.points)
    target = np.asarray(target_pcd.points)

    # Get number of points and dimensions
    n, d = source.shape

    # Center the points
    source_centroid = np.mean(source, axis=0)
    target_centroid = np.mean(target, axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute covariance matrix
    cov = target_centered.T @ source_centered / n

    # SVD of covariance matrix
    u, s_vals, vh = np.linalg.svd(cov)

    # Handle reflection case
    reflection_matrix = np.eye(d)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        reflection_matrix[d - 1, d - 1] = -1

    # Calculate rotation
    R = u @ reflection_matrix @ vh

    # Calculate scaling
    var_source = np.sum(np.var(source_centered, axis=0))
    s = 1.0 if var_source == 0 else np.sum(s_vals) / var_source

    # Calculate translation
    t = target_centroid - s * R @ source_centroid

    # Create transformation matrix
    transformation = np.eye(4)
    transformation[:3, :3] = s * R
    transformation[:3, 3] = t

    # Apply transformation to create the transformed source point cloud
    transformed_source_pcd = copy.deepcopy(source_pcd)
    transformed_source_pcd.transform(transformation)

    # Compute alignment error (loss) as mean squared error
    transformed_points = np.asarray(transformed_source_pcd.points)
    loss = np.mean(np.sum((target - transformed_points) ** 2, axis=1))

    return transformation, loss, transformed_source_pcd


def register_point_clouds(
    source_pcd, target_pcd, voxel_size=0.005, default_color=[0, 0.651, 0.929]
):
    """
    Registers a source point cloud to a target point cloud using Mean Alignment + RANSAC + ICP.

    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud to be aligned
        target_pcd (o3d.geometry.PointCloud): Target point cloud to align to
        voxel_size (float): Voxel size for downsampling and normal/feature estimation
        default_color (list): RGB color (0-1 range) to paint the source cloud if it lacks colors

    Returns:
        tuple: (transformed_source_pcd, final_transformation)
            - transformed_source_pcd (o3d.geometry.PointCloud): The source point cloud after alignment
            - final_transformation (np.ndarray): The 4x4 transformation matrix combining all steps
    """
    print(f"\nStarting registration with voxel size: {voxel_size}")

    # Make copies to avoid modifying the originals
    source_pcd = copy.deepcopy(source_pcd)
    target_pcd = copy.deepcopy(target_pcd)

    # --- Initial Mean Alignment ---
    print("Performing initial mean alignment...")
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)

    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    translation_vector = target_mean - source_mean

    # Create a 4x4 transformation matrix for the mean alignment
    mean_alignment_transform = np.eye(4)
    mean_alignment_transform[:3, 3] = translation_vector

    # Apply translation directly to the source point cloud
    source_pcd.translate(translation_vector)
    print(f"Applied translation vector: {translation_vector}")

    # Add uniform color if the point clouds don't have colors
    if not source_pcd.has_colors():
        source_pcd.paint_uniform_color(default_color)
    if not target_pcd.has_colors():
        target_pcd.paint_uniform_color(default_color)

    # --- Global Registration (RANSAC) ---
    print("Downsampling point clouds...")
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print("Estimating normals for RANSAC...")
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    print("Normals estimated for RANSAC.")

    radius_feature = voxel_size * 5
    print("Computing FPFH features...")
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    print("FPFH features computed.")

    distance_threshold_global = voxel_size * 1.5
    print("Running RANSAC...")
    result_ransac = (
        o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold_global,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold_global
                ),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
    )
    print("RANSAC finished.")
    print("Global registration (RANSAC) fitness:", result_ransac.fitness)
    print("Global registration (RANSAC) inlier_rmse:", result_ransac.inlier_rmse)

    if result_ransac.fitness < 0.1:  # Check if RANSAC failed significantly
        print("Warning: RANSAC fitness is low. ICP might fail or be inaccurate.")

    # --- Fine-tuning with ICP ---
    print("Estimating normals for ICP...")
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normal, max_nn=30
        )
    )
    print("Normals estimated for ICP.")

    distance_threshold_icp = voxel_size * 0.4
    print("Running ICP...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        distance_threshold_icp,
        result_ransac.transformation,  # Use RANSAC result as initial guess
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    print("ICP finished.")
    print("ICP Fitness:", result_icp.fitness)
    print("ICP Inlier RMSE:", result_icp.inlier_rmse)

    # ICP transformation (which already includes RANSAC as initial guess)
    icp_transformation = result_icp.transformation

    # Apply the ICP transformation to the source point cloud
    source_pcd.transform(icp_transformation)

    # Combine transformations: first mean alignment, then ICP
    final_transformation = np.matmul(icp_transformation, mean_alignment_transform)

    print("Registration complete.")
    return source_pcd, final_transformation
