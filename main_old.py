# Standard library imports
import argparse
import os
from pathlib import Path
import copy  # Add import for deepcopy

# Third-party imports
import cv2
import numpy as np
import open3d as o3d
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data.dataloader import default_collate

# Local imports
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel
from utils.manipulation import *

# Constants
LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_FOCAL_LENGTH_BOUNDS = (1, 5000)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_OUTPUT_FOLDER = "out_demo"

"""
python process_images.py --ego_image /local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png --exo_image /local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png --out_folder demo_out
"""


def vit_pose_detection(
    img, detector, cpm, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    """
    Detect human hand keypoints in the image.

    Args:
        img (np.ndarray): Input image in BGR format
        detector: Object detector model
        cpm: Keypoint detector model
        confidence_threshold (float): Confidence threshold for keypoint detection

    Returns:
        tuple: (bboxes, is_right_hand_flags)
    """
    # Detect humans in image
    det_out = detector(img)
    img_rgb = img.copy()[:, :, ::-1]  # Convert BGR to RGB

    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (
        det_instances.scores > confidence_threshold
    )
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        img_rgb,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    bboxes = []
    is_right = []

    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # Process left hand keypoints
        keyp = left_hand_keyp
        valid = keyp[:, 2] > confidence_threshold
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(0)

        # Process right hand keypoints
        keyp = right_hand_keyp
        valid = keyp[:, 2] > confidence_threshold
        if sum(valid) > 3:
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(1)

    if len(bboxes) == 0:
        raise ValueError("No hands detected")

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    return boxes, right


def process_image(
    img_cv2, model, model_cfg, device, detector, cpm, renderer, out_folder, out_name
):
    """
    Process an image to detect and render hands.

    Args:
        img_cv2 (np.ndarray): Input image in BGR format
        model: HaMeR model
        model_cfg: Model configuration
        device: Torch device
        detector: Object detector model
        cpm: Keypoint detector model
        renderer: Renderer object
        out_folder (str): Output folder path
        out_name (str): Output file name prefix

    Returns:
        tuple: Hand vertices, camera parameters, and other detection data
    """
    boxes, right = vit_pose_detection(img_cv2, detector, cpm)
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=2.0)

    if len(dataset) != 2:
        raise ValueError(f"Dataset length must be 2. Got {len(dataset)}")

    batch = default_collate([dataset[0], dataset[1]])
    batch = recursive_to(batch, device)

    with torch.no_grad():
        out = model(batch)

    # Process model output
    multiplier = 2 * batch["right"] - 1
    pred_cam = out["pred_cam"]
    pred_cam[:, 1] = multiplier * pred_cam[:, 1]
    box_center = batch["box_center"].float()
    box_size = batch["box_size"].float()
    img_size = batch["img_size"].float()

    ret_verts = out["pred_vertices"]
    ret_verts[:, :, 0] = multiplier.reshape(-1, 1) * ret_verts[:, :, 0]

    scaled_focal_length = (
        model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
    )

    pred_cam_t_full = (
        cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length)
        .detach()
        .cpu()
        .numpy()
    )

    is_right = batch["right"]

    all_verts = [ret_verts[n].detach().cpu().numpy() for n in range(2)]
    all_cam_t = [pred_cam_t_full[n] for n in range(2)]
    all_right = [is_right[n].cpu().numpy() for n in range(2)]

    # Render front view
    render_args = dict(
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        focal_length=scaled_focal_length,
        name=os.path.join(out_folder, f"{out_name}_all.obj"),
    )
    cam_view = renderer.render_rgba_multiple(
        all_verts,
        cam_t=all_cam_t,
        render_res=img_size[0],
        is_right=all_right,
        **render_args,
    )

    # Overlay image
    input_img = img_cv2.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate(
        [input_img, np.ones_like(input_img[:, :, :1])], axis=2
    )  # Add alpha channel
    input_img_overlay = (
        input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
        + cam_view[:, :, :3] * cam_view[:, :, 3:]
    )

    # Save output images
    cv2.imwrite(
        os.path.join(out_folder, f"{out_name}_all.jpg"),
        255 * input_img_overlay[:, :, ::-1],
    )

    mask = renderer.render_mask_multiple(
        all_verts,
        cam_t=all_cam_t,
        render_res=img_size[0],
        is_right=all_right,
        focal_length=scaled_focal_length,
    )
    cv2.imwrite(os.path.join(out_folder, f"{out_name}_all_mask.png"), 255 * mask)

    return ret_verts, pred_cam, box_center, box_size, img_size, is_right, mask


def pcd2dense(pcd, faces, vert_count, mesh_base_color=(1.0, 1.0, 0.9)):
    faces_left = faces[:, [0, 2, 1]]
    faces_right = faces
    vertex_colors = np.array([mesh_base_color] * len(pcd.points))
    both_hands_faces = np.concatenate(
        [faces_right, faces_left + len(pcd.points) // 2], axis=0
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(pcd.points)
    mesh.triangles = o3d.utility.Vector3iVector(both_hands_faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    pcd = mesh.sample_points_uniformly(number_of_points=vert_count)
    return pcd


def compute_alignment_loss(
    focal_length,
    ego_pred_cam,
    ego_box_center,
    ego_box_size,
    ego_img_size,
    ego_ret_verts,
    ego_is_right,
    exo_pred_cam,
    exo_box_center,
    exo_box_size,
    exo_img_size,
    exo_ret_verts,
    exo_is_right,
    return_details=False,
):
    ego_pcd = transform_verts_focal_length(
        ego_pred_cam,
        ego_box_center,
        ego_box_size,
        ego_img_size,
        ego_ret_verts,
        ego_is_right,
        focal_length,
    )
    exo_pcd = transform_verts_focal_length(
        exo_pred_cam,
        exo_box_center,
        exo_box_size,
        exo_img_size,
        exo_ret_verts,
        exo_is_right,
        focal_length,
    )
    transformation, loss, _ = umeyama_alignment(ego_pcd, exo_pcd)
    if return_details:
        return loss, transformation, ego_pcd, exo_pcd
    return loss


def transform_verts_focal_length(
    pred_cam, box_center, box_size, img_size, ret_verts, is_right, focal_length
):
    pred_cam_t = (
        cam_crop_to_full(pred_cam, box_center, box_size, img_size, focal_length)
        .detach()
        .cpu()
        .numpy()
    )
    verts_t = ret_verts.detach().cpu().numpy() + pred_cam_t[:, None, :]
    mano_verts = np.concatenate(
        verts_t if is_right[0] else verts_t[::-1], axis=0
    )  # make sure right hand comes first
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mano_verts)
    return pcd


def optimize_focal_length(ego_params, exo_params):
    loss_args = ego_params + exo_params

    result = minimize_scalar(
        compute_alignment_loss,
        bounds=DEFAULT_FOCAL_LENGTH_BOUNDS,
        args=loss_args,
        method="bounded",
    )

    loss, transformation, ego_pcd, exo_pcd = compute_alignment_loss(
        result.x, *loss_args, return_details=True
    )

    if result.success:
        return result.x, loss, transformation, ego_pcd, exo_pcd
    else:
        raise ValueError(f"Optimization failed: {result.message}")


def main():
    """Main function to process images and align hand models."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="HaMeR demo code")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--ego_image", type=str, required=True, help="Path to ego image"
    )
    parser.add_argument(
        "--exo_image", type=str, required=True, help="Path to exo image"
    )
    parser.add_argument(
        "--out_folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help="Output folder to save rendered results",
    )
    parser.add_argument(
        "--body_detector",
        type=str,
        default="regnety",
        choices=["vitdet", "regnety"],
        help="Using regnety improves runtime and reduces memory",
    )
    args = parser.parse_args()

    # Initialize
    model, model_cfg, device, detector, cpm, renderer = initialize(args)
    ego_img = cv2.imread(args.ego_image)
    exo_img = cv2.imread(args.exo_image)
    (
        ego_ret_verts,
        ego_pred_cam,
        ego_box_center,
        ego_box_size,
        ego_img_size,
        ego_is_right,
        ego_mask,
    ) = process_image(
        ego_img,
        model,
        model_cfg,
        device,
        detector,
        cpm,
        renderer,
        args.out_folder,
        "ego",
    )
    (
        exo_ret_verts,
        exo_pred_cam,
        exo_box_center,
        exo_box_size,
        exo_img_size,
        exo_is_right,
        exo_mask,
    ) = process_image(
        exo_img,
        model,
        model_cfg,
        device,
        detector,
        cpm,
        renderer,
        args.out_folder,
        "exo",
    )

    # Focal length optimization
    ego_params = (
        ego_pred_cam,
        ego_box_center,
        ego_box_size,
        ego_img_size,
        ego_ret_verts,
        ego_is_right,
    )
    exo_params = (
        exo_pred_cam,
        exo_box_center,
        exo_box_size,
        exo_img_size,
        exo_ret_verts,
        exo_is_right,
    )
    optimal_focal_length, min_loss, alignment_transform, ego_mano_pcd, exo_mano_pcd = (
        optimize_focal_length(ego_params, exo_params)
    )
    print(f"Optimal Focal length: {optimal_focal_length:.2f}")
    print(f"Final Alignment Loss: {min_loss}")

    ## test
    o3d.io.write_point_cloud(
        f"{args.out_folder}/focal_length_test.ply",
        copy.deepcopy(ego_mano_pcd).transform(alignment_transform)
        + copy.deepcopy(exo_mano_pcd),
    )

    # Create Point Clouds
    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(args.ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(args.exo_image)
    ego_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext, mask=ego_mask)
    exo_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext, mask=exo_mask)
    ego_mano_pcd = pcd2dense(ego_mano_pcd, model.mano.faces, len(ego_pcd.points))
    exo_mano_pcd = pcd2dense(exo_mano_pcd, model.mano.faces, len(exo_pcd.points))

    # Register ego point cloud)
    aligned_ego_mano_pcd, ego_transformation = register_point_clouds(
        source_pcd=ego_mano_pcd, target_pcd=ego_pcd
    )
    ## test
    o3d.io.write_point_cloud(
        f"{args.out_folder}/ego_registration_test.ply",
        copy.deepcopy(ego_mano_pcd).transform(ego_transformation)
        + copy.deepcopy(ego_pcd),
    )

    # Register exo point cloud
    aligned_exo_mano_pcd, exo_transformation = register_point_clouds(
        source_pcd=exo_mano_pcd, target_pcd=exo_pcd
    )
    ## test
    o3d.io.write_point_cloud(
        f"{args.out_folder}/exo_registration_test.ply",
        copy.deepcopy(exo_mano_pcd).transform(exo_transformation)
        + copy.deepcopy(exo_pcd),
    )

    # Compute transformation chain
    ego_to_exo_transform = np.matmul(
        exo_transformation,
        np.matmul(alignment_transform, np.linalg.inv(ego_transformation)),
    )
    exo_to_ego_transform = np.linalg.inv(ego_to_exo_transform)

    # Apply transformation chain to ego point cloud
    ego_total_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext)
    exo_total_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext)

    # Apply transformation chain to ego point cloud
    transformed_ego_total_pcd = copy.deepcopy(ego_total_pcd)
    transformed_ego_total_pcd.transform(ego_to_exo_transform)
    combined_total_pcd = transformed_ego_total_pcd + exo_total_pcd
    o3d.io.write_point_cloud(f"{args.out_folder}/combined_pcd.ply", combined_total_pcd)
    print(f"Saved combined point cloud to {args.out_folder}/combined_pcd.ply")


if __name__ == "__main__":
    main()
