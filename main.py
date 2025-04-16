import argparse

import cv2
import torch
from torch.utils.data.dataloader import default_collate

from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import Renderer, cam_crop_to_full
from utils.defaults import *
from utils.models import (
    initialize_hamer,
    initialize_detector,
    initialize_vitpose,
    initialize_vggt,
    vit_pose_detection,
    vggt_create_point_cloud,
    print_GPU_memory,
)
from hamer.utils import recursive_to


from utils.alignment import (
    optimize_focal_length,
    FocalLengthOptArgs,
    register_point_clouds,
    compute_alignment_loss,
    register_point_clouds_with_scale_ambiguity,
)
from utils.data import load_from_rgb_path
from utils.manipulation import depth2pcd, pcd2dense
import open3d as o3d
import copy
import numpy as np
from typing import Any
import os
from utils.manipulation import exo2ego_force

USE_WEIGHTED = True
USE_PSUEDO_DEPTH = True


def main(args_in=None):
    if args_in is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--hamer_checkpoint", default=DEFAULT_CHECKPOINT, type=str)
        parser.add_argument("--body_detector", default="regnety", type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--out_folder", default="output", type=str)
        parser.add_argument(
            "--ego_image",
            default="/local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png",
            type=str,
        )
        parser.add_argument(
            "--exo_image",
            default="/local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png",
            type=str,
        )
        args = parser.parse_args()
    else:
        args = args_in
        if not hasattr(args, "hamer_checkpoint"):
            args.hamer_checkpoint = DEFAULT_CHECKPOINT
        if not hasattr(args, "body_detector"):
            args.body_detector = "regnety"
        if not hasattr(args, "device"):
            args.device = "cuda"
        if not hasattr(args, "out_folder"):
            args.out_folder = "output"

    hamer_model, hamer_model_cfg = initialize_hamer(args.hamer_checkpoint, args.device)
    detector = initialize_detector(args.body_detector)
    cpm = initialize_vitpose(args.device)
    renderer = Renderer(hamer_model_cfg, faces=hamer_model.mano.faces)

    ego_image = cv2.imread(args.ego_image)
    exo_image = cv2.imread(args.exo_image)

    images = [ego_image, exo_image]
    params = []
    masks = []
    verts = []
    viewable_indices = []

    for image in images:
        boxes, right = vit_pose_detection(image, detector, cpm)
        dataset = ViTDetDataset(
            hamer_model_cfg, image, boxes, right, rescale_factor=2.0
        )
        if len(dataset) != 2:
            raise ValueError(f"Dataset length must be 2. Got {len(dataset)}")

        batch: Any = recursive_to(
            default_collate([dataset[0], dataset[1]]), args.device
        )
        with torch.no_grad():
            out = hamer_model(batch)

        # Process model output
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        ret_verts = out["pred_vertices"]
        ret_verts[:, :, 0] = multiplier.reshape(-1, 1) * ret_verts[:, :, 0]

        keypoints = out["pred_keypoints_3d"]
        keypoints[:, :, 0] = multiplier.reshape(-1, 1) * keypoints[:, :, 0]

        is_right = batch["right"]

        scaled_focal_length = (
            hamer_model_cfg.EXTRA.FOCAL_LENGTH
            / hamer_model_cfg.MODEL.IMAGE_SIZE
            * img_size.max()
        )

        pred_cam_t_full = (
            cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            .detach()
            .cpu()
            .numpy()
        )

        all_verts = [ret_verts[n].detach().cpu().numpy() for n in range(2)]
        all_cam_t = [pred_cam_t_full[n] for n in range(2)]
        all_right = [is_right[n].cpu().numpy() for n in range(2)]

        mask, viewable_index = renderer.render_mask_multiple(
            all_verts,
            cam_t=all_cam_t,
            render_res=img_size[0],
            is_right=all_right,
            focal_length=scaled_focal_length,
            viewable_threshold=0.008,
        )

        np.save(f"mask.npy", mask)

        image_processed_result: FocalLengthOptArgs = {
            "cam_bbox": pred_cam,
            "box_center": box_center,
            "box_size": box_size,
            "img_size": img_size,
            "focal_length": scaled_focal_length,
            "ret_verts": ret_verts if not USE_WEIGHTED else keypoints,
            "is_right": is_right,
        }

        params.append(image_processed_result)
        masks.append(mask)
        verts.append(ret_verts)
        viewable_indices.append(viewable_index)

    # release all gpu memory
    torch.cuda.empty_cache()
    mano_faces = hamer_model.mano.faces
    del hamer_model, hamer_model_cfg, detector, cpm, renderer
    print_GPU_memory()

    ego_mask = masks[0]
    exo_mask = masks[1]

    ego_verts = verts[0]
    exo_verts = verts[1]

    ego_viewable_indices = viewable_indices[0]
    exo_viewable_indices = viewable_indices[1]

    ego_params: FocalLengthOptArgs = params[0]
    exo_params: FocalLengthOptArgs = params[1]

    weights = None
    if USE_WEIGHTED:
        hand_level_index = [
            [2, 5, 9, 13, 17],
            [3, 6, 10, 14, 18, 0],
            [4, 7, 11, 15, 19, 1],
            [8, 12, 16, 20],
        ]

        weights = np.ones((21,))
        decay = 0.75
        for index in range(len(hand_level_index)):
            weights[hand_level_index[index]] = decay**index

        weights = np.repeat(weights, 2)

    opt_focal_length, loss, alignment_transform, ego_mano_pcd, exo_mano_pcd = (
        optimize_focal_length(ego_params, exo_params, weights)
    )

    o3d.io.write_point_cloud(
        f"{args.out_folder}/focal_length_test.ply",
        copy.deepcopy(ego_mano_pcd)
        .transform(alignment_transform)
        .paint_uniform_color([1, 0, 0])
        + copy.deepcopy(exo_mano_pcd).paint_uniform_color([0, 0, 1]),
    )

    print(f"Optimized focal length: {opt_focal_length}")
    print(f"Loss: {loss}")
    print(f"Alignment transform: {alignment_transform}")

    if USE_WEIGHTED:
        ego_params["ret_verts"] = ego_verts
        exo_params["ret_verts"] = exo_verts
        _, _, ego_mano_pcd, exo_mano_pcd = compute_alignment_loss(
            opt_focal_length, ego_params, exo_params, return_details=True
        )
        o3d.io.write_point_cloud(
            f"{args.out_folder}/weighted_focal_length_test.ply",
            copy.deepcopy(ego_mano_pcd)
            .transform(alignment_transform)
            .paint_uniform_color([1, 0, 0])
            + copy.deepcopy(exo_mano_pcd).paint_uniform_color([0, 0, 1]),
        )

    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(args.ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(args.exo_image)

    if USE_PSUEDO_DEPTH:
        vggt_model = initialize_vggt()

        ego_pcd: o3d.geometry.PointCloud = vggt_create_point_cloud(
            args.ego_image, vggt_model, mask=ego_mask
        )
        exo_pcd: o3d.geometry.PointCloud = vggt_create_point_cloud(
            args.exo_image, vggt_model, mask=exo_mask
        )
    else:
        ego_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext, mask=ego_mask)
        exo_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext, mask=exo_mask)

    ego_mano_pcd = pcd2dense(
        ego_mano_pcd,
        mano_faces,
        len(ego_pcd.points),
        viewable_indices=ego_viewable_indices,
    )
    exo_mano_pcd = pcd2dense(
        exo_mano_pcd,
        mano_faces,
        len(exo_pcd.points),
        viewable_indices=exo_viewable_indices,
    )

    if USE_PSUEDO_DEPTH:
        aligned_ego_mano_pcd, ego_transformation, ego_final_scale = (
            register_point_clouds_with_scale_ambiguity(
                source_pcd=ego_mano_pcd, target_pcd=ego_pcd
            )
        )
        print(f"Ego Final scale: {ego_final_scale}")
        aligned_exo_mano_pcd, exo_transformation, exo_final_scale = (
            register_point_clouds_with_scale_ambiguity(
                source_pcd=exo_mano_pcd, target_pcd=exo_pcd
            )
        )
        print(f"Exo Final scale: {exo_final_scale}")
    else:
        aligned_ego_mano_pcd, ego_transformation = register_point_clouds(
            source_pcd=ego_mano_pcd, target_pcd=ego_pcd
        )
        aligned_exo_mano_pcd, exo_transformation = register_point_clouds(
            source_pcd=exo_mano_pcd, target_pcd=exo_pcd
        )

    ## test
    o3d.io.write_point_cloud(
        f"{args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_ego_registration_test.ply",
        copy.deepcopy(ego_mano_pcd).transform(ego_transformation)
        + copy.deepcopy(ego_pcd),
    )

    o3d.io.write_point_cloud(
        f"{args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_exo_registration_test.ply",
        copy.deepcopy(exo_mano_pcd).transform(exo_transformation)
        + copy.deepcopy(exo_pcd),
    )

    ego_to_exo_transform = np.matmul(
        exo_transformation,
        np.matmul(alignment_transform, np.linalg.inv(ego_transformation)),
    )
    exo_to_ego_transform = np.linalg.inv(ego_to_exo_transform)

    if USE_PSUEDO_DEPTH:
        ego_total_pcd = vggt_create_point_cloud(args.ego_image, vggt_model)
        exo_total_pcd = vggt_create_point_cloud(args.exo_image, vggt_model)
    else:
        ego_total_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext)
        exo_total_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext)
    o3d.io.write_point_cloud(
        f"{args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_ego_total_pcd.ply",
        ego_total_pcd,
    )
    o3d.io.write_point_cloud(
        f"{args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_exo_total_pcd.ply",
        exo_total_pcd,
    )

    transformed_ego_total_pcd = copy.deepcopy(ego_total_pcd)
    transformed_ego_total_pcd.transform(ego_to_exo_transform)
    combined_total_pcd = transformed_ego_total_pcd + exo_total_pcd
    o3d.io.write_point_cloud(
        f"{args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_combined_pcd.ply",
        combined_total_pcd,
    )
    print(
        f"Saved combined point cloud to {args.out_folder}/{os.path.basename(args.ego_image).replace('.png', '')}_combined_pcd.ply"
    )

    ##################################

    # exo_rgb_path = args.exo_image
    # exo_cam_id = "cam2"
    # ego_cam_id = "cam4"

    # exo_rgb_path = exo_rgb_path.replace(exo_cam_id, exo_cam_id)
    # exo_depth_path = exo_rgb_path.replace("rgb", "depth")
    # exo_rgb_id = os.path.join("rgb", exo_rgb_path.split("/")[-1])
    # exo_cam_int_path = exo_rgb_path.replace(exo_rgb_id, "cam_intrinsics.txt")
    # exo_cam_ext_path = exo_rgb_path.replace("rgb", "cam_pose").replace("png", "txt")
    # exo_hand_path = exo_rgb_path.replace("rgb", "hand_pose").replace("png", "txt")
    # # egos
    # ego_cam_int_path = exo_cam_int_path.replace(exo_cam_id, ego_cam_id)
    # ego_cam_ext_path = exo_cam_ext_path.replace(exo_cam_id, ego_cam_id)
    # ego_hand_path = exo_hand_path.replace(exo_cam_id, ego_cam_id)
    # ego_rgb_pred = exo2ego_force(
    #     exo_rgb_path,
    #     exo_depth_path,
    #     exo_cam_int_path,
    #     exo_cam_ext_path,
    #     exo_hand_path,
    #     ego_cam_int_path,
    #     ego_cam_ext_path,
    #     ego_hand_path,
    #     np.linalg.inv(ego_to_exo_transform),
    # )
    # cv2.imwrite(
    #     f"{args.out_folder}/{os.path.basename(args.exo_image)}.png", ego_rgb_pred
    # )


if __name__ == "__main__":
    import glob
    import random

    # list image paths under /local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/
    ego_image_paths = glob.glob(
        "/local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/*.png"
    )

    # randomly mix
    random.shuffle(ego_image_paths)

    # cut 10 paths
    ego_image_paths = ego_image_paths[:10]

    # create exo by replacing with cam2
    exo_image_paths = [path.replace("cam4", "cam2") for path in ego_image_paths]

    for ego_image_path, exo_image_path in zip(ego_image_paths, exo_image_paths):
        try:
            print(f"Running main for image {ego_image_path}")
            loop_args = argparse.Namespace(
                ego_image=ego_image_path,
                exo_image=exo_image_path,
                hamer_checkpoint=DEFAULT_CHECKPOINT,
                body_detector="regnety",
                device="cuda",
                out_folder="output",
            )
            main(loop_args)
        except Exception as e:
            print(f"Error running main for image {ego_image_path}: {e}")
