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
    vit_pose_detection,
)
from hamer.utils import recursive_to


from utils.alignment import (
    optimize_focal_length,
    FocalLengthOptArgs,
    register_point_clouds,
)
from utils.data import load_from_rgb_path
from utils.manipulation import depth2pcd, pcd2dense

import open3d as o3d
import copy
import numpy as np
from typing import Any


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--hamer_checkpoint", default=DEFAULT_CHECKPOINT, type=str)
    args.add_argument("--body_detector", default="regnety", type=str)
    args.add_argument("--device", default="cuda", type=str)
    args.add_argument("--out_folder", default="output", type=str)
    args.add_argument(
        "--ego_image",
        default="/local/home/andrye/dev/H2O/subject1/h1/2/cam4/rgb/000043.png",
        type=str,
    )
    args.add_argument(
        "--exo_image",
        default="/local/home/andrye/dev/H2O/subject1/h1/2/cam2/rgb/000043.png",
        type=str,
    )
    args = args.parse_args()

    hamer_model, hamer_model_cfg = initialize_hamer(args.hamer_checkpoint, args.device)
    detector = initialize_detector(args.body_detector)
    cpm = initialize_vitpose(args.device)
    renderer = Renderer(hamer_model_cfg, faces=hamer_model.mano.faces)

    ego_image = cv2.imread(args.ego_image)
    exo_image = cv2.imread(args.exo_image)

    images = [ego_image, exo_image]
    params = []
    masks = []

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

        mask = renderer.render_mask_multiple(
            all_verts,
            cam_t=all_cam_t,
            render_res=img_size[0],
            is_right=all_right,
            focal_length=scaled_focal_length,
        )

        image_processed_result: FocalLengthOptArgs = {
            "cam_bbox": pred_cam,
            "box_center": box_center,
            "box_size": box_size,
            "img_size": img_size,
            "focal_length": scaled_focal_length,
            "ret_verts": ret_verts,
            "is_right": is_right,
        }

        params.append(image_processed_result)
        masks.append(mask)

    ego_mask = masks[0]
    exo_mask = masks[1]

    ego_params: FocalLengthOptArgs = params[0]
    exo_params: FocalLengthOptArgs = params[1]

    opt_focal_length, loss, alignment_transform, ego_mano_pcd, exo_mano_pcd = (
        optimize_focal_length(ego_params, exo_params)
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

    ego_depth, ego_rgb, ego_cam_int, ego_cam_ext = load_from_rgb_path(args.ego_image)
    exo_depth, exo_rgb, exo_cam_int, exo_cam_ext = load_from_rgb_path(args.exo_image)
    ego_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext, mask=ego_mask)
    exo_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext, mask=exo_mask)
    ego_mano_pcd = pcd2dense(ego_mano_pcd, hamer_model.mano.faces, len(ego_pcd.points))
    exo_mano_pcd = pcd2dense(exo_mano_pcd, hamer_model.mano.faces, len(exo_pcd.points))

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

    ego_to_exo_transform = np.matmul(
        exo_transformation,
        np.matmul(alignment_transform, np.linalg.inv(ego_transformation)),
    )
    exo_to_ego_transform = np.linalg.inv(ego_to_exo_transform)

    ego_total_pcd = depth2pcd(ego_depth, ego_rgb, ego_cam_int, ego_cam_ext)
    exo_total_pcd = depth2pcd(exo_depth, exo_rgb, exo_cam_int, exo_cam_ext)

    transformed_ego_total_pcd = copy.deepcopy(ego_total_pcd)
    transformed_ego_total_pcd.transform(ego_to_exo_transform)
    combined_total_pcd = transformed_ego_total_pcd + exo_total_pcd
    o3d.io.write_point_cloud(f"{args.out_folder}/combined_pcd.ply", combined_total_pcd)
    print(f"Saved combined point cloud to {args.out_folder}/combined_pcd.ply")


if __name__ == "__main__":
    main()
