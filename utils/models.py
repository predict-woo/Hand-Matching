# Local imports
from hamer.configs import CACHE_DIR_HAMER
from hamer.utils import recursive_to
import torch
from pathlib import Path
import numpy as np
from torch.utils.data.dataloader import default_collate
import cv2
import os
from typing import Any
from vggt.models.vggt import VGGT
import open3d as o3d
from hamer.utils.renderer import cam_crop_to_full
from hamer.datasets.vitdet_dataset import ViTDetDataset
from PIL import Image
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
DEFAULT_FOCAL_LENGTH_BOUNDS = (1, 5000)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5


def print_GPU_memory():
    """
    Print the memory usage of the GPU.
    """
    print(f"GPU memory usage: {torch.cuda.memory_allocated() / 1024 ** 2} MB")


def initialize_hamer(checkpoint, device="cuda"):
    """
    Initialize models, detector, and renderer.

    Args:
        args: Command line arguments

    Returns:
        tuple: (model, model_cfg, device, detector, cpm, renderer)
    """
    from hamer.models import download_models, load_hamer

    print(f"Initializing HAMER from {checkpoint}")
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(checkpoint)

    model = model.to(device)
    model.eval()
    torch.cuda.empty_cache()
    print_GPU_memory()
    return model, model_cfg


def initialize_detector(body_detector="regnety"):
    """
    Initialize the body detector.

    Args:
        body_detector (str): The body detector to use.

    Returns:
        detector: The body detector.
    """
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    print(f"Initializing body detector: {body_detector}")
    if body_detector == "vitdet":
        from detectron2.config import LazyConfig
        import hamer

        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif body_detector == "regnety":
        from detectron2 import model_zoo

        detectron2_cfg = model_zoo.get_config(
            "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
        )
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        raise ValueError(f"Invalid body detector: {body_detector}")
    torch.cuda.empty_cache()
    print_GPU_memory()
    return detector


def initialize_vitpose(device="cuda"):
    """
    Initialize the ViTPose model.

    Args:
        checkpoint (str): The path to the checkpoint file.
        device (str): The device to run the model on.

    Returns:
        cpm: The ViTPose model.
    """
    from utils.vitpose_model import ViTPoseModel

    print(f"Initializing ViTPose model")
    cpm = ViTPoseModel(device)
    torch.cuda.empty_cache()
    print_GPU_memory()
    return cpm


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

    batch: Any = recursive_to(default_collate([dataset[0], dataset[1]]), device)

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

    is_right: torch.Tensor = batch["right"]

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


def initialize_vggt(device="cuda"):
    """
    Initialize the VGG-T model.
    """

    model = VGGT()
    print(f"Loading model to {device}...")
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=device,
        )
    )
    model.eval()
    model = model.to(device)
    print_GPU_memory()
    return model


def vggt_create_point_cloud(
    image_path: str,
    model: VGGT,
    mask=None,
    device="cuda",
):

    image_names = [image_path]
    images_tensor = load_and_preprocess_images(image_names).to(device)

    H, W = images_tensor.shape[2:]

    original_image_pil = Image.open(image_path).convert("RGB")
    resized_image_pil = original_image_pil.resize((W, H), Image.Resampling.LANCZOS)
    resized_image_np = np.array(resized_image_pil)
    colors = resized_image_np / 255.0

    with torch.no_grad():
        predictions = model(images_tensor)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    # Check if values are None before processing
    if extrinsic is None or intrinsic is None:
        print(
            "Error: Failed to extract camera parameters. extrinsic or intrinsic is None."
        )
        print(f"predictions[pose_enc] shape: {predictions['pose_enc'].shape}")
        return

    depth_map_np = predictions["depth"].cpu().numpy().squeeze(0).squeeze(0)
    extrinsic_np = extrinsic.cpu().numpy().squeeze(0).squeeze(0)
    intrinsic_np = intrinsic.cpu().numpy().squeeze(0).squeeze(0)

    depth_map_np_unsqueezed = np.expand_dims(depth_map_np, axis=0)  # (1, H, W, 1)
    extrinsic_np_unsqueezed = np.expand_dims(extrinsic_np, axis=0)  # (1, 4, 3)
    intrinsic_np_unsqueezed = np.expand_dims(intrinsic_np, axis=0)  # (1, 3, 3)

    world_points_np = unproject_depth_map_to_point_map(
        depth_map_np_unsqueezed, extrinsic_np_unsqueezed, intrinsic_np_unsqueezed
    )
    world_points_np = world_points_np.squeeze(0)

    if mask is not None:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize((W, H), Image.Resampling.NEAREST)
        mask_np = np.array(mask_pil)
        mask_np = mask_np > 0

        world_points_masked = world_points_np[~mask_np]
        colors_masked = colors[~mask_np]
    else:
        world_points_masked = world_points_np.reshape(-1, 3)
        colors_masked = colors.reshape(-1, 3)

    points = world_points_masked
    colors = colors_masked

    pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
