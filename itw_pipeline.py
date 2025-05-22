import argparse
import json
from os.path import join
import os
from typing import Any
import cv2
import numpy as np
import open3d as o3d
import torch
from PIL import Image  # noqa: F401 (kept for downstream imports that expect it)

import open3d as o3d


from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import unproject_depth_map_to_point_map

from random import sample
import copy

from decord import VideoReader
from decord import cpu


REORDER = np.array(
    [
        4,
        3,
        2,
        1,
        8,
        7,
        6,
        5,
        12,
        11,
        10,
        9,
        16,
        15,
        14,
        13,
        20,
        19,
        18,
        17,
        0,
        25,
        24,
        23,
        22,
        29,
        28,
        27,
        26,
        33,
        32,
        31,
        30,
        37,
        36,
        35,
        34,
        41,
        40,
        39,
        38,
        21,
    ]
)

BONES = [
    # right
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 20],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 20],
    [8, 9],
    [9, 10],
    [10, 11],
    [11, 20],
    [12, 13],
    [13, 14],
    [14, 15],
    [15, 20],
    [16, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    # left
    [21, 22],
    [22, 23],
    [23, 24],
    [24, 41],
    [25, 26],
    [26, 27],
    [27, 28],
    [28, 41],
    [29, 30],
    [30, 31],
    [31, 32],
    [32, 41],
    [33, 34],
    [34, 35],
    [35, 36],
    [36, 41],
    [37, 38],
    [38, 39],
    [39, 40],
    [40, 41],
]


def vis_2d_skeleton(img, kps):
    """
    img: numpy array (H*W*C)
    kps: numpy array (42*2)
    """
    kps = kps[REORDER]
    skeleton_num = len(BONES)
    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i) for i in np.linspace(0, 1, skeleton_num + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    kp_mask = np.copy(img)

    for l in range(len(BONES)):
        i1 = BONES[l][0]
        i2 = BONES[l][1]
        p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
        p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
        cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
        )
        cv2.circle(
            kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
        )

    res = cv2.addWeighted(img, 0.0, kp_mask, 1.0, 0)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


EGO_PSEUDO_DEPTH = False
ALLO_PSEUDO_DEPTH = True
DEBUG = True

if DEBUG:
    import matplotlib.pyplot as plt


import sys, cv2
from acr.config import parse_args, ConfigContext
from acr.main import ACR


import torch.nn as nn
import torch.optim as optim

from ego_hand_pred.models import HandPoseEstimator
from ego_hand_pred.learners import load_checkpoint
from torchvision import transforms
from utils.alignment import umeyama_alignment, umeyama_alignment_pure
from utils.manipulation import points2image
import tqdm


def load_hand_pose_estimator(device):
    model = HandPoseEstimator("vit_224", root_relative=True).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    pretrained_path = "ego_hand_pred/checkpoint0100.pth"
    model, _, _ = load_checkpoint(model, optimizer, filename=pretrained_path)
    model.eval()
    return model


from contextlib import redirect_stdout
from io import StringIO


class NullIO(StringIO):
    def write(self, txt):
        pass


def silent(fn):
    """Decorator to silence functions."""

    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)

    return silent_fn


def run_acr(cv2_image, fov, focal_length):

    ################## Model Initialization ####################
    with ConfigContext(parse_args(sys.argv[1:])) as args_set:
        args_set.FOV = fov
        args_set.focal_length = focal_length
        acr = ACR(args_set=args_set)

    ################## RUN on image forlder ####################
    outputs, depth_map = acr(cv2_image, "res")
    return outputs, depth_map


def vggt_create_points_colors(
    image: np.ndarray,
    model: VGGT,
    mask=None,
    device="cuda",
    return_depth=False,
):
    images_tensor = load_and_preprocess_images([image]).to(device)
    H, W = images_tensor.shape[2:]
    original_image_pil = Image.fromarray(image).convert("RGB")
    original_W, original_H = original_image_pil.size
    resized_image_pil = original_image_pil.resize((W, H), Image.Resampling.LANCZOS)
    resized_image_np = np.array(resized_image_pil)
    colors = resized_image_np / 255.0

    with torch.no_grad():
        predictions = model(images_tensor)

    depth_pred = predictions["depth"].cpu().numpy().squeeze()  # (h_d, w_d)
    H_rgb, W_rgb = image.shape[:2]

    depth_pred = cv2.resize(depth_pred, (W_rgb, H_rgb), interpolation=cv2.INTER_NEAREST)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], (H, W))

    if extrinsic is None or intrinsic is None:
        print(
            "Error: Failed to extract camera parameters. extrinsic or intrinsic is None."
        )
        print(f"predictions[pose_enc] shape: {predictions['pose_enc'].shape}")
        raise ValueError("Failed to extract camera parameters.")

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

    points: np.ndarray = world_points_masked
    colors: np.ndarray = colors_masked

    cx = intrinsic_np[0, 2]
    cy = intrinsic_np[1, 2]
    fx = intrinsic_np[0, 0]
    fy = intrinsic_np[1, 1]
    x_ratio = H_rgb / H
    y_ratio = W_rgb / W

    cx = cx * x_ratio
    cy = cy * y_ratio
    fx = fx * x_ratio
    fy = fy * y_ratio

    intrinsic_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return points, colors, extrinsic_np, intrinsic_np, depth_pred


def estimate_scale_from_depth(
    gt_depth_mm: np.ndarray,
    pred_depth_units: np.ndarray,
    depth_scale: float = 4000.0,
    eps: float = 1e-8,
) -> float:
    """Robust global scale between predicted and GT depth maps.

    Ignores zero-masked GT pixels and drops the top 10% ratios as outliers.
    """
    H, W = gt_depth_mm.shape
    if pred_depth_units.shape != (H, W):
        pred_depth_units = cv2.resize(
            pred_depth_units, (W, H), interpolation=cv2.INTER_NEAREST
        )

    mask = (gt_depth_mm > 0) & np.isfinite(pred_depth_units)
    if not mask.any():
        raise ValueError("No valid depth overlap for scale estimation.")

    gt_m = gt_depth_mm.astype(np.float32) / depth_scale
    ratios = gt_m[mask] / (pred_depth_units[mask] + eps)
    ratios = ratios[np.isfinite(ratios)]
    if ratios.size == 0:
        raise ValueError("All ratio values are NaN/Inf.")

    q90 = np.percentile(ratios, 90)
    filtered_ratios = ratios[ratios <= q90]
    ratios = (
        filtered_ratios if filtered_ratios.size > 0 else ratios
    )  # fallback to full if empty
    return float(np.median(ratios))


def huber_scale(gt, pred, delta=1.0, eps=1e-8, iters=5):
    """
    Robust 1-parameter scale using IRLS with Huber weights.
      delta – where the loss switches from L2 to L1 (in **metres**)
    """
    gt = gt.flatten()
    pred = pred.flatten()

    mask = (gt > 0) & np.isfinite(pred)
    if not mask.any():
        raise ValueError("No valid depth overlap.")

    p = pred[mask].astype(np.float32)
    g = gt[mask]

    # initialise with plain least-squares
    s = (p * g).sum() / ((p * p).sum() + eps)

    for _ in range(iters):
        r = g - s * p  # residuals
        w = np.minimum(1, delta / (np.abs(r) + eps))  # Huber weights
        s = (w * p * g).sum() / ((w * p * p).sum() + eps)

    return float(s)


def ransac_scale(gt, pred, n_iter=100, min_inliers=0.5, thresh=0.05):
    gt = gt.flatten()
    pred = pred.flatten()
    mask = (gt > 0) & np.isfinite(pred)
    p, g = pred[mask].ravel(), gt[mask].ravel()

    best_s, best_in = None, 0
    N = p.size
    for _ in range(n_iter):
        i = sample(range(N), 2)  # 2-point model fits a scale
        s = (g[i] @ p[i]) / (p[i] @ p[i])
        inliers = np.abs(g - s * p) < thresh
        n_in = np.count_nonzero(inliers)
        if n_in > best_in:
            best_s, best_in = s, n_in

    if best_in < min_inliers * N:
        raise RuntimeError("RANSAC failed")

    # re-fit on inliers for a cleaner estimate
    inliers = np.abs(g - best_s * p) < thresh
    p_i, g_i = p[inliers], g[inliers]
    s = (p_i * g_i).sum() / (p_i * p_i).sum()
    return float(s)


def robust_scale_from_depth(gt_m, pred_u, eps=1e-8):
    """
    One-parameter scale between allocentric VGGT depth (§pred_u§)
    and rendered world depth (§gt_m§, metres).
    * Ignores invalid/∞ pixels
    * IRLS-Huber refinement after a RANSAC initialisation
    """
    # ---------- flatten & mask
    mask: Any = np.isfinite(gt_m) & np.isfinite(pred_u) & (gt_m > 0)
    if not mask.any():
        raise RuntimeError("No depth overlap")

    g = gt_m[mask].astype(np.float32)
    p = pred_u[mask].astype(np.float32)

    # ---------- RANSAC (2-point model) for a hardy initial guess
    try:
        s0 = ransac_scale(g, p)  # reuse your fn
        return s0

    except RuntimeError:
        s = huber_scale(g, p, delta=0.05)  # 5 cm switch
        return s


###############################################################################
# ---------------------------  GT RGB‑D point‑cloud -------------------------- #
###############################################################################


def rgbd2points_colors(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast RGB-D → Open3D PointCloud (NumPy only).
    """
    # Convert and mask depth in meters
    u, v = np.meshgrid(np.arange(rgb.shape[1]), np.arange(rgb.shape[0]))

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    u_flat = u.flatten()
    v_flat = v.flatten()
    depth = depth.flatten()

    # Unproject to 3D camera coordinates
    x_cam = (u_flat - cx) * depth / fx
    y_cam = (v_flat - cy) * depth / fy
    z_cam = depth

    # Stack to form 3D points
    points = np.stack((x_cam, y_cam, z_cam), axis=-1)
    homo_points = np.concatenate((points, np.ones((points.shape[0], 1))), axis=-1)

    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0

    return homo_points, colors


def points_project_to_2d(points: np.ndarray, K: np.ndarray, device: str = "cuda"):
    """
    Project 3D points to 2D points
    """
    pts = torch.as_tensor(points[:, :3], dtype=torch.float32, device=device)  # (N,3)
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)  # (3,3)

    # --- 2. Project to pixel space -------------------------------------------------------
    pix_cam = (K_t @ pts.T).T  # (N,3) in homogeneous px coords
    pix_uv = pix_cam[:, :2] / pix_cam[:, 2:3]  # perspective divide
    return pix_uv  # (N,2)


def points_colors2rgbd_torch(
    points: np.ndarray,
    colors: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    height_ratio: int = 1,
    width_ratio: int = 1,
    device: str = "cuda",
):
    if not torch.cuda.is_available() and (
        device == "cuda" or str(device).startswith("cuda")
    ):
        device = "cpu"  # graceful fallback

    # --- 1. Load → Torch -----------------------------------------------------------------
    pts = torch.as_tensor(points[:, :3], dtype=torch.float32, device=device)  # (N,3)
    cols = torch.as_tensor(
        colors, dtype=torch.float32, device=device
    )  # (N,3) RGB ∈ [0,1]
    K_t = torch.as_tensor(K, dtype=torch.float32, device=device)  # (3,3)

    # --- 2. Project to pixel space -------------------------------------------------------
    pix_cam = (K_t @ pts.T).T  # (N,3) in homogeneous px coords
    pix_uv = pix_cam[:, :2] / pix_cam[:, 2:3]  # perspective divide

    u = torch.round(pix_uv[:, 0]).to(torch.int64)
    v = torch.round(pix_uv[:, 1]).to(torch.int64)

    mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if mask.sum() == 0:
        return (
            np.zeros((height, width, 3), np.uint8),
            np.zeros((height // height_ratio, width // width_ratio), np.float32),
        )

    u = u[mask]
    v = v[mask]
    depth = pix_cam[mask, 2]  # Z in metres
    cols = cols[mask]

    # --- 3. Z-buffer : sort back-to-front so nearest wins -------------------------------
    order = torch.argsort(depth, descending=True)  # far → near
    u, v, depth, cols = u[order], v[order], depth[order], cols[order]

    # --- 4. Scatter to image & depth tensors --------------------------------------------
    lin_px = v * width + u  # flattened image index
    img = torch.zeros((height * width, 3), dtype=torch.uint8, device=device)
    img.index_put_(  # last assignment (=nearest) wins
        (lin_px,), (cols * 255).to(torch.uint8), accumulate=False
    )

    # depth (optionally down-sampled)
    tgt_H, tgt_W = height // height_ratio, width // width_ratio
    lin_dm = (v // height_ratio) * tgt_W + (u // width_ratio)
    depth_map = torch.zeros(tgt_H * tgt_W, dtype=torch.float32, device=device)
    depth_map.index_put_((lin_dm,), depth, accumulate=False)

    # --- 5. (Optional) morphological dilation to match cv2.circle radius=3 ------------
    img = (
        img.view(height, width, 3).permute(2, 0, 1).unsqueeze(0).float() / 255
    )  # (1,3,H,W)
    # disc kernel radius=3 → (7×7) kernel; implemented as max-pool2d for each channel
    pad = 3
    img_dil = torch.nn.functional.max_pool2d(img, kernel_size=7, stride=1, padding=pad)
    img = (img_dil.squeeze(0).permute(1, 2, 0) * 255).to(torch.uint8)

    # --- 6. Finalise & return -----------------------------------------------------------
    img_bgr = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_RGB2BGR)
    depth_cpu = depth_map.view(tgt_H, tgt_W).cpu().numpy()

    return img_bgr, depth_cpu


def overlay_images(background_image, foreground_image):
    # Blend the two images
    alpha = 0.5  # Transparency factor for the original frame
    beta = 1.0 - alpha  # Transparency factor for the point cloud rendering
    gamma = 0.0  # Scalar added to each sum

    overlaid_image = cv2.addWeighted(
        background_image, alpha, foreground_image, beta, gamma
    )

    return overlaid_image


def process_video(video_path):

    vr = VideoReader(video_path, ctx=cpu(0))
    height, width = vr[0].asnumpy().shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VGGT()
    model.load_state_dict(
        torch.hub.load_state_dict_from_url(
            "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt",
            map_location=device,
        )
    )
    model.eval()
    model = model.to(device)

    hand_pose_estimator = load_hand_pose_estimator(device)

    save_folder = video_path.split("/")[-1].split(".")[0]
    video_parent_folder = "/".join(video_path.split("/")[:-1])
    save_folder = join(video_parent_folder, save_folder)
    os.makedirs(save_folder, exist_ok=True)

    for frame_number in tqdm.tqdm(range(len(vr))):
        if frame_number != 17:
            continue
        print(f"Processing frame {frame_number}")

        frame: np.ndarray = vr[frame_number].asnumpy()

        if DEBUG:
            cv2.imwrite(
                join(save_folder, f"org_{frame_number:04d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        # save point cloud
        vggt_points, colors, extrinsic, intrinsic, depth_pred = (
            vggt_create_points_colors(frame, model)
        )

        if DEBUG:
            # apply minmax to depth_pred
            debug_depth_pred = depth_pred.copy()
            debug_depth_pred = (debug_depth_pred - debug_depth_pred.min()) / (
                debug_depth_pred.max() - debug_depth_pred.min()
            )
            debug_depth_pred = (debug_depth_pred * 255).astype(np.uint8)
            cv2.imwrite(
                join(save_folder, f"vggt_{frame_number:04d}.png"),
                debug_depth_pred,
            )

        # Calculate focal length, point of view (POV), and dimensions
        fx, fy = intrinsic[0, 0], intrinsic[1, 1]  # Focal lengths
        cx, cy = intrinsic[0, 2], intrinsic[1, 2]  # Principal point

        # Calculate width and height from principal point
        width = frame.shape[1]
        height = frame.shape[0]

        # Calculate field of view (in degrees)
        fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi

        outputs, depth_hands = silent(run_acr)(frame, fov_y, fy)
        if DEBUG:
            debug_depth_hands = depth_hands[0].copy()
            debug_depth_hands = (debug_depth_hands - debug_depth_hands.min()) / (
                debug_depth_hands.max() - debug_depth_hands.min()
            )
            debug_depth_hands = (debug_depth_hands * 255).astype(np.uint8)
            cv2.imwrite(
                join(save_folder, f"acr_{frame_number:04d}.png"),
                debug_depth_hands,
            )

        depth_hands = depth_hands[0]

        # Create masks for non-zero values in hand_depth_map
        hand_mask = depth_hands > 0
        hand_depths = depth_hands[hand_mask]

        vggt_depths_at_hand = depth_pred[hand_mask]

        # Calculate the median ratio between hand depths and vggt depths
        # Using median is more robust to outliers than mean
        ratios = hand_depths / vggt_depths_at_hand
        median_ratio = np.median(ratios)

        # Scale the vggt points
        scaled_vggt_points = vggt_points * median_ratio

        pil_frame = Image.fromarray(frame)

        transform_frame = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
            ]
        )

        hand_frame = torch.as_tensor(transform_frame(pil_frame))
        hand_frame = hand_frame.unsqueeze(0)
        hand_frame = hand_frame.to(device)

        preds_pose, preds_root = hand_pose_estimator(hand_frame)
        preds_pose[:, :21, :] += preds_root[:, :1, :]
        preds_pose[:, 21:, :] += preds_root[:, 1:, :]

        pred_ego_j3d = preds_pose.detach().cpu().numpy().squeeze(0)
        np.save(join(save_folder, f"{frame_number:04d}.npy"), pred_ego_j3d)

        pred_ego_j3d_2d = points_project_to_2d(pred_ego_j3d, intrinsic).cpu().numpy()
        np.save(join(save_folder, f"{frame_number:04d}_2d.npy"), pred_ego_j3d_2d)
        if DEBUG:
            debug_black_frame = np.zeros_like(frame)
            debug_pred_ego_j3d_2d = vis_2d_skeleton(debug_black_frame, pred_ego_j3d_2d)
            cv2.imwrite(
                join(save_folder, f"pred_ego_j3d_2d_{frame_number:04d}.png"),
                debug_pred_ego_j3d_2d,
            )

        if len(outputs["res"]) != 2:
            print(f"Frame {frame_number} has {len(outputs['res'])} hands")
            continue

        left_hand = outputs["res"][0]
        right_hand = outputs["res"][1]

        left_cam_transform = left_hand["cam_trans"]
        right_cam_transform = right_hand["cam_trans"]

        left_j3d = left_hand["j3d"]  # (21,3)
        right_j3d = right_hand["j3d"]  # (21,3)

        left_j3d = left_j3d + left_cam_transform
        right_j3d = right_j3d + right_cam_transform

        acr_exo_j3d = np.concatenate([left_j3d, right_j3d], axis=0)  # (42,3)

        if DEBUG:
            debug_acr_exo_j3d_2d = vis_2d_skeleton(
                frame, points_project_to_2d(acr_exo_j3d, intrinsic).cpu().numpy()
            )
            cv2.imwrite(
                join(save_folder, f"acr_exo_j3d_{frame_number:04d}.png"),
                debug_acr_exo_j3d_2d,
            )

        transformation = umeyama_alignment_pure(pred_ego_j3d, acr_exo_j3d)

        image = points2image(
            scaled_vggt_points,
            colors * 255.0,
            np.linalg.inv(transformation),
            intrinsic,
            width,
            height,
        )

        # plot just to be sure
        plt.figure()
        plt.imshow(image)
        plt.scatter(pred_ego_j3d_2d[:, 0], pred_ego_j3d_2d[:, 1], c="red", s=10)
        plt.savefig(join(save_folder, f"{frame_number:04d}_ego_j3d.png"))
        plt.close()

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        cv2.imwrite(join(save_folder, f"{frame_number:04d}.png"), image)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    return parser.parse_args()


def main():
    from glob import glob

    args = arg_parse()
    video_folder = args.video_folder
    # for video_path in tqdm.tqdm(glob(join(video_folder, "*.mov"))):
    #     print(f"Processing {video_path}")
    #     process_video(video_path)

    video_path = join(video_folder, "1.mov")
    process_video(video_path)


if __name__ == "__main__":
    main()
