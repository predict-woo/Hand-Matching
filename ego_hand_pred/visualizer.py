from numpy import ndarray
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from os.path import join
from decord import VideoReader, cpu
import torch


def get_frame_decord(path: str, n: int):
    vr = VideoReader(path, ctx=cpu(0))
    if n < 0 or n >= len(vr):
        raise IndexError(f"Frame index {n} out of bounds (0–{len(vr)-1})")
    # returns a H×W×3 RGB NDArray
    frame = vr[n].asnumpy()
    return frame


def mp42imgs(video_path, return_rgb=False, max_cnt=None):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    imgs = []

    suc = cap.isOpened()
    frame_cnt = -1
    while True:
        frame_cnt += 1
        suc, img = cap.read()
        if not suc:
            break
        if return_rgb:
            img = img[:, :, ::-1].astype(np.uint8)  # bgr2rgb
        imgs.append(img)
        if (not max_cnt is None) and (frame_cnt + 1 >= max_cnt):
            break
    cap.release()

    return imgs


ROOT_PATH = "/local/home/andrye/dev/TACO-Instructions/TACO_Hands"
TACO_PATH = "/local/home/andrye/dev/TACO-Instructions/TACO"

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


# Allocentric_RGB_Videos/(screw, screwdriver, box)/20231102_009/22070938.mp4


K_path = join(
    TACO_PATH,
    "Egocentric_Camera_Parameters",
    "(screw, screwdriver, box)",
    "20231102_009",
    "egocentric_intrinsic.txt",
)

K = np.loadtxt(K_path).reshape(3, 3)
F = (K[0, 0], K[1, 1])
C = (K[0, 2], K[1, 2])


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / cam_coord[:, 2] * f[0] + c[0]
    y = cam_coord[:, 1] / cam_coord[:, 2] * f[1] + c[1]
    z = cam_coord[:, 2]
    return np.stack((x, y, z), 1)


def vis_keypoint_pure(
    save_paths: list[str],
    pred_poses: list[torch.Tensor],
    save_folder: str,
):
    for i, (save_path, pred_pose) in enumerate(zip(save_paths, pred_poses)):
        pred_pose = pred_pose.detach().cpu().numpy()
        black_ego = np.zeros((1080, 1920, 3), dtype=np.uint8)
        pred_pose_2d = cam2pixel(pred_pose, F, C)
        black_ego = vis_2d_skeleton(black_ego, pred_pose_2d, "pred")
        black_ego_save_path = os.path.join(save_folder, save_path)

        os.makedirs(os.path.dirname(black_ego_save_path), exist_ok=True)
        cv2.imwrite(black_ego_save_path, black_ego)


def vis_keypoint(image_paths, pred_poses, targets, save_folder):
    for image_path, pred_pose, target in zip(image_paths, pred_poses, targets):
        exo = cv2.imread(image_path)

        # /local/home/andrye/dev/TACO-Instructions/TACO/Allocentric_RGB_Videos/(brush, brush, kettle)/20231006_165/22070938/000055.png
        triplet = image_path.split("/")[-4]
        sequence_name = image_path.split("/")[-3]
        sequence_number = image_path.split("/")[-2]
        frame_number = image_path.split("/")[-1].split(".")[0]

        ego_video_path = join(
            TACO_PATH,
            "Egocentric_RGB_Videos",
            triplet,
            sequence_name,
            "color.mp4",
        )

        ego_frame = get_frame_decord(ego_video_path, int(frame_number) - 1)
        ego = cv2.cvtColor(ego_frame, cv2.COLOR_RGB2BGR)
        black_ego = np.zeros_like(ego)

        pred_pose = pred_pose.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        vis_3d = vis_3d_skeleton([pred_pose, target], ["pred", "gt"])
        pred_pose_2d = cam2pixel(pred_pose, F, C)
        target_2d = cam2pixel(target, F, C)
        ego = vis_2d_skeleton(ego, pred_pose_2d, "pred")
        ego = vis_2d_skeleton(ego, target_2d, "gt")
        black_ego = vis_2d_skeleton(black_ego, pred_pose_2d, "pred")
        exo_save_path = image_path.replace(TACO_PATH, save_folder).replace(
            ".png", "_exo.png"
        )
        ego_save_path = image_path.replace(TACO_PATH, save_folder).replace(
            ".png", "_ego.png"
        )
        vis_3d_save_path = image_path.replace(TACO_PATH, save_folder).replace(
            ".png", "_vis_3d.png"
        )
        black_ego_save_path = image_path.replace(TACO_PATH, save_folder).replace(
            ".png", "_black_ego.png"
        )

        os.makedirs(os.path.dirname(exo_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(ego_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(vis_3d_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(black_ego_save_path), exist_ok=True)

        cv2.imwrite(exo_save_path, exo)
        cv2.imwrite(ego_save_path, ego)
        cv2.imwrite(vis_3d_save_path, vis_3d)
        cv2.imwrite(black_ego_save_path, black_ego)


def vis_2d_skeleton(img, kps, title):
    kps = kps[REORDER]
    skeleton_num = len(BONES)
    if title == "pred":
        color = (255, 255, 0)
    elif title == "gt":
        color = (0, 255, 0)
    colors = [color for _ in range(skeleton_num)]

    kp_mask = np.copy(img)

    for l in range(len(BONES)):
        i1 = BONES[l][0]
        i2 = BONES[l][1]
        p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
        p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
        cv2.line(kp_mask, p1, p2, color=colors[l], thickness=6, lineType=cv2.LINE_AA)
        cv2.circle(
            kp_mask, p1, radius=8, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
        )
        cv2.circle(
            kp_mask, p2, radius=8, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
        )

    return cv2.addWeighted(img, 0.0, kp_mask, 1.0, 0)


def vis_3d_skeleton(kpt_3ds, titles):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for kpt_3d, title in zip(kpt_3ds, titles):
        kpt_3d = kpt_3d[REORDER]
        skeleton_num = len(BONES)

        if title == "pred":
            color = (1, 0, 0)
        elif title == "gt":
            color = (0, 1, 0)
        colors = [color for _ in range(skeleton_num)]

        for l in range(len(BONES)):
            i1 = BONES[l][0]
            i2 = BONES[l][1]
            x = np.array([kpt_3d[i1, 0], kpt_3d[i2, 0]])
            y = np.array([kpt_3d[i1, 1], kpt_3d[i2, 1]])
            z = np.array([kpt_3d[i1, 2], kpt_3d[i2, 2]])

            ax.plot(x, z, -y, c=colors[l], linewidth=2)
            ax.scatter(
                kpt_3d[i1, 0], kpt_3d[i1, 2], -kpt_3d[i1, 1], c=colors[l], marker="o"
            )
            ax.scatter(
                kpt_3d[i2, 0], kpt_3d[i2, 2], -kpt_3d[i2, 1], c=colors[l], marker="o"
            )

    fig.canvas.draw()
    img_np = np.array(fig.canvas.renderer._renderer)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.resize(img_bgr, (1920, 1080))

    return img_bgr


# def vis_2d_skeleton(img, kps):
#     """
#     img: numpy array (H*W*C)
#     kps: numpy array (42*2)
#     """
#     kps = kps[REORDER]
#     skeleton_num = len(BONES)
#     cmap = plt.get_cmap("rainbow")
#     colors = [cmap(i) for i in np.linspace(0, 1, skeleton_num + 2)]
#     colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

#     kp_mask = np.copy(img)

#     for l in range(len(BONES)):
#         i1 = BONES[l][0]
#         i2 = BONES[l][1]
#         p1 = kps[i1, 0].astype(np.int32), kps[i1, 1].astype(np.int32)
#         p2 = kps[i2, 0].astype(np.int32), kps[i2, 1].astype(np.int32)
#         cv2.line(kp_mask, p1, p2, color=colors[l], thickness=2, lineType=cv2.LINE_AA)
#         cv2.circle(
#             kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
#         )
#         cv2.circle(
#             kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA
#         )

#     res = cv2.addWeighted(img, 0.0, kp_mask, 1.0, 0)
#     res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
#     return res
