import cv2
import os
import numpy as np


def overlay_cam_view(image, cam_view):
    input_img = image.astype(np.float32)[:, :, ::-1] / 255.0
    input_img = np.concatenate(
        [input_img, np.ones_like(input_img[:, :, :1])], axis=2
    )  # Add alpha channel
    input_img_overlay = (
        input_img[:, :, :3] * (1 - cam_view[:, :, 3:])
        + cam_view[:, :, :3] * cam_view[:, :, 3:]
    )
    return 255 * input_img_overlay[:, :, ::-1]
