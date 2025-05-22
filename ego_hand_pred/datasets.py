from typing import Any


import torch
from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
import os
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms

from decord import VideoReader, cpu


def get_frame_decord(path: str, n: int):
    vr = VideoReader(path, ctx=cpu(0))
    if n < 0 or n >= len(vr):
        raise IndexError(f"Frame index {n} out of bounds (0–{len(vr)-1})")
    frame = vr[n].asnumpy()
    return frame


ROOT_PATH = "/local/home/andrye/dev/TACO-Instructions/TACO_Hands"
TACO_PATH = "/local/home/andrye/dev/TACO-Instructions/TACO"


class HandPoseDataset(Dataset):
    def __init__(
        self,
        mode="train",
        input_size=224,
        root_relative=False,
        sequence_number="22070938",
    ):
        self.root_relative = root_relative
        self.mode: str = mode
        with open(os.path.join(ROOT_PATH, f"taco_{mode}_list.txt"), "r") as file:
            self.video_paths = [
                os.path.join(TACO_PATH, line.strip()) for line in file.readlines()
            ]

        self.hand_pose_paths = []
        for video_dir in self.video_paths:
            triplet = video_dir.split("/")[-3]
            sequence_name = video_dir.split("/")[-2]
            sequence_number = video_dir.split("/")[-1].split(".")[0]
            self.hand_pose_paths.append(
                os.path.join(
                    ROOT_PATH,
                    self.mode,
                    triplet,
                    sequence_name,
                    "3d_hand_poses.npy",
                )
            )

        # Create frame lookup table
        self.frame_lookup = []
        global_idx = 0

        for video_idx, (video_path, hand_pose_path) in enumerate(
            zip(self.video_paths, self.hand_pose_paths)
        ):
            num_frames = np.load(hand_pose_path).shape[0]
            frame_parent = video_path.replace(".mp4", "")
            for local_frame_idx in range(num_frames):

                self.frame_lookup.append(
                    {
                        "frame_path": os.path.join(
                            frame_parent, f"{local_frame_idx+1:06}.png"
                        ),
                        "hand_pose_path": hand_pose_path,
                        "frame_idx": local_frame_idx,
                    }
                )
                global_idx += 1

        self.length = len(self.frame_lookup)

        self.transform = transforms.Compose(
            [
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.to(torch.float32)),
            ]
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        # Get video and frame info from lookup table
        entry = self.frame_lookup[idx]
        frame_path = entry["frame_path"]
        hand_pose_path = entry["hand_pose_path"]
        frame_idx = entry["frame_idx"]
        # Get the frame using the lookup information
        image = cv2.imread(frame_path)
        image = Image.fromarray(image)
        image = self.transform(image)

        pose = np.load(hand_pose_path)[frame_idx]
        pose = torch.tensor(pose, dtype=torch.float32)  # (42,3)

        pose_right = pose[:21].reshape(21, 3)
        pose_left = pose[21:].reshape(21, 3)
        if self.root_relative:
            root_left = pose_left.clone()[:1]
            root_right = pose_right.clone()[:1]
            pose_left -= root_left
            pose_right -= root_right
            root = np.concatenate([root_left, root_right])
            root = torch.tensor(root, dtype=torch.float32)
        pose = np.concatenate([pose_left, pose_right])
        pose = torch.tensor(pose, dtype=torch.float32)

        if self.root_relative:
            if self.mode == "test":
                return image, (pose, root), frame_path
            else:
                return image, (pose, root)
        else:
            if self.mode == "test":
                return image, pose, frame_path
            else:
                return image, pose

        return image, pose


if __name__ == "__main__":
    import tqdm

    dataset = HandPoseDataset(mode="train", sequence_number="22070938")
    image, pose = dataset[100]
    print(image.shape)
    print(pose.shape)
    # speed test
    # for i in tqdm.tqdm(range(len(dataset))):
    #     dataset[i]
