import torch.nn as nn
from torchvision import models
import os

class HandPoseEstimator(nn.Module):
    def __init__(self, model_name, root_relative=False):
        super(HandPoseEstimator, self).__init__()

        self.model_name = model_name
        if self.model_name == 'vit_224':
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.vit.heads = nn.Identity()
            self.feat_dim = 768
        elif self.model_name == 'vit_384':
            self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.vit.heads = nn.Identity()
            self.feat_dim = 768
        elif self.model_name == 'r50':
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            self.resnet.fc = nn.Identity()
            self.feat_dim = 2048

        self.hidden_dim = 512
        self.num_joints = 42
        self.joint_dim = 3
        
        self.pose_regressor = nn.Sequential(
            nn.Linear(self.feat_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_joints * self.joint_dim)
        )

        self.root_relative = root_relative
        if self.root_relative:
            self.num_roots = 2
            self.root_regressor = nn.Sequential(
                nn.Linear(self.feat_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.num_roots * self.joint_dim)
            )
            self.ckpt_folder = os.path.join("ckpts", self.model_name, "root_relative")
        else:
            self.ckpt_folder = os.path.join("ckpts", self.model_name, "absolute")
        
        os.makedirs(self.ckpt_folder, exist_ok=True)

    def forward(self, x):
        if self.model_name in ['vit_224', 'vit_384']:
            features = self.vit(x)
        elif self.model_name in ['r50']:
            features = self.resnet(x)
        pose_output = self.pose_regressor(features)
        pose = pose_output.view(-1, self.num_joints, self.joint_dim)
        if self.root_relative:
            root_output = self.root_regressor(features)
            root = root_output.view(-1, self.num_roots, self.joint_dim)
            return pose, root
        else:
            return pose

