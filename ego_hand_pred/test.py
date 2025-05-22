import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

from models import HandPoseEstimator
from datasets import HandPoseDataset
from learners import evaluate, load_checkpoint


test_dataset = HandPoseDataset(mode="test", input_size=224, root_relative=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandPoseEstimator("vit_224", root_relative=True).to(device)
model = torch.compile(model)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

pretrained_path = "ckpts/vit_224/root_relative/checkpoint0100.pth"
model, _, _ = load_checkpoint(model, optimizer, filename=pretrained_path)
model.eval()

save_folder = os.path.join("results", pretrained_path.split("/")[-1][:-4])
results = evaluate(
    device, model, test_loader, criterion, save_folder, root_relative=True
)

print("Finished.")
