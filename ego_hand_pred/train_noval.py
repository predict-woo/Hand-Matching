import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import HandPoseEstimator
from datasets import HandPoseDataset
from learners import train, validate, load_checkpoint


train_dataset = HandPoseDataset(mode="train")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

val_dataset = HandPoseDataset(mode="val")
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandPoseEstimator("vit_224", root_relative=True).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

pretrained_path = ""
if pretrained_path:
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, filename=pretrained_path
    )
else:
    start_epoch = 0

epochs = 100
for epoch in range(start_epoch, epochs):
    train(
        device,
        epoch,
        epochs,
        model,
        train_loader,
        criterion,
        optimizer,
        root_relative=True,
    )
    validate(device, model, val_loader, criterion, root_relative=True)

print("Finished.")
