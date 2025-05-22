import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import HandPoseEstimator
from datasets import HandPoseDataset
from learners import train, validate, load_checkpoint


# DataLoader(
#     train_dataset,
#     batch_size=8,
#     shuffle=True,
#     num_workers=8,  # see below
#     pin_memory=True,
#     persistent_workers=True,  # keeps workers alive between epochs
#     prefetch_factor=2,  # how many batches each worker prefetches
# )


train_dataset = HandPoseDataset(mode="train", input_size=224, root_relative=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_dataset = HandPoseDataset(mode="val", input_size=224, root_relative=True)
val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HandPoseEstimator("vit_224", root_relative=True).to(device)
model = torch.compile(model)
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

# 1500 / 60 = 25
# 1500 / 120 = 12.5
