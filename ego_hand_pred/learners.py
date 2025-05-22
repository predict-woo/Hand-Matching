import torch
from tqdm import tqdm
import os
from .visualizer import vis_keypoint


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}!")


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Checkpoint loaded from epoch {epoch}")
    return model, optimizer, epoch


def train(
    device, epoch, epochs, model, dataloader, criterion, optimizer, root_relative=False
):
    model.train()
    checkpoint_interval = 5
    total_loss = 0
    for images, targets in tqdm(dataloader, desc=f"Train - epoch {epoch}"):
        images = images.to(device)
        if root_relative:
            targets_pose = targets[0].to(device)
            targets_root = targets[1].to(device)
            optimizer.zero_grad()
            preds_pose, preds_root = model(images)
            loss_pose = criterion(preds_pose, targets_pose)
            loss_root = criterion(preds_root, targets_root)
            loss = loss_pose + 10000 * loss_root
        else:
            targets = targets.to(device)
            optimizer.zero_grad()
            preds_pose = model(images)
            loss = criterion(preds_pose, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch [{epoch}/{epochs}], Train Loss: {total_loss/len(dataloader):.4f}")
    if (epoch + 1) % checkpoint_interval == 0:
        save_path = os.path.join(
            model.ckpt_folder, f"checkpoint{str(epoch+1).zfill(4)}.pth"
        )
        save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            total_loss / len(dataloader),
            filename=save_path,
        )


def validate(device, model, dataloader, criterion, root_relative=False):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            if root_relative:
                targets_pose = targets[0].to(device)
                targets_root = targets[1].to(device)
                preds_pose, preds_root = model(images)
                loss_pose = criterion(preds_pose, targets_pose)
                loss_root = criterion(preds_root, targets_root)
                loss = loss_pose + loss_root
            else:
                targets = targets.to(device)
                preds_pose = model(images)
                loss = criterion(preds_pose, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(device, model, test_loader, criterion, save_folder, root_relative=False):
    total_loss = 0
    with torch.no_grad():
        for images, targets, image_paths in tqdm(test_loader, desc="Test"):
            images = images.to(device)
            if root_relative:
                targets_pose = targets[0].to(device)
                targets_root = targets[1].to(device)
                targets_pose[:, :21, :] += targets_root[:, :1, :]
                targets_pose[:, 21:, :] += targets_root[:, 1:, :]
                preds_pose, preds_root = model(images)
                preds_pose[:, :21, :] += preds_root[:, :1, :]
                preds_pose[:, 21:, :] += preds_root[:, 1:, :]
                loss_pose = criterion(preds_pose, targets_pose)
                loss_root = criterion(preds_root, targets_root)
                loss = loss_pose + loss_root
            else:
                targets_pose = targets.to(device)
                preds_pose = model(images)
                loss = criterion(preds_pose, targets_pose)
            total_loss += loss.item()
            vis_keypoint(image_paths, preds_pose, targets_pose, save_folder)
    avg_loss = total_loss / len(test_loader)
    print(f"Evaluation Result: {avg_loss}")
    return avg_loss
