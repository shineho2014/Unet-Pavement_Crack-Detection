import torch
import torchvision
from dataset import Crack500, Crack500_result
from torch.utils.data import DataLoader


def save_checkpoint(
    state, filename="/content/drive/MyDrive/Unet/my_checkpoint.pth.tar"
):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = Crack500(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = Crack500(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )
    # data set 바꾸면됨
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    total_pixels = 0
    dice_score = 0
    aiu_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            aiu_score += (
                ((preds * y).sum())
                / ((preds + y).sum() - (preds * y).sum())
                / 240
                * 160
            )

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    print(f"Dice score: {dice_score/len(loader)}")
    print(f"AIU score : {aiu_score/len(loader)}")
    model.train()


def save_predictions_as_imgs(
    loader,
    model,
    folder="/content/drive/MyDrive/Unet/data/saved_images/",
    device="cuda",
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        print(f"pred_shape :{preds.shape},val_shape :{y.shape}")

    model.train()