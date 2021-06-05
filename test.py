import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import torchvision
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from utils import load_checkpoint
from dataset import Crack500_result


# Hyperparameters.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
TEST_IMG_DIR = "/content/drive/MyDrive/learning/Unet/data/test_images"
LEARNING_RATE = 1e-4

val_transforms = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ],
)

model = UNET(in_channels=3, out_channels=1).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
load_checkpoint(
    torch.load("/content/drive/MyDrive/learning/Unet/my_checkpoint.pth.tar"),
    model,
    optimizer,
)

dataset = Crack500_result(TEST_IMG_DIR, val_transforms)
loader = DataLoader(dataset, batch_size=2)
loop = tqdm(loader)
for batch_idx, data in enumerate(loop):
    data = data.to(device=DEVICE)
    preds = model.forward(data)
    torchvision.utils.save_image(
        preds,
        f"/content/drive/MyDrive/learning/Unet/result_images/{batch_idx}_pred.png",
    )
