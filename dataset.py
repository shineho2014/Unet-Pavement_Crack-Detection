import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)  # 디렉토리 안 파일명들 리스트로 만듦

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(
            self.image_dir, self.images[index]
        )  # 폴더 경로 + image 파일 명
        mask_path = os.path.join(
            self.mask_dir, self.images[index].replace(".jpg", "_mask.gif")
        )  # mask는 파일명 .jpg가 아니라 _mask.gif
        image = np.array(Image.open(img_path).convert("RGB"))  # PIL쓰려면 numpy array 필요
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        # mask는 Gray scale로 불러온다 "L" 사용

        # 마스크 이미지 라벨 설정 (0.0,255.0)
        mask[mask == 255.0] = 1.0
        # 마스크 흰색이면 1.0으로 correct label로 sigmoid activation에 입력
        # augmentation 안되있으면 argumentation 실행
        if self.transform is not None:  # None is Null
            augmentation = self.transform(image=image, mask=mask)
