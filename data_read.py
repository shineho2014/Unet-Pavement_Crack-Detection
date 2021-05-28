# 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Data Load
dir_data = "./datasets"
name_label = "train-lables.tif"
name_input = "train-volume.tif"

img_label = Image.open(
    os.path.join("/Users/nohyeonbin/Documents/study/Unet/datasets/train-labels.tif")
)
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

# data 셋 분리
nframe_train = 24
nframe_val = 3
nframe_test = 3
# 디렉토리 정의
dir_save_train = os.path.join(dir_data, "train")
dir_save_val = os.path.join(dir_data, "val")
dir_save_test = os.path.join(dir_data, "test")
# train, validation, test 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)
# test,train,validation set 무작위로 저장
# frame 별 랜덤하게 저장
id_frame = np.arange(nframe)  # 0~29(nframe)까지 배열 생성
np.random.shuffle(id_frame)  # 배열 (index) 무작위로 배치
# train data set 저장
offset_nframe = 0
for i in range(nframe_train):
    img_label.seek(id_frame)