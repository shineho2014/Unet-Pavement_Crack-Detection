# Heart Disease Datathon 2021
> 팀원 : 김진민, 노현빈, 신홍재
## 1. 필요 Package
+ tqdm
+ torch
+ torchvision
+ albumentations
+ matplotlib
+ numpy  
> Google Colab을 사용할 경우 다음의 코드만 실행
```
!pip install --upgrade albumentations
```
## 2. Dataset 분리  
경로 설정을 위해 Image 파일과 Mask 파일을 분리하여 정리  
  
![Image Classification](https://user-images.githubusercontent.com/49667821/144740807-eeafe95e-ccd0-4c74-9aae-473920997a5a.png)
## 3. Hyperparameter 설정
### train.py line 18 ~ line 28
```
'''Hyperparameters'''
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
NUM_WORKERS = 4
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
```
## 4. 경로 설정
### train.py line 29 ~ line 36
```
'''DIRECTORY SETTING '''
SAVE_IMAGE_DIR = "/content/drive/MyDrive/Project/saved_images/"
TRAIN_IMG_DIR = "/content/drive/MyDrive/Project/dataset_main/train/A2C"
TRAIN_MASK_DIR = "/content/drive/MyDrive/Project/dataset_main/train/A2C_mask"
VAL_IMG_DIR = "/content/drive/MyDrive/Project/dataset_main/validation/A2C"
VAL_MASK_DIR = "/content/drive/MyDrive/Project/dataset_main/validation/A2C_mask"
SAVE_MODEL_DIR = "/content/drive/MyDrive/Project/my_checkpoint.pth.tar"
LOAD_MODEL_DIR = "/content/drive/MyDrive/Project/my_checkpoint.pth.tar"
```
+ SAVE_IMAGE_DIR를 통해 학습한 모델이 예측한 좌심방의 위치를 저장할 경로를 설정
+ TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR를 통해 Train dataset, Validation dataset의 경로를 설정
+ SAVE_MODEL_DIR, LOAD_MODEL_DIR를 통해 학습시킨 Model을 저장할 위치, 저장된 Model을 불러올 위치를 설정  
## 5. 실행
train.py Python3로 실행
```
!python3 "{train.py PATH}/train.py"
```

## 6. 결과
![Result_Sample](https://user-images.githubusercontent.com/49667821/144741292-5fa60953-c516-463f-981e-abfb5325c7ac.png)  
  
Dice score: 체적일치도(Dice Similarity Coefficient, DSC)  
JI score: 유사성측도(Jaccard index, JI)  
SAVE_IMAGE_DIR에서 설정한 경로에 학습된 모델이 예측한 좌심방의 Mask가 저장됨  
  
![Prediction_Image](https://user-images.githubusercontent.com/49667821/144741360-39d077cc-44c1-4cc2-9dfd-5ebf5797be36.png)