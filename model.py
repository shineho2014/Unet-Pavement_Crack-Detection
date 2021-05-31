# dataset crop할때 unet paper와 다르게 진행  mirroring padding을 변형
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

# Unet convolution 모델 생성
class Doubleconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Doubleconv, self).__init__()
        # Squential 모듈을 Sequential하나에 다 집어 넣고 하나의 입력이 순차적으로 거치도록 함
        # convolution 각 과정에서 2번한다
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,  # out_channels : 필터 개수
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Inplace input들어오면 바로 수정, 수정함으로 Input사라지고 memory 사용은 조금 좋아짐
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)  # sequential로 정의했기 때문에 sequential 한번만 돌리면 됨


# UNET 모델 생성 Contratcion Path & Expansive Path
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        # out channel = 1 binary segmentation 예정 , features = featuremap 두께
        super(UNET, self).__init__()
        # Expansive Path Upsampling 구현
        self.ups = nn.ModuleList()
        # nn.ModuleList() nn.Module을 List안에 집어 넣어 하나씩 이용 가능하도록 함
        # contraction path UNET down paprt구현
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 문제 발생 : 161x161 max pooling 하면 80x80 > upsampling하면 160x160 즉, 사이즈 달라서 skip connection완벽하게 붙일 수 없다.

        # Down part of UNET
        for feature in features:
            self.downs.append(Doubleconv(in_channels, feature))
            # Doubleconv out_channel에 feature들어감, modulelist로 convolution 모델에 값들어간 모듈들 리스트로 저장
            in_channels = feature  # 컨볼루션 두번 pooling 한번 거치면 feature의 값만큼 두께 생성하고 그 다음 down 과정으로 넘어감

        # UP part of UNET
        for feature in reversed(features):
            # convolution 된 이미지를 convtranspose2d로 복원, up sampling 과정
            self.ups.append(
                nn.ConvTranspose2d(
                    in_channels=feature * 2,
                    out_channels=feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            # up된 이미지를 2번 컨볼루션
            self.ups.append(Doubleconv(feature * 2, feature))

        # Contraction & Expansive Path 맨 밑 과정
        self.bottleneck = Doubleconv(features[-1], features[-1] * 2)
        # bottleneck 시작이 512임으로 features 뒤에서 호출 , 1024호출

        # final convolution UNET 알고리즘의 결과를 Binary Segmentation으로 출력
        self.final_conv = nn.Conv2d(
            features[0], out_channels, kernel_size=1
        )  # out_channels default =1

        # unet forward : down part convolution과 pooling 배치

    def forward(self, x):
        skip_connections = []  # 하얀색 화살표
        for down in self.downs:
            x = down(x)  # down part 따라서 컨볼루션 두번
            skip_connections.append(x)  # up part에 colummn 연결 시켜주는 부분(확실하지 않음)
            x = self.pool(x)  # 컨볼루션 두번 후 pooling 하고 다음 step으로

        x = self.bottleneck(x)  # down part 끝나고 맨 밑 up part 넘어가기 직전
        skip_connections = skip_connections[::-1]
        # list reverse up ward에서 skip connection 쓸때는 저화질에서 고화질로 사용됨

        # UP part
        for idx in range(0, len(self.ups), 2):
            # step 2로 해서 double convolution 넘기고 UP 하도록 함
            x = self.ups[idx](x)  # (x) : UP, Double convolution 에 이미지 값 넣음
            skip_connection = skip_connections[idx // 2]  # range step 수 2
            # down part의 skip_connection up part에 붙인다.

            # 문제 일반화 concatenate 이전 사이즈 체크한다. 논문에서는 crop & copy
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
                # down part 거치면서 x가 무조건 더 작다. , skip_connection 채널, 배치 수 무시

            concat_skip = torch.cat((skip_connection, x), dim=1)
            # concatenate , 텐서 1번째 채널을 skip connection으로 연결
            # Tensor 구조 batch size, channel , height width
            x = self.ups[idx + 1](concat_skip)  # double convolution concatenate 텐서 이용해서

        return self.final_conv(x)  # final convolution 으로 마무리


# test
def test():
    x = torch.randn((3, 1, 161, 161))  # batch size, channel,height,width
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape


if __name__ == "__main__":
    test()
