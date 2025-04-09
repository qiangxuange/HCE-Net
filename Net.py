import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torchvision.transforms.functional as TF
from DCT import DHN
from IAFF import iAFF


class encoderx(nn.Module):  # 卷积类
    def __init__(self, in_channels, out_channels):  # 固定方法
        super(encoderx, self).__init__()  # 继承DoubleConv类
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            DHN(out_channels, 3),

            nn.Conv2d(out_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            DHN(out_channels, 3),

            nn.Conv2d(out_channels, out_channels, 7, 1, 3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.r = nn.Conv2d(in_channels, out_channels, 1)
    #网络推进
    def forward(self, x):
        x = self.conv(x) + self.r(x)  # 残差
        return x


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )
    def forward(self,x):
        x = self.conv(x)
        return x

def sobel_func(x, device):
    # 定义Sobel算子
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    # 定义Sobel算子
    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)
    kernel_x = torch.FloatTensor(sobel_x).expand(len(x[0]), len(x[0]), 3, 3)
    weight_x = torch.nn.Parameter(data=kernel_x, requires_grad=False).to(device=device)
    kernel_y = torch.FloatTensor(sobel_y).expand(len(x[0]), len(x[0]), 3, 3)
    weight_y = torch.nn.Parameter(data=kernel_y, requires_grad=False).to(device=device)
    sx = torch.nn.functional.conv2d(x, weight_x, padding=1)
    sy = torch.nn.functional.conv2d(x, weight_y, padding=1)
    x = torch.sqrt(sx ** 2 + sy ** 2 + 1e-6)
    bn = nn.BatchNorm2d(x.shape[1], momentum=0.1).to(device=device)
    x = bn(x)
    x = F.relu(x, inplace=True)
    return x


class encodery(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encodery, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 5, 1, 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.r = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        x = x.to(self.device)
        x1 = sobel_func(x, self.device)
        x2 = sobel_func(x, self.device)  
        x1 = self.conv2(self.conv4(x1) + self.conv(x1))
        x2 = self.conv2(self.conv4(x2) + self.conv1(x2))
        out = self.conv3(x1 + x2)
        return out + self.r(x)
        
class HCE(nn.Module):#Net主体
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(NBnet, self).__init__()
        #声明list用于上采样和下采样存储
        self.ups = nn.ModuleList()
        self.downs1 = nn.ModuleList()
        self.downs2 = nn.ModuleList()
        self.iaff = nn.ModuleList()
        #池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            #下采样
            self.downs1.append(encoderx(in_channels, feature))  # 第一条编码器
            self.downs2.append(encodery(in_channels, feature))  # 第二条编码器
            in_channels = feature

        for feature in reversed(features):
            #上采样--包括一个卷积和一个转置卷积
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(encoderx(feature * 2, feature))
            self.iaff.append(iAFF(feature))
        #unet网络底层卷积
        self.bottleneck = encoder(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.convv = nn.Conv2d(1024, 512, kernel_size=1)

    def forward(self, x):
        x1 = x
        x2 = x
        skip_connections1, skip_connections2 = [], []

        for down in range(0, len(self.downs1)):
            #对x进行下采样
            x1 = self.downs1[down](x1)
            x2 = self.downs2[down](x2)
            #将此处状态加入跳跃连接list
            x1 = x1 + x2
            x2 = x1 + x2
            skip_connections1.append(x1)
            skip_connections2.append(x2)
            #进行池化操作
            x1 = self.pool(x1)
            x2 = self.pool(x2)

        x = self.bottleneck(self.convv(torch.cat([x1, x2], dim = 1)))
        #因为上采样是自下而上，所以反转当前列表
        skip_connections1 = skip_connections1[::-1]
        skip_connections2 = skip_connections2[::-1]
        temp_x = x
        for i in range(0, len(self.ups), 2):
            #先进行转置卷积
            x = self.ups[i](x)
            encoder1 = skip_connections1[i // 2]
            encoder2 = skip_connections2[i // 2]
            combined = self.iaff[i // 2](encoder1, encoder2)
            concat_skip = torch.cat([combined, x], dim=1)
            #粘贴后的两个特征在进行一次卷积操作
            x = self.ups[i + 1](concat_skip) + self.ups[i](temp_x)
            temp_x = x

        return self.final_conv(x)  # 最后的1*1卷积操作


if __name__ == '__main__':
    x = torch.randn(4, 3, 256, 256).to(device)
    model = HCE(in_channels=3, out_channels=1).to(device)
    #将x传入模型
    preds = model(x)
    print(preds.shape)

