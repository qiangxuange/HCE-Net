


import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction_ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class DetailHoldBlock(nn.Module):
    def __init__(self, in_planes):
        super(DetailHoldBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)

    def forward(self, x):
        attention_weights = self.channel_attention(x)
        return x * attention_weights


class DHN(nn.Module):
    def __init__(self, in_planes, num_detail_hold_blocks):
        super(DHN, self).__init__()
        self.detail_hold_blocks = nn.ModuleList([DetailHoldBlock(in_planes) for _ in range(num_detail_hold_blocks)])

    def forward(self, x):
        for block in self.detail_hold_blocks:
            x = block(x)
        return x


if __name__ == '__main__':
    in_planes = 64
    num_detail_hold_blocks = 3

    dhsnet = DHN(in_planes, num_detail_hold_blocks)
    input_tensor = torch.randn(4, in_planes, 32, 32)
    output_tensor = dhsnet(input_tensor)
    print(output_tensor.shape)  