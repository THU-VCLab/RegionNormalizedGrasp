"""Resnet in pytorch.

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.     Deep Residual Learning for Image Recognition
https://arxiv.org/abs/1512.03385v1
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34."""

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=(kernel_size - 1) // 2,
                      bias=False), nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BasicBlock.expansion,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion))

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BasicBlock.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion))

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) +
                                          self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers."""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      stride=stride,
                      kernel_size=kernel_size,
                      padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels * BottleNeck.expansion,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * BottleNeck.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) +
                                          self.shortcut(x))


class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.LeakyReLU(True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2,
                               1,
                               kernel_size,
                               padding=(kernel_size - 1) // 2,
                               bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return torch.sigmoid(x)


# resnet for hggd
class HGGDResNet(nn.Module):

    def __init__(self, block, num_blocks, in_dim=3, planes=64):
        super().__init__()
        # mul 4 for BottleNeck to get the same dim as BasicBlock
        self.in_channels = planes * 4 if block is BottleNeck else planes
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=self.in_channels,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False), nn.BatchNorm2d(self.in_channels),
            nn.LeakyReLU(inplace=True))
        self.conv2_x = self._make_layer(block, planes * 2, num_blocks[0], 2)
        self.conv3_x = self._make_layer(block, planes * 4, num_blocks[1], 2)
        self.conv4_x = self._make_layer(block, planes * 8, num_blocks[2], 2)
        self.conv5_x = self._make_layer(block, planes * 16, num_blocks[3], 2)
        self.embeddings = planes * 16 * block.expansion  # * 1 for resnet<50, * 4 for resnet>=50

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv2_x(o1)
        o3 = self.conv3_x(o2)
        o4 = self.conv4_x(o3)
        o5 = self.conv5_x(o4)
        return [o1, o2, o3, o4, o5]


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 in_dim,
                 num_planes,
                 num_blocks,
                 dedifferential=True):
        super().__init__()
        # mul 4 for BottleNeck to get the same dim as BasicBlock
        self.in_channels = num_planes[
            0] * 4 if block is BottleNeck else num_planes[0]
        self.conv_blocks = nn.ModuleList()
        self.dedifferential = dedifferential and (in_dim > 3)
        if self.dedifferential:
            self.embed_rgb = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=self.in_channels // 2,
                          kernel_size=1,
                          stride=1,
                          bias=False), nn.BatchNorm2d(self.in_channels // 2))
            self.embed_feat = nn.Sequential(
                nn.Conv2d(in_channels=in_dim - 3,
                          out_channels=self.in_channels // 2,
                          kernel_size=1,
                          stride=1,
                          bias=False), nn.BatchNorm2d(self.in_channels // 2))
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.in_channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False), nn.BatchNorm2d(self.in_channels),
                nn.LeakyReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,
                          out_channels=self.in_channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False), nn.BatchNorm2d(self.in_channels),
                nn.LeakyReLU(inplace=True))

        for i in range(len(num_blocks)):
            self.conv_blocks.append(
                self._make_layer(block, num_planes[i], num_blocks[i], 2))

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        if self.dedifferential:
            x = torch.cat(
                [self.embed_rgb(x[:, :3]),
                 self.embed_feat(x[:, 3:])], 1)
            # x = self.stem_rgb(x[:, :3]) + self.stem_feat(x[:, 3:])
            x = F.leaky_relu(x, inplace=True)
        x = self.stem(x)
        for i in range(len(self.conv_blocks)):
            x = self.conv_blocks[i](x)
            outs.append(x)
        return outs


class CoordResNet(nn.Module):

    def __init__(self,
                 block,
                 in_dim,
                 num_planes,
                 num_blocks,
                 dedifferential=True,
                 gated=True,
                 cbam=False):
        super().__init__()
        self.dedifferential = dedifferential
        self.gated = gated
        self.cbam = cbam
        # mul 4 for BottleNeck to get the same dim as BasicBlock
        self.in_channels = num_planes[
            0] * 4 if block is BottleNeck else num_planes[0]
        self.conv_blocks = nn.ModuleList()
        if cbam:
            self.att_blocks = nn.ModuleList()
        if dedifferential:
            self.de_blocks = nn.ModuleList()
            self.de_blocks_coord = nn.ModuleList()
            self.embed_rgb = nn.Sequential(
                nn.Conv2d(in_channels=3,
                          out_channels=self.in_channels // 2,
                          kernel_size=1,
                          stride=1,
                          bias=False), nn.BatchNorm2d(self.in_channels // 2))
            self.embed_feat = nn.Sequential(
                nn.Conv2d(in_channels=in_dim - 3,
                          out_channels=self.in_channels // 2,
                          kernel_size=1,
                          stride=1,
                          bias=False), nn.BatchNorm2d(self.in_channels // 2))
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels,
                          out_channels=self.in_channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False), nn.BatchNorm2d(self.in_channels),
                nn.LeakyReLU(inplace=True))
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels=in_dim,
                          out_channels=self.in_channels,
                          kernel_size=5,
                          stride=1,
                          padding=2,
                          bias=False), nn.BatchNorm2d(self.in_channels),
                nn.LeakyReLU(inplace=True))

        for i in range(len(num_blocks)):
            if dedifferential:
                if gated:
                    self.de_blocks_coord.append(
                        nn.Sequential(
                            nn.Conv2d(3, self.in_channels // 2, 1),
                            nn.BatchNorm2d(self.in_channels // 2),
                            nn.LeakyReLU(True),
                            nn.Conv2d(self.in_channels // 2, self.in_channels,
                                      1), nn.BatchNorm2d(self.in_channels),
                            nn.LeakyReLU(True),
                            nn.Conv2d(self.in_channels, self.in_channels, 1),
                            nn.Sigmoid()))
                else:
                    self.de_blocks.append(
                        nn.Sequential(
                            nn.Conv2d(self.in_channels, self.in_channels, 1),
                            nn.BatchNorm2d(self.in_channels)))
                    self.de_blocks_coord.append(
                        nn.Sequential(nn.Conv2d(3, self.in_channels, 1),
                                      nn.BatchNorm2d(self.in_channels)))
                self.conv_blocks.append(
                    self._make_layer(block, num_planes[i], num_blocks[i], 2))
            else:
                self.in_channels += 3
                self.conv_blocks.append(
                    self._make_layer(block, num_planes[i], num_blocks[i], 2))
            if cbam:
                self.att_blocks.append(
                    nn.ModuleList([
                        ChannelAttention(self.in_channels),
                        SpatialAttention(kernel_size=5)
                    ]))

    def _make_layer(self, block, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        outs = []
        coord = x[:, 3:].clone()
        if self.dedifferential:
            # x = self.embed_rgb(x[:, :3]) + self.embed_faet(coord)
            x = torch.cat([self.embed_rgb(x[:, :3]),
                           self.embed_feat(coord)], 1)
            x = F.leaky_relu(x, inplace=True)
        x = self.stem(x)
        for i in range(len(self.conv_blocks)):
            # dowmsample
            if i > 0:
                coord = F.interpolate(coord, scale_factor=0.5, mode='nearest')
            if self.dedifferential:
                if self.gated:
                    spatial_gate = self.de_blocks_coord[i](coord)
                    x = self.conv_blocks[i](x * spatial_gate)
                else:
                    x = F.leaky_relu(self.de_blocks[i](x) +
                                     self.de_blocks_coord[i](coord),
                                     inplace=True)
                    x = self.conv_blocks[i](x)
            else:
                x = torch.cat([x, coord], 1)
                x = self.conv_blocks[i](x)
            # channel and spatial attention
            if self.cbam:
                x = self.att_blocks[i][0](x) * x
                x = self.att_blocks[i][1](x) * x
            outs.append(x)
        return outs


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    from thop import clever_format, profile
    net = ResNet(BasicBlock,
                 in_dim=6,
                 num_blocks=[2, 2, 2, 2],
                 num_planes=[32, 64, 128, 256])
    net = net.cuda()
    net.eval()

    x = torch.randn((48, 6, 64, 64), device='cuda')
    print(net(x)[-1].shape)
    macs, params = clever_format(profile(net, inputs=(x, )), '%.3f')
    print('macs ==', macs, 'params ==', params)
    net_params = sum(p.numel() for p in net.parameters())
    print(f'net paras == {net_params}')

    from time import time
    torch.cuda.synchronize()
    start = time()
    T = 200
    with torch.no_grad():
        for _ in range(T):
            x = torch.randn((192, 6, 64, 64), device='cuda')
            feat = net(x)
            torch.cuda.synchronize()
    print((time() - start) * 1e3 / T)
