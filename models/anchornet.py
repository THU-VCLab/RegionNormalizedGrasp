# just copied from hggd repo
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, BottleNeck, HGGDResNet


class Backbone(nn.Module):

    def __init__(self, in_dim, planes=16, mode='50'):
        # if out_dim is None then return features
        super().__init__()
        if mode == '18':
            self.net = HGGDResNet(BasicBlock, [2, 2, 2, 2],
                                  in_dim=in_dim,
                                  planes=planes)
        elif mode == '34':
            self.net = HGGDResNet(BasicBlock, [3, 4, 6, 3],
                                  in_dim=in_dim,
                                  planes=planes)
        elif mode == '50':
            # need to div 4 for same channels
            self.net = HGGDResNet(BottleNeck, [3, 4, 6, 3],
                                  in_dim=in_dim,
                                  planes=planes // 4)
        else:
            raise RuntimeError(f'Backbone mode {mode} not found!')
        logging.info(f'HGGD using {mode} as backbone')

    def forward(self, x):
        x = self.net(x)
        return x


class convolution(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel,
                 bias=False,
                 stride=1,
                 act=nn.ReLU):
        super(convolution, self).__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel, stride, pad, bias=bias),
            nn.BatchNorm2d(out_dim), act(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class trconvolution(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 padding=1,
                 output_padding=0,
                 act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim,
                               out_dim,
                               kernel_size=4,
                               stride=2,
                               padding=padding,
                               output_padding=output_padding,
                               bias=False), nn.BatchNorm2d(out_dim),
            act(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        return x


class upsampleconvolution(nn.Module):

    def __init__(self, in_dim, out_dim, output_size=None, act=nn.ReLU):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2) if output_size is None else
            nn.UpsamplingBilinear2d(size=output_size),
            nn.Conv2d(in_channels=in_dim,
                      out_channels=out_dim,
                      kernel_size=5,
                      padding=2,
                      bias=False), nn.BatchNorm2d(out_dim), act(inplace=True))

    def forward(self, x):
        x = self.deconv(x)
        return x


def conv_with_dim_reshape(in_dim, mid_dim, out_dim, bias=True):
    return nn.Sequential(convolution(in_dim, mid_dim, 5, bias=bias),
                         nn.Conv2d(mid_dim, out_dim, 1, bias=bias))


class AnchorGraspNet(nn.Module):

    def __init__(self,
                 mode='50',
                 ratio=1,
                 in_dim=3,
                 anchor_k=6,
                 mid_dim=32,
                 dropout=0.2,
                 feature_dim=128,
                 use_upsampling=False):
        super(AnchorGraspNet, self).__init__()

        # Using imagenet pre-trained model as feature extractor
        self.ratio = ratio
        self.dropout_p = dropout
        self.trconv = nn.ModuleList()
        # backbone
        self.feature_dim = feature_dim
        self.backbone = Backbone(in_dim, self.feature_dim // 16, mode=mode)
        # dropout
        if dropout > 0:
            self.dropout = nn.Dropout2d(self.dropout_p)

        # transconv
        self.depth = 4
        channels = [
            max(8, self.feature_dim // (2**(i + 1)))
            for i in range(self.depth + 1)
        ]
        cur_dim = self.feature_dim
        output_sizes = [(40, 23), (80, 45)]
        for i, dim in enumerate(channels):
            if use_upsampling:
                if i < min(5 - np.log2(ratio), 2):
                    self.trconv.append(
                        upsampleconvolution(cur_dim, dim, output_sizes[i]))
                else:
                    self.trconv.append(upsampleconvolution(cur_dim, dim))
            else:
                if i < min(5 - np.log2(ratio), 2):
                    self.trconv.append(
                        trconvolution(cur_dim,
                                      dim,
                                      padding=(1, 2),
                                      output_padding=(0, 1)))
                else:
                    self.trconv.append(trconvolution(cur_dim, dim))
            cur_dim = dim

        # Heatmap predictor
        cur_dim = channels[self.depth - int(np.log2(self.ratio))]
        self.hmap = conv_with_dim_reshape(channels[-1], mid_dim, 1)
        self.cls_mask_conv = conv_with_dim_reshape(cur_dim, mid_dim, anchor_k)
        self.theta_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)
        self.width_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)
        self.depth_offset_conv = conv_with_dim_reshape(cur_dim, mid_dim,
                                                       anchor_k)

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # use backbone to get downscaled features
        xs = self.backbone(x)
        # use transposeconve or upsampling + conv to get perpoint features
        # using dropout here for encoder
        if self.dropout_p > 0:
            xs[-1] = self.dropout(xs[-1])
        x = xs[-1]
        for i, layer in enumerate(self.trconv):
            # skip connection
            x = layer(x + xs[self.depth - i])
            # down sample classification mask
            if x.shape[2] == 80:
                features = x
            if int(np.log2(self.ratio)) == self.depth - i:
                if self.dropout_p > 0:
                    x = self.dropout(x)
                # use dropout features
                cls_mask = self.cls_mask_conv(x)
                theta_offset = self.theta_offset_conv(x)
                width_offset = self.width_offset_conv(x)
                depth_offset = self.depth_offset_conv(x)
        # full scale location map
        loc_map = self.hmap(x)
        return (loc_map, cls_mask, theta_offset, depth_offset,
                width_offset), features.detach()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    from thop import clever_format, profile
    net = AnchorGraspNet(mode='50',
                         ratio=8,
                         in_dim=4,
                         anchor_k=6,
                         mid_dim=32,
                         dropout=0.0,
                         feature_dim=128)
    net = net.cuda()
    net.eval()
    # checkpoint = torch.load('./logs/checkpoint/graspnet_hggd_realsense')
    # net.load_state_dict(checkpoint['anchor'])

    x = torch.randn((1, 4, 640, 360), device='cuda')
    print(net(x)[-1].shape)
    macs, params = clever_format(profile(net, inputs=(x, )), '%.3f')
    print('macs ==', macs, 'params ==', params)
    net_params = sum(p.numel() for p in net.parameters())
    print(f'net paras == {net_params}')

    from time import time
    torch.cuda.synchronize()
    T = 200
    total_time = 0
    with torch.no_grad():
        for _ in range(T):
            x = torch.randn((1, 4, 640, 360), device='cuda')
            start = time()
            feat = net(x)
            torch.cuda.synchronize()
            total_time += (time() - start)
            torch.cuda.empty_cache()
    print(total_time * 1e3 / T)
