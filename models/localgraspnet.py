import logging
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import BasicBlock, BottleNeck, ResNet, CoordResNet

eps = 1e-6


class LinearLNReLU(nn.Sequential):

    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.append(nn.Linear(in_dim, out_dim))
        self.append(nn.LayerNorm(out_dim))
        self.append(nn.ReLU(inplace=True))


class ResMLP(nn.Module):

    def __init__(self, in_dim, out_dim, expand=0.25) -> None:
        super().__init__()
        neck_dim = int(expand * out_dim)
        self.net = nn.Sequential(LinearLNReLU(in_dim, neck_dim),
                                 nn.Linear(neck_dim, out_dim),
                                 nn.LayerNorm(out_dim))

    def forward(self, x1, x2):
        return F.relu(x1 + self.net(x2), True)


@torch.no_grad()
def get_grasp_infos(theta_cls, theta_offset, width_reg):
    # get anchor position
    anchors = theta_cls.sigmoid().clip(eps, 1 - eps).max(1)[1]  # (B, 1)
    anchors = anchors.unsqueeze(1)
    da = torch.gather(theta_offset.clip(-0.5, 0.5), dim=1, index=anchors)
    # get normalized w
    w = torch.gather(width_reg, dim=1, index=anchors)
    # get theta
    theta_range = torch.pi
    anchor_step = theta_range / theta_cls.shape[1]
    theta = anchor_step * (anchors + da + 0.5) - theta_range / 2
    return torch.cat([theta, w], 1)


class PatchMultiGraspNet(nn.Module):

    def __init__(self,
                 k_cls,
                 theta_k_cls=6,
                 feat_dim=256,
                 anchor_w=60.0,
                 dropout_p=0.2,
                 model_type='small'):
        super().__init__()
        self.anchor_w = anchor_w
        self.dropout_p = dropout_p
        self.theta_k_cls = theta_k_cls
        self.k_cls = k_cls  # anchor_num**2

        self.patch_dim = 6
        logging.info(f'Patchnet with dim == {self.patch_dim}')
        # feature extractor
        logging.info(f'Using {model_type}')
        num_planes = [feat_dim // 2**(3 - i) for i in range(4)]
        blocks_dict = {
            'tiny': BasicBlock,
            'small': BasicBlock,
            'small-bottleneck': BottleNeck,
            'base': BottleNeck,
            'large': BottleNeck
        }
        num_blocks_dict = {
            'tiny': [1, 1, 1, 1],
            'small': [2, 2, 2, 2],
            'small-bottleneck': [2, 2, 2, 2],
            'base': [3, 3, 9, 3],
            'large': [3, 3, 27, 3]
        }
        num_blocks = num_blocks_dict[model_type]
        channels_str = ', '.join(list(map(str, num_planes)))
        logging.info(f'Patchnet channels: ({channels_str})')
        blocks_str = ', '.join(list(map(str, num_blocks)))
        logging.info(f'Patchnet blocks: ({blocks_str})')
        self.backbone = CoordResNet(blocks_dict[model_type],
                                    in_dim=self.patch_dim,
                                    num_planes=num_planes,
                                    num_blocks=num_blocks)
        if blocks_dict[model_type] is BottleNeck:
            feat_dim *= 4

        # feature pooling
        self.pool_layer = nn.AdaptiveAvgPool2d(1)

        info_dim, neck_dim = 2, 256
        # heads
        self.theta_cls = nn.Sequential(LinearLNReLU(feat_dim, neck_dim),
                                       nn.Dropout(dropout_p),
                                       nn.Linear(neck_dim, theta_k_cls))
        self.theta_offset = nn.Sequential(LinearLNReLU(feat_dim, neck_dim),
                                          nn.Dropout(dropout_p),
                                          nn.Linear(neck_dim, theta_k_cls))

        self.width_reg = nn.Sequential(LinearLNReLU(feat_dim, neck_dim),
                                       nn.Dropout(dropout_p),
                                       nn.Linear(neck_dim, theta_k_cls))
        self.info_layer = LinearLNReLU(info_dim, 32)
        self.feat_layer = ResMLP(feat_dim + 32, feat_dim)
        self.anchor_cls = nn.Sequential(LinearLNReLU(feat_dim, neck_dim),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(neck_dim, k_cls))
        self.offset_reg = nn.Sequential(LinearLNReLU(feat_dim, neck_dim),
                                        nn.Dropout(dropout_p),
                                        nn.Linear(neck_dim, k_cls * 3))

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

    def forward(self, patches):
        patches = patches.permute(0, 3, 1, 2).contiguous()
        # get features
        features = self.backbone(patches)
        feat_map = features[-1]

        # pooling features
        feat_map = self.pool_layer(feat_map)
        f = feat_map.flatten(1)

        # get theta, theta_offset and width
        theta_cls = self.theta_cls(f)
        theta_offset = self.theta_offset(f)
        width_reg = self.width_reg(f)

        info = get_grasp_infos(theta_cls, theta_offset, width_reg)
        info = self.info_layer(info)
        f = self.feat_layer(f, torch.cat([f, info], 1))
        pred = self.anchor_cls(f)
        offset = self.offset_reg(f).view(-1, self.k_cls, 3)
        return features, pred, offset, theta_cls, theta_offset, width_reg


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    net = PatchMultiGraspNet(k_cls=49,
                             feat_dim=256,
                             anchor_w=60,
                             dropout_p=0.2,
                             model_type='small')
    net = net.cuda()
    net.eval()
    input_data = torch.randn((48, 64, 64, 6),
                             dtype=torch.float32,
                             device='cuda')
    net(input_data)

    T = 100
    total_time = 0
    with torch.no_grad():
        for i in range(T):
            start = time()
            a = net(input_data)
            torch.cuda.synchronize()
            total_time += time() - start
            torch.cuda.empty_cache()
    print(total_time * 1e3 / T)
