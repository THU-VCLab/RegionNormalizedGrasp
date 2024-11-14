from functools import wraps
from time import time
from typing import List

import random
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import rotate
from matplotlib import pyplot as plt
from numba import njit
from pytorch3d.ops import ball_query, knn_points, sample_farthest_points
from pytorch3d.ops.utils import masked_gather
from scipy import interpolate
from skimage.feature import peak_local_max

from .config import get_camera_intrinsic
from .grasp import RectGraspGroup
from .utils import (angle_distance, convert_2d_to_3d, fast_sample,
                    square_distance)


def timing(f):

    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'func:{f.__name__} took: {te-ts} sec')
        return result

    return wrap


@torch.jit.script
def get_local_patches(centers, rgbd, radius, size: int):
    patches = torch.zeros((len(centers), size, size, rgbd.shape[-1]),
                          device='cuda',
                          dtype=torch.float32)
    for i in range(len(centers)):
        x, y = centers[i, 0], centers[i, 1]
        cur_patch = rgbd[x:x + radius[i] * 2, y:y + radius[i] * 2]
        patches[i] = F.interpolate(cur_patch.permute(2, 0, 1)[None],
                                   (size, size),
                                   mode='nearest').squeeze().permute(1, 2, 0)
        # plt.subplot(131)
        # plt.imshow(patches[i, ..., :3].cpu().numpy())
        # plt.subplot(132)
        # plt.imshow(patches[i, ..., 3].cpu().numpy())
        # plt.subplot(133)
        # plt.imshow(patches[i, ..., 4].cpu().numpy())
        # plt.show()
        # plt.savefig('local_patch.png', dpi=400)
        # from time import sleep
        # sleep(1)
    return patches


def get_group_pc(
    pc: torch.Tensor,
    local_centers: List,
    group_num,
    rect_ggs,
    grasp_widths=None,
    min_points=32,
    is_training=True,
    mode='fps',
):
    batch_size, feature_len = pc.shape[0], pc.shape[2]
    pc_group = torch.zeros((0, group_num, feature_len),
                           dtype=torch.float32,
                           device='cuda')
    pc_group_ori = []
    # get grasp width for pc segmentation

    if grasp_widths is None:
        grasp_widths_list = []
        for rect_gg in rect_ggs:
            grasp_widths_list.append(rect_gg.get_6d_width())
    else:
        grasp_widths_list = [grasp_widths]
    # print(grasp_widths[0].mean())
    valid_local_centers = []
    valid_center_masks = []
    # get the points around one scored center
    if mode == 'fps':
        for i in range(batch_size):  # batch_size=1 for generating local data
            # deal with empty input
            if len(local_centers[i]) == 0:
                # no need to append pc_group
                lengths = None
                valid_local_centers.append(local_centers[i])
                valid_center_masks.append(
                    torch.ones((0, ), dtype=torch.bool, device='cuda'))
                continue
            # cal distance and get masks (for all centers)
            dis = square_distance(local_centers[i], pc[i])
            # using grasp width for ball segment
            grasp_widths_tensor = torch.from_numpy(grasp_widths_list[i]).to(
                device='cuda', dtype=torch.float32)[..., None]
            # add noise when trainning
            width_scale = 1
            if is_training:
                # 0.8 ~ 1.2
                width_scale = 0.8 + 0.4 * torch.rand(
                    (len(grasp_widths_tensor), 1), device='cuda')
            masks = (dis < (grasp_widths_tensor * width_scale)**2)
            # select valid center from all center
            center_cnt = len(local_centers[i])
            valid_mask = torch.ones((center_cnt, ),
                                    dtype=torch.bool,
                                    device='cuda')
            # concat pc first
            max_pc_cnt = max(group_num, masks.sum(1).max())
            partial_pcs = torch.zeros((center_cnt, max_pc_cnt, feature_len),
                                      device='cuda')
            lengths = torch.zeros((center_cnt, ), device='cuda')
            for j in range(center_cnt):
                # seg points
                partial_points = pc[i, masks[j]]
                point_cnt = partial_points.shape[0]
                if point_cnt < group_num:
                    if point_cnt >= min_points:
                        if point_cnt == 0:
                            partial_points = torch.zeros(
                                (group_num, feature_len))
                            point_cnt = group_num
                        else:
                            idxs = torch.randint(point_cnt, (group_num, ),
                                                 device='cuda')
                            # idxs = np.random.choice(point_cnt, group_num, replace=True)
                            partial_points = partial_points[idxs]
                            point_cnt = group_num
                    else:
                        valid_mask[j] = False
                        lengths[j] = group_num
                        continue
                partial_pcs[j, :point_cnt] = partial_points
                lengths[j] = point_cnt
            # add a little noise to avoid repeated points
            partial_pcs[..., :3] += torch.randn(partial_pcs.shape[:-1] + (3, ),
                                                device='cuda') * 5e-4
            # get ori pc group
            partial_pcs_ori = partial_pcs[valid_mask]
            partial_pcs_ori[..., :3] = partial_pcs_ori[
                ..., :3] - local_centers[i][valid_mask][:, None]
            pc_group_ori.append(partial_pcs_ori)

            # doing fps
            _, idxs = sample_farthest_points(partial_pcs[..., :3],
                                             lengths=lengths,
                                             K=group_num,
                                             random_start_point=True)
            # mv center of pc to (0, 0, 0), stack to pc_group
            temp_idxs = idxs[..., None].repeat(1, 1, feature_len)
            cur_pc = torch.gather(partial_pcs, 1, temp_idxs)
            cur_pc = cur_pc[valid_mask]
            cur_pc[..., :3] = cur_pc[
                ..., :3] - local_centers[i][valid_mask][:, None]
            pc_group = torch.concat([pc_group, cur_pc], 0)
            # stack pc and get valid center list
            valid_local_centers.append(local_centers[i][valid_mask])
            valid_center_masks.append(valid_mask)

            lengths = lengths[valid_mask]

        return pc_group, valid_local_centers, valid_center_masks, pc_group_ori, lengths
    elif mode == 'bq':
        local_centers_tensor = torch.stack(local_centers).to(
            device='cuda', dtype=torch.float32)
        _, idxs, _ = ball_query(local_centers_tensor,
                                pc[..., :3],
                                K=group_num,
                                radius=0.2,
                                return_nn=False)
        pc_group = masked_gather(pc, idxs).view(
            batch_size * local_centers_tensor.shape[1], group_num, feature_len)
        valid_center_masks = []
        for i in range(batch_size):
            valid_center_masks.append(
                torch.ones((len(local_centers[i]), ), dtype=torch.bool).cuda())
        return pc_group, local_centers, valid_center_masks
    else:
        raise RuntimeError(f'mode {mode} not found')


def center2dtopc(rect_ggs: List,
                 center_num,
                 depths: torch.Tensor,
                 output_size,
                 append_random_center=True,
                 is_training=True):
    # add extra axis when valid, avoid dim errors
    batch_size = depths.shape[0]
    center_batch_pc = []

    scale_x, scale_y = 1280 / output_size[0], 720 / output_size[1]
    for i in range(batch_size):  # batch_size=1 for generating local data
        center_2d = rect_ggs[i].centers.copy()
        # add random center when local max count not enough
        if append_random_center and len(center_2d) < center_num:
            # print(f'current center_2d == {len(center_2d)}. using random center')
            random_local_max = np.random.rand(center_num - len(center_2d), 2)
            random_local_max = np.vstack([
                (random_local_max[:, 0] * output_size[0]).astype(np.int32),
                (random_local_max[:, 1] * output_size[1]).astype(np.int32),
            ]).T
            center_2d = np.vstack([center_2d, random_local_max])

        # scale
        center_2d[:, 0] = center_2d[:, 0] * scale_x
        center_2d[:, 1] = center_2d[:, 1] * scale_y
        # mask d != 0
        d = depths[i, center_2d[:, 0], center_2d[:, 1]].float()
        mask = (d != 0)
        # convert
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        center_tensor = torch.from_numpy(center_2d).to(dtype=torch.float32,
                                                       device='cuda')
        # add delta depth
        delta_d = torch.from_numpy(rect_ggs[i].depths).cuda()
        z = (d + delta_d) / 1000.0
        x = z / fx * (center_tensor[:, 0] - cx)
        y = z / fy * (center_tensor[:, 1] - cy)
        pc_tensor = torch.vstack([x, y, z]).T
        # deal with d == 0
        idxs = torch.nonzero(~mask).cpu().numpy().squeeze(-1)
        for j in idxs:
            x, y = center_2d[j, 0], center_2d[j, 1]
            # choose neighbor average to fix zero depth
            neighbor = 8
            x_range = slice(max(0, x - neighbor), min(1279, x + neighbor))
            y_range = slice(max(0, y - neighbor), min(719, y + neighbor))
            neighbor_depths = depths[i, x_range, y_range]
            depth_mask = (neighbor_depths > 0)
            if not depth_mask.any():
                # continue
                # this will use all centers
                # cur_d = depths[i].mean()
                cur_d = d.mean()
            else:
                cur_d = neighbor_depths[depth_mask].float().median(
                ) + delta_d[j]
            # set valid mask
            mask[j] = True
            # convert
            pc_tensor[j] = torch.from_numpy(convert_2d_to_3d(
                x, y, cur_d.cpu())).cuda()
        # modify rect_ggs and append
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]
        # convert delta depth to actual depth for further width conversion
        rect_ggs[i].actual_depths = pc_tensor[:, 2].cpu().numpy() * 1000.0
        # attention: rescale here
        rect_ggs[i].actual_depths *= 1280 // output_size[0]
        # add small noise to local centers (when train)
        if is_training:
            pc_tensor += torch.randn(*pc_tensor.shape, device='cuda') * 5e-3
        center_batch_pc.append(pc_tensor)
    return center_batch_pc, mask.cpu().numpy()


@torch.no_grad()
def process_local_data(data_list,
                       group_num,
                       center_num,
                       mode='fps',
                       thres=-1.0,
                       min_width=0.01,
                       max_width=0.10,
                       sample_fullpc=True,
                       aug_scale=False,
                       aug_theta=False):
    # get local data
    all_partial_pcs = None
    all_full_pcs = None
    all_centers = None
    all_patches = None
    total_labels = None
    gg_labels = []

    for data in data_list:
        cur_patches = data['patches'] if 'patches' in data else None
        cur_centers = data['centers'] if 'centers' in data else None
        if 'pcs' in data:
            cur_partial_pcs = data['pcs'][..., :3]
            cur_full_pcs = data['full_pcs'][..., :3]
        else:
            cur_partial_pcs = None
            cur_full_pcs = None
        cur_grasps = data['grasps']

        # permutate the order of points in a local region
        # permutation_idx = np.random.rand(cur_full_pcs.shape[1]).argsort()
        # cur_partial_pcs = np.take(cur_partial_pcs, permutation_idx, axis=1)
        # cur_full_pcs = np.take(cur_full_pcs, permutation_idx, axis=1)

        if center_num < len(cur_grasps):
            idx = np.random.choice(range(cur_partial_pcs.shape[0]),
                                   size=center_num,
                                   replace=False)
            if cur_patches is not None:
                cur_patches = torch.from_numpy(cur_patches[idx])
            if cur_centers is not None:
                cur_centers = torch.from_numpy(cur_centers[idx])
            if cur_partial_pcs is not None:
                cur_partial_pcs = torch.from_numpy(cur_partial_pcs[idx])
            if cur_full_pcs is not None:
                cur_full_pcs = torch.from_numpy(cur_full_pcs[idx])
            cur_grasps = cur_grasps[idx]

        else:
            if cur_patches is not None:
                cur_patches = torch.from_numpy(cur_patches)
            if cur_centers is not None:
                cur_centers = torch.from_numpy(cur_centers)
            cur_partial_pcs = torch.from_numpy(cur_partial_pcs)
            cur_full_pcs = torch.from_numpy(cur_full_pcs)

        # cur_partial_pcs += torch.randn(cur_partial_pcs.shape[:-1] + (3, ),
        #                                         device='cuda') * 5e-3

        # filter for some data error
        if cur_full_pcs is not None:
            if thres > 0:
                mask = (cur_full_pcs.max(1)[0].max(1)[0] < thres)
                mask &= (cur_full_pcs.min(1)[0].min(1)[0] > -thres)
                # if not mask.all():
                #     print(f'found error data: {mask.sum()} / {len(mask)}')
                mask = mask.cpu()
            else:
                mask = torch.ones((len(cur_centers), ), dtype=torch.bool)

        if all_partial_pcs is None:
            if cur_centers is not None:
                all_centers = cur_centers[mask].clone()
            if cur_patches is not None:
                all_patches = cur_patches[mask].clone()
            if cur_partial_pcs is not None:
                all_partial_pcs = cur_partial_pcs[mask].clone()
            if cur_full_pcs is not None:
                all_full_pcs = cur_full_pcs[mask].clone()
            cur_grasps_array = np.concatenate(cur_grasps[mask], 0)
            total_labels = torch.from_numpy(cur_grasps_array)
        else:
            if cur_centers is not None:
                all_centers = torch.cat([all_centers, cur_centers[mask]], 0)
            if cur_patches is not None:
                all_patches = torch.cat([all_patches, cur_patches[mask]], 0)
            if cur_partial_pcs is not None:
                all_partial_pcs = torch.concat(
                    [all_partial_pcs, cur_partial_pcs[mask]], 0)
            if cur_full_pcs is not None:
                all_full_pcs = torch.concat([all_full_pcs, cur_full_pcs[mask]],
                                            0)
            cur_grasps_array = np.concatenate(cur_grasps[mask], 0)
            total_labels = torch.cat(
                [total_labels,
                 torch.from_numpy(cur_grasps_array)], 0)

        gg_labels.extend(cur_grasps[mask])

    # mask grasp with max width
    for i in range(len(gg_labels)):
        if len(gg_labels[i]) > 0:
            width_mask = (gg_labels[i][:, -1]
                          > min_width) & (gg_labels[i][:, -1] < max_width)
            gg_labels[i] = gg_labels[i][width_mask]

    # fps if necessary
    if all_partial_pcs is not None:
        if all_partial_pcs.shape[1] > group_num:
            if mode == 'fps':
                all_partial_pcs, _ = sample_farthest_points(
                    all_partial_pcs, K=group_num, random_start_point=True)
                if sample_fullpc:
                    all_full_pcs, _ = sample_farthest_points(
                        all_full_pcs, K=group_num, random_start_point=True)
            elif mode == 'random':
                idxs = fast_sample(all_partial_pcs.shape[1], group_num)
                all_partial_pcs = all_partial_pcs[:, idxs]
                if sample_fullpc:
                    all_full_pcs = all_full_pcs[:, idxs]
            else:
                raise NotImplementedError(f'Sample mode {mode} not found!')

    all_centers = all_centers.to(
        'cuda', torch.float32) if all_centers is not None else None
    all_patches = all_patches.to(
        'cuda', torch.float32) if all_patches is not None else None
    if cur_partial_pcs is not None:
        all_partial_pcs = all_partial_pcs.to('cuda', torch.float32)
    if cur_full_pcs is not None:
        all_full_pcs = all_full_pcs.to('cuda', torch.float32)
    total_labels = total_labels.to('cuda', torch.float32)

    # data augmentation
    if aug_theta:
        # random rotate
        delta_thetas = torch.pi / 2 * torch.rand(len(gg_labels), device='cuda')
        # delta_thetas = torch.pi * torch.full(
        #     (len(gg_labels), ), 0.25, device='cuda')
        # rotate patches
        s = torch.sin(delta_thetas)
        c = torch.cos(delta_thetas)
        rot_mat = torch.stack(
            [torch.stack([c, -s], dim=1),
             torch.stack([s, c], dim=1)], dim=1)
        zeros = torch.zeros((rot_mat.shape[0], 2, 1), device='cuda')
        aff_mat = torch.cat([rot_mat, zeros], 2)
        # not rotate xy meshgrid
        patch_rgbds = all_patches[..., :5].permute(0, 3, 1, 2)
        grid = F.affine_grid(aff_mat, patch_rgbds.shape, align_corners=False)
        all_patches[..., :5] = F.grid_sample(patch_rgbds,
                                             grid,
                                             align_corners=False,
                                             mode='nearest').permute(
                                                 0, 2, 3, 1)
        # modify labels
        delta_thetas = delta_thetas.cpu().numpy()
        rot_mat = rot_mat.cpu().numpy().squeeze()
        for i in range(len(gg_labels)):
            gg_label = gg_labels[i]
            if len(gg_label) == 0:
                continue
            # inplane rotation
            # -pi / 2 ~ pi / 2, symmetry
            thetas = gg_label[:, 3] - delta_thetas[i]
            theta_mask = np.logical_or(thetas > np.pi / 2, thetas < -np.pi / 2)
            thetas[theta_mask] = (thetas[theta_mask] +
                                  np.pi / 2) % np.pi - np.pi / 2
            gammas = gg_label[:, 4]
            betas = gg_label[:, 5]
            gammas[theta_mask] *= -1
            betas[theta_mask] *= -1
            gg_label[:, 3] = thetas
            gg_label[:, 4] = gammas
            gg_label[:, 5] = betas
            # offset
            gg_label[:, :2] = gg_label[:, :2] @ rot_mat[i].T
            gg_labels[i] = gg_label
    if aug_scale:
        # not applied to total_labels, because only use its rotation
        scale = torch.rand(len(gg_labels), device='cuda')
        scale = (2**scale)[:, None]  # (B, 1) (1 ~ 2)
        all_centers *= scale
        # rgb z gtz xy, only need to aug z
        all_patches[..., 3:5] *= scale[:, None, None]
        if all_partial_pcs is not None:
            all_partial_pcs *= scale[:, None]
            all_full_pcs *= scale[:, None]
        for i in range(len(gg_labels)):
            if len(gg_labels[i]) == 0:
                continue
            # x y z theta gamma beta width_2d score width_6d
            # offset
            gg_labels[i][:, :3] *= scale[i].cpu().numpy()
            # width
            gg_labels[i][:, -1:] *= scale[i].cpu().numpy()
            # filter only remaining valid grasps
            # offset filter
            mask = (gg_labels[i][:, :3]**2).sum(1) < 0.02**2
            gg_labels[i] = gg_labels[i][mask]
            # width filter
            mask = (gg_labels[i][:, -1] < 0.16)
            gg_labels[i] = gg_labels[i][mask]
    return all_centers, all_patches, all_partial_pcs, all_full_pcs, gg_labels, total_labels


def get_local_data_fps(localpaths, group_num, center_num):

    # get local data
    all_partial_pcs = torch.zeros((0, group_num, 3),
                                  dtype=torch.float32,
                                  device='cuda')
    # all_full_pcs = torch.zeros((0, group_num, 3),
    #                            dtype=torch.float32,
    #                            device='cuda')
    gg_labels = []

    for localpath in localpaths:
        data = np.load(localpath, allow_pickle=True)

        cur_partial_pcs_ori = torch.from_numpy(data['pcs_ori']).cuda()

        # add a little noise to avoid repeated points
        cur_partial_pcs_ori[..., :3] += torch.randn(
            cur_partial_pcs_ori.shape[:-1] + (3, ), device='cuda') * 5e-4

        # cur_full_pcs_ori = torch.from_numpy(data['full_pcs_ori']).cuda()
        lengths = torch.from_numpy(data['lengths']).cuda()
        # full_lengths = torch.from_numpy(data['full_lengths']).cuda()
        cur_grasps = data['grasps']

        # doing partial fps
        _, idxs = sample_farthest_points(cur_partial_pcs_ori[..., :3],
                                         lengths=lengths,
                                         K=group_num,
                                         random_start_point=True)
        # mv center of pc to (0, 0, 0), stack to pc_group
        temp_idxs = idxs[..., None].repeat(1, 1, 3)
        cur_partial_pcs = torch.gather(cur_partial_pcs_ori, 1, temp_idxs)

        # doing full fps
        # _, idxs = sample_farthest_points(cur_full_pcs_ori[..., :3],
        #                                     lengths=full_lengths,
        #                                     K=group_num,
        #                                     random_start_point=True)
        # # mv center of pc to (0, 0, 0), stack to pc_group
        # temp_idxs = idxs[..., None].repeat(1, 1, 3)
        # cur_full_pcs = torch.gather(cur_full_pcs_ori, 1, temp_idxs)

        # # permutate the order of points in a local region
        # permutation_idx = np.random.rand(cur_full_pcs.shape[1]).argsort()
        # cur_partial_pcs = np.take(cur_partial_pcs, permutation_idx, axis=1)
        # cur_full_pcs = np.take(cur_full_pcs, permutation_idx, axis=1)

        if center_num < cur_partial_pcs.shape[0]:
            idx = np.random.choice(range(cur_partial_pcs.shape[0]),
                                   size=center_num,
                                   replace=False)
            cur_partial_pcs = cur_partial_pcs[idx]
            # cur_full_pcs = cur_full_pcs[idx]
            cur_grasps = cur_grasps[idx]

        all_partial_pcs = torch.concat([all_partial_pcs, cur_partial_pcs], 0)
        # all_full_pcs = torch.concat([all_full_pcs, cur_full_pcs], 0)
        gg_labels.extend(cur_grasps)

    return all_partial_pcs, None, gg_labels


def get_local_data_collision(localpaths, collisionpaths, group_num,
                             center_num):
    # get local data
    all_partial_pcs = torch.zeros((0, group_num, 3),
                                  dtype=torch.float32,
                                  device='cuda')
    all_full_pcs = torch.zeros((0, group_num, 3),
                               dtype=torch.float32,
                               device='cuda')
    gg_labels = []
    partial_collision_labels = []
    full_collision_labels = []

    for localpath, collisionpath in zip(localpaths, collisionpaths):
        data = np.load(localpath, allow_pickle=True)
        collision = np.load(collisionpath, allow_pickle=True)

        cur_partial_pcs = data['pcs'][..., :3]
        cur_full_pcs = data['full_pcs'][..., :3]
        cur_grasps = data['grasps']
        cur_partial_colli_labels = collision[
            'partial_collision_labels'].astype('bool')
        cur_full_colli_labels = collision['full_collision_labels'].astype(
            'bool')

        if center_num < cur_partial_pcs.shape[0]:
            idx = np.random.choice(range(cur_partial_pcs.shape[0]),
                                   size=center_num,
                                   replace=False)
            cur_partial_pcs = torch.from_numpy(cur_partial_pcs[idx]).cuda()
            cur_full_pcs = torch.from_numpy(cur_full_pcs[idx]).cuda()
            cur_grasps = cur_grasps[idx]
            cur_partial_colli_labels = cur_partial_colli_labels[idx]
            cur_full_colli_labels = cur_full_colli_labels[idx]

        else:
            cur_partial_pcs = torch.from_numpy(cur_partial_pcs).cuda()
            cur_full_pcs = torch.from_numpy(cur_full_pcs).cuda()

        all_partial_pcs = torch.concat([all_partial_pcs, cur_partial_pcs], 0)
        all_full_pcs = torch.concat([all_full_pcs, cur_full_pcs], 0)
        gg_labels.extend(cur_grasps)
        partial_collision_labels.extend(cur_partial_colli_labels)
        full_collision_labels.extend(cur_full_colli_labels)

    partial_collision_labels = torch.Tensor(
        np.array(partial_collision_labels)).cuda()
    full_collision_labels = torch.Tensor(
        np.array(full_collision_labels)).cuda()
    return all_partial_pcs, all_full_pcs, gg_labels, partial_collision_labels, full_collision_labels


def get_ori_local_grasp_label(localpaths):
    # only for evaluation, thus batch size equals one
    # load grasp label and set it to torch size (all grasp in a view, 8)
    # not use now, beause evaluation uses original HGGD method
    localpath = localpaths[0]
    data = np.load(localpath, allow_pickle=True)
    grasps = data['grasps']
    gg_ori_local_labels = -torch.ones(
        (0, 8), dtype=torch.float32, device='cuda')

    for g in grasps:  # iteration valid grasp centers
        gg_ori_local_labels = torch.cat(
            [gg_ori_local_labels,
             torch.from_numpy(g).cuda()], 0)

    return gg_ori_local_labels


def get_ori_grasp_label(grasppath):
    """
    input:
        grasppaths:     tuple __len__ == batch size
        intricsics:     [3, 3]
    output:
        gg_ori_labels   [grasp_num, 8]
    """
    # load grasp
    grasp_label = np.load(grasppath[0])  # B=1
    grasp_num = grasp_label['centers_2d'].shape[0]
    gg_ori_labels = -np.ones((grasp_num, 8), dtype=np.float32)

    # get grasp original labels
    centers_2d = grasp_label['centers_2d']
    grasp_num = centers_2d.shape[0]
    gg_ori_labels[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                            grasp_label['center_z_depths'])
    gg_ori_labels[:, 3] = grasp_label['thetas_rad']
    gg_ori_labels[:, 4] = grasp_label['gammas_rad']
    gg_ori_labels[:, 5] = grasp_label['betas_rad']
    gg_ori_labels[:, 6] = grasp_label['widths_2d']
    gg_ori_labels[:, 7] = grasp_label['scores_from_6d']
    gg_ori_labels = torch.from_numpy(gg_ori_labels).cuda()

    return gg_ori_labels


# @timing
def get_center_group_label(local_center: List,
                           grasp_labels: List,
                           local_grasp_num,
                           dis=0.02) -> List:
    batch_size = len(local_center)
    gg_group_labels = []
    total_labels = torch.zeros((0, 9), dtype=torch.float32, device='cuda')
    for i in range(batch_size):
        # get grasp
        grasp_label = grasp_labels[i]
        # set up numpy grasp label
        centers_2d = grasp_label['centers_2d']
        grasp_num = centers_2d.shape[0]
        gg_label = -np.ones((grasp_num, 9), dtype=np.float32)
        gg_label[:, :3] = convert_2d_to_3d(centers_2d[:, 0], centers_2d[:, 1],
                                           grasp_label['center_z_depths'])
        gg_label[:, 3] = grasp_label['thetas_rad']
        gg_label[:, 4] = grasp_label['gammas_rad']
        gg_label[:, 5] = grasp_label['betas_rad']
        gg_label[:, 6] = grasp_label['widths_2d']
        gg_label[:, 7] = grasp_label['scores_from_6d']

        gt_rect_gg = RectGraspGroup()
        gt_rect_gg.load_from_dict(grasp_label, min_width=0, min_score=0)
        widths_6d = gt_rect_gg.get_gt_6d_width(gg_label[:, 2])
        gg_label[:, 8] = widths_6d

        # convert to cuda tensor
        gg_label = torch.from_numpy(gg_label).cuda()

        # cal distance to valid center
        valid_center = local_center[i]
        distance = square_distance(
            valid_center, gg_label)  # distance: (center_num, grasp_num)

        # select nearest grasp labels for all center
        mask = distance < dis**2
        for j in range(len(distance)):
            # mask with min dis
            mask_gg = gg_label[mask[j]]
            mask_distance = distance[j][mask[j]]
            if len(mask_distance) == 0:
                gg_group_labels.append(torch.zeros((0, 9)).cuda())
            else:
                # sorted and select nearest
                _, topk_idxs = torch.topk(mask_distance,
                                          k=min(local_grasp_num,
                                                mask_distance.shape[0]),
                                          largest=False)
                gg_nearest = mask_gg[topk_idxs]
                # move to (0, 0, 0)
                gg_nearest[:, :3] = gg_nearest[:, :3] - valid_center[j]
                gg_group_labels.append(gg_nearest)
                total_labels = torch.cat([total_labels, gg_nearest], 0)
    return gg_group_labels, total_labels


@njit
def select_area(loc_map, top, bottom, left, right, grid_size, overlap):
    center_num = len(top)
    local_areas = np.zeros((center_num, (grid_size + overlap * 2)**2))
    for j in range(center_num):
        # extend to make overlap
        local_area = loc_map[top[j]:bottom[j], left[j]:right[j]]
        local_area = np.ascontiguousarray(local_area).reshape((-1, ))
        local_areas[j, :len(local_area)] = local_area
    return local_areas


def select_2d_center(loc_maps,
                     center_num,
                     reduce='max',
                     grid_size=20,
                     use_local_max=False,
                     use_grid_max=True,
                     overlap=1) -> List:
    # deal with validation stage
    if isinstance(loc_maps, np.ndarray):
        loc_maps = np.copy(loc_maps)
    else:
        loc_maps = loc_maps.clone()
    if len(loc_maps.shape) == 2:
        loc_maps = loc_maps[None]
    # using torch to downsample
    if not use_local_max and isinstance(loc_maps, np.ndarray):
        loc_maps = torch.from_numpy(loc_maps).cuda()
    batch_size = loc_maps.shape[0]
    center_2ds = []
    if use_local_max:
        for i in range(batch_size):
            loc_map = loc_maps[i]
            local_max = peak_local_max(loc_map,
                                       min_distance=grid_size,
                                       threshold_abs=0,
                                       num_peaks=center_num)
            center_2ds.append(local_max)
    else:
        # using downsampled grid to avoid center too near
        new_size = (loc_maps.shape[1] // grid_size,
                    loc_maps.shape[2] // grid_size)
        # heat_grids = F.interpolate(loc_maps[None],
        #                              size=new_size,
        #                              mode='bilinear').squeeze()
        if reduce == 'avg':
            heat_grids = F.avg_pool2d(loc_maps[None], grid_size).squeeze()
        elif reduce == 'max':
            heat_grids = F.max_pool2d(loc_maps[None], grid_size).squeeze()
        else:
            raise RuntimeError(f'Unrecognized reduce: {reduce}')
        heat_grids = heat_grids.view((batch_size, -1))
        # get topk grid point
        for i in range(batch_size):
            local_idx = torch.topk(heat_grids[i],
                                   k=min(heat_grids.shape[1], center_num),
                                   dim=0)[1]
            local_max = np.zeros((len(local_idx), 2), dtype=np.int64)
            local_max[:, 0] = torch.div(local_idx,
                                        new_size[1],
                                        rounding_mode='floor').cpu().numpy()
            local_max[:, 1] = (local_idx % new_size[1]).cpu().numpy()
            if use_grid_max and grid_size > 1:
                # get local max in this grid point
                top, bottom = local_max[:, 0] * grid_size - overlap, (
                    local_max[:, 0] + 1) * grid_size + overlap
                top, bottom = np.maximum(0, top), np.minimum(
                    bottom, loc_maps.shape[1] - 1)
                left, right = local_max[:, 1] * grid_size - overlap, (
                    local_max[:, 1] + 1) * grid_size + overlap
                left, right = np.maximum(0, left), np.minimum(
                    right, loc_maps.shape[2] - 1)
                # using numba to faster get local areas
                local_areas = select_area(loc_maps[i].cpu().numpy(), top,
                                          bottom, left, right, grid_size,
                                          overlap)
                local_areas = torch.from_numpy(local_areas).float().cuda()
                # batch calculate
                grid_idxs = torch.argmax(local_areas, dim=1).cpu().numpy()
                local_max[:, 0] = top + grid_idxs // (right - left)
                local_max[:, 1] = left + grid_idxs % (right - left)
            else:
                # use grid center
                local_max[:, 0] = local_max[:, 0] * grid_size + grid_size // 2
                local_max[:, 1] = local_max[:, 1] * grid_size + grid_size // 2
            center_2ds.append(local_max)
    return center_2ds


# @timing
def data_process(
    points: torch.Tensor,
    depths: torch.Tensor,
    rect_ggs: List,
    center_num,
    group_num,
    output_size,
    grasp_widths=None,
    min_points=32,
    is_training=True,
):
    # select partial pc centers
    local_center, valid_depth_masks = center2dtopc(rect_ggs,
                                                   center_num,
                                                   depths,
                                                   output_size,
                                                   append_random_center=False,
                                                   is_training=is_training)
    if grasp_widths is not None:
        valid_grasp_widths = grasp_widths[valid_depth_masks]

    # seg point cloud
    pc_group, valid_local_centers, valid_center_masks, pc_group_ori, lengths = get_group_pc(
        points,
        local_center,
        group_num,
        rect_ggs,
        None if grasp_widths is None else valid_grasp_widths,
        min_points=min_points,
        is_training=is_training)
    # modify rect_ggs
    for i, mask in enumerate(valid_center_masks):
        rect_ggs[i] = rect_ggs[i][mask.cpu().numpy()]

    if grasp_widths is None:
        return pc_group, valid_local_centers, pc_group_ori, lengths

    valid_grasp_widths = valid_grasp_widths[
        valid_center_masks[0].cpu().numpy()]

    invalid_depth_index = np.where(valid_depth_masks == False)
    invalid_center_index = torch.where(valid_center_masks[0] == False)

    # if len(invalid_depth_index[0]) != 0 or len(invalid_center_index[0]) != 0:
    #     print('invalid index:', invalid_depth_index[0], invalid_center_index[0])

    return (
        pc_group,
        valid_local_centers,
        valid_grasp_widths,
        pc_group_ori,
        lengths,
        (invalid_depth_index[0], invalid_center_index[0]),
    )
