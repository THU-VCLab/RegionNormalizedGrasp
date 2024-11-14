from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from numba import jit, njit
from scipy.signal import medfilt2d
from torchvision.transforms.functional import gaussian_blur

from dataset.collision_detector import ModelFreeCollisionDetector

from .config import get_camera_intrinsic
from .grasp import GraspGroup, RectGrasp, RectGraspGroup
from .pc_dataset_tools import select_2d_center
from .utils import (angle_distance, convert_2d_to_3d, rotation_distance,
                    square_distance)


def anchor_output_process(loc_map,
                          cls_mask,
                          theta_offset,
                          depth_offset,
                          width_offset,
                          sigma=10):
    """Post-process the raw output of the network, convert to numpy arrays,
    apply filtering.

    :param loc_map: Location Map output of network (as torch Tensors)
    :param cls_mask: Classification Mask for anchor boxes
    :param theta_offset: Offset
    :param height_offset: Offset
    :param width_offset: Offset
    :return: Filtered Location Map, Filtered Cls Mask, Filtered Offset
    """
    # normlize every location map
    loc_map = torch.clamp(torch.sigmoid(loc_map), min=1e-4, max=1 - 1e-4)
    loc_map = gaussian_blur(loc_map, kernel_size=9, sigma=sigma)
    loc_map_gaussian = loc_map.detach().cpu().numpy().squeeze()
    # using sigmoid to norm class map
    cls_mask = torch.clamp(torch.sigmoid(cls_mask), min=1e-4, max=1 - 1e-4)
    cls_mask = cls_mask.detach().cpu().numpy().squeeze()
    # clamp regress offset
    theta_offset = torch.clamp(theta_offset.detach(), -0.5,
                               0.5).cpu().numpy().squeeze()
    depth_offset = torch.clamp(depth_offset.detach(), -0.5,
                               0.5).cpu().numpy().squeeze()
    width_offset = torch.clamp(width_offset.detach(), -0.5,
                               0.5).cpu().numpy().squeeze()

    return loc_map_gaussian, cls_mask, theta_offset, depth_offset, width_offset


@njit(fastmath=True)
def get_thetas_widths(theta_cls,
                      theta_offset,
                      width_reg,
                      anchor_w,
                      rotation_num,
                      log_width=True):
    thetas = np.zeros((0, ))
    widths_6d = np.zeros((0, ))
    center_num = theta_cls.shape[0]
    anchor_k = theta_cls.shape[1]
    for i in range(center_num):
        theta_clss = np.argsort(theta_cls[i])
        for anchor_cls in theta_clss[::-1][:rotation_num]:
            da = theta_offset[i, anchor_cls]
            # attention: our theta in [0, pi]
            theta_range = np.pi
            anchor_step = theta_range / anchor_k
            theta = anchor_step * (anchor_cls + da + 0.5) - theta_range / 2
            thetas = np.hstack((thetas, np.array([theta])))
            # normalize
            dw = width_reg[i, anchor_cls]
            if log_width:
                w = anchor_w * np.exp(dw) / 1e3
            else:
                w = anchor_w * dw / 1e3

            widths_6d = np.hstack((widths_6d, np.array([w])))

    return thetas, widths_6d


@njit(fastmath=True)
def faster_detect_2d(loc_map, local_max, anchor_clss, grid_points,
                     theta_offset, depth_offset, width_offset, rotation_num,
                     anchor_k, anchor_w, anchor_z, grasp_nms, center_num):
    centers = np.zeros((0, 2))
    scores = np.zeros((0, ))
    depths = np.zeros((0, ))
    thetas = np.zeros((0, ))
    widths = np.zeros((0, ))
    for i in range(len(local_max)):
        if len(centers) >= center_num:
            break
        for anchor_cls in anchor_clss[i, ::-1][:rotation_num]:
            if len(centers) >= center_num:
                break
            pos = (anchor_cls, grid_points[i, 0], grid_points[i, 1])
            da = theta_offset[pos]
            dz = depth_offset[pos]
            dw = width_offset[pos]
            # recover depth and width
            depth = dz * anchor_z * 2
            w = anchor_w * np.exp(dw)
            # attention: our theta in [0, pi]
            theta_range = np.pi
            anchor_step = theta_range / anchor_k
            theta = anchor_step * (anchor_cls + da + 0.5) - theta_range / 2
            score = loc_map[local_max[i, 0], local_max[i, 1]]
            # grasp nms, dis > grid_size and delta angle > pi /6
            isnew = True
            if grasp_nms > 0 and len(centers) > 0:
                center_dis = np.sqrt(
                    np.sum(np.square(centers - local_max[i]), axis=1))
                angle_dis = np.abs(thetas - theta)
                angle_dis = np.minimum(np.pi - angle_dis, angle_dis)
                mask = np.logical_and(center_dis < grasp_nms, angle_dis
                                      < np.pi / 6)
                isnew = (not mask.any())
            if isnew:
                centers = np.vstack((centers, np.expand_dims(local_max[i], 0)))
                thetas = np.hstack((thetas, np.array([theta])))
                scores = np.hstack((scores, np.array([score])))
                depths = np.hstack((depths, np.array([depth])))
                widths = np.hstack((widths, np.array([w])))
    return centers, widths, depths, scores, thetas


@njit(fastmath=True)
def center_nms(local_max, grasp_nms, center_num):
    centers = np.zeros((0, 2))
    for i in range(len(local_max)):
        if len(centers) >= center_num:
            break
        # center nms, dis > grid_size
        isnew = True
        if grasp_nms > 0 and len(centers) > 0:
            center_dis = np.sum((centers - local_max[i])**2, axis=1)
            mask = (center_dis < grasp_nms**2)
            isnew = (not mask.any())
        if isnew:
            centers = np.vstack((centers, np.expand_dims(local_max[i], 0)))
    thetas = np.zeros((len(centers), ))
    scores = np.zeros((len(centers), ))
    depths = np.zeros((len(centers), ))
    widths = np.zeros((len(centers), ))
    return centers, widths, depths, scores, thetas


def sample_2d_centers(loc_map,
                      grid_size,
                      center_num,
                      grasp_nms=8,
                      reduce='max',
                      use_local_max=False):
    local_max = select_2d_center(loc_map,
                                 center_num * 3,
                                 grid_size=grid_size,
                                 reduce=reduce,
                                 use_local_max=use_local_max)
    centers, widths, depths, scores, thetas = center_nms(
        local_max[0], grasp_nms, center_num)
    return RectGraspGroup(centers=np.array(centers, dtype=np.int64),
                          heights=np.full((len(centers), ), 25),
                          widths=np.array(widths),
                          depths=np.array(depths),
                          scores=np.array(scores),
                          thetas=np.array(thetas))


def detect_2d_grasp(loc_map,
                    cls_mask,
                    theta_offset,
                    depth_offset,
                    width_offset,
                    ratio,
                    grid_size=20,
                    anchor_k=6,
                    anchor_w=25.0,
                    anchor_z=20.0,
                    mask_thre=0,
                    center_num=1,
                    rotation_num=1,
                    reduce='max',
                    use_local_max=False,
                    grasp_nms=8,
                    random_center=False) -> RectGraspGroup:
    """Detect grasps in a network output.

    :return: list of Grasps
    """
    if random_center:
        local_max = np.zeros((center_num * 10, 2))
        local_max[:, 0] = np.random.randint(0,
                                            loc_map.shape[0] - 1,
                                            size=center_num * 10)
        local_max[:, 1] = np.random.randint(0,
                                            loc_map.shape[1] - 1,
                                            size=center_num * 10)
        local_max = [local_max.astype(np.int64)]
    else:
        local_max = select_2d_center(loc_map,
                                     center_num * 3,
                                     grid_size=grid_size,
                                     reduce=reduce,
                                     use_local_max=use_local_max)

    # batch calculation
    loc_map = loc_map.squeeze()
    local_max = local_max[0]
    # filter by heatmap
    qualitys = loc_map[local_max[:, 0], local_max[:, 1]]
    quality_mask = (qualitys > mask_thre)
    local_max = local_max[quality_mask]
    grid_points = local_max // ratio
    cls_qualitys = cls_mask[:, grid_points[:, 0], grid_points[:, 1]].T
    # sort cls score
    anchor_clss = np.argsort(cls_qualitys)
    # print(grid_points.shape, cls_mask.shape, cls_qualitys.shape, anchor_clss.shape)
    centers, widths, depths, scores, thetas = faster_detect_2d(
        loc_map, local_max, anchor_clss, grid_points, theta_offset,
        depth_offset, width_offset, rotation_num, anchor_k, anchor_w, anchor_z,
        grasp_nms, center_num)
    grasps = RectGraspGroup(centers=np.array(centers, dtype=np.int64),
                            heights=np.full((len(centers), ), 25),
                            widths=np.array(widths),
                            depths=np.array(depths),
                            scores=np.array(scores),
                            thetas=np.array(thetas))
    return grasps


@njit(fastmath=True)
def faster_detect_6d(pred_gammas, pred_betas, offset, scores, local_centers,
                     anchor_idxs, thetas, widths, k, intrinsics):
    pred_grasp = np.zeros((0, 8))
    cur_grasp = np.zeros((k, 8))
    center_2ds = np.zeros((0, 2))
    center_depths = np.zeros((0, ))
    cur_offsets = np.zeros((k, 3))
    for i in range(len(pred_gammas)):
        for j in range(k):
            cur_offsets[j] = offset[i, anchor_idxs[i, j]]
        # pred grasp: x,y,z,theta,gamma,beta,width
        cur_centers = local_centers[i] + cur_offsets
        cur_grasp[:, :3] = cur_centers
        cur_grasp[:, 3] = np.repeat(thetas[i], k)
        cur_grasp[:, 4] = pred_gammas[i]
        cur_grasp[:, 5] = pred_betas[i]
        cur_grasp[:, 6] = np.repeat(widths[i], k)
        cur_grasp[:, 7] = scores[i]
        # stack on all grasp array
        pred_grasp = np.vstack((pred_grasp, cur_grasp))
        # cal 2d centers after offset
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        cur_center_2ds = np.zeros((k, 2))
        cur_center_2ds[:, 0] = cur_centers[:, 0] / cur_centers[:, 2] * fx + cx
        cur_center_2ds[:, 1] = cur_centers[:, 1] / cur_centers[:, 2] * fy + cy
        # update center info for generating RectGraspGroup
        center_2ds = np.vstack((center_2ds, cur_center_2ds))
        center_depths = np.concatenate(
            (center_depths, cur_centers[:, 2] * 1e3))
    return pred_grasp, center_2ds, center_depths


def get_rotation(thetas, gammas, betas):
    # TODO: check theta initialization way
    rotations = np.zeros((len(thetas), 3, 3))
    rotations[:, 0, 1] = -np.cos(thetas)
    rotations[:, 0, 2] = -np.sin(thetas)
    rotations[:, 1, 1] = np.sin(thetas)
    rotations[:, 1, 2] = -np.cos(thetas)
    rotations[:, 2, 0] = 1

    # 绕夹具本身坐标系z轴，旋转gamma角，右乘！！！
    R = np.zeros((len(thetas), 3, 3))
    R[:, 0, 0] = np.cos(gammas)
    R[:, 0, 1] = -np.sin(gammas)
    R[:, 1, 0] = np.sin(gammas)
    R[:, 1, 1] = np.cos(gammas)
    R[:, 2, 2] = 1
    rotations = rotations @ R  # batch mm

    # 绕夹具本身坐标系y轴，旋转beta角, 右乘！！！
    R = np.zeros((len(thetas), 3, 3))
    R[:, 0, 0] = np.cos(betas)
    R[:, 0, 2] = np.sin(betas)
    R[:, 1, 1] = 1
    R[:, 2, 0] = -np.sin(betas)
    R[:, 2, 2] = np.cos(betas)
    rotations = rotations @ R  # batch mm

    return rotations


def detect_6d_grasp_multi(thetas,
                          widths_6d,
                          pred: torch.Tensor,
                          offset: torch.Tensor,
                          valid_center_list: List,
                          anchors,
                          alpha=0.02,
                          k=5):
    # pre-process
    anchor_num = len(anchors['gamma'])
    pred = torch.sigmoid(pred)
    offset = alpha * torch.clip(offset, -1, 1)
    # select top-k every row
    scores, anchor_idxs = torch.topk(pred, k, 1)
    local_centers = valid_center_list[0].cpu().numpy()
    offset = offset.cpu().numpy()
    scores = scores.cpu().numpy()
    anchor_idxs = anchor_idxs.cpu().numpy()
    # split anchor to gamma and beta
    gamma_idxs = (anchor_idxs % anchor_num).flatten()
    beta_idxs = (anchor_idxs // anchor_num).flatten()
    norm_gammas = anchors['gamma'][gamma_idxs].cpu().numpy().reshape(
        anchor_idxs.shape)
    norm_betas = anchors['beta'][beta_idxs].cpu().numpy().reshape(
        anchor_idxs.shape)
    # make back to [-pi / 2, pi / 2]
    pred_gammas = norm_gammas * torch.pi / 2
    pred_betas = norm_betas * torch.pi / 2
    # get offset
    intrinsics = get_camera_intrinsic()
    pred_grasp, center_2ds, center_depths = faster_detect_6d(
        pred_gammas, pred_betas, offset, scores, local_centers, anchor_idxs,
        thetas, widths_6d, k, intrinsics)
    # pred_grasp (valid_center_num * local_k, 8)
    # filter by gamma
    gamma_mask = (np.abs(pred_grasp[:, 4]) < np.pi / 3).reshape(-1)
    pred_grasp = pred_grasp[gamma_mask]
    center_2ds = center_2ds[gamma_mask]
    center_depths = center_depths[gamma_mask]

    # print(pred_gammas.shape, pred_grasp.shape, gamma_mask.reshape(-1).shape)
    rotations = get_rotation(pred_grasp[:, 3], pred_grasp[:, 4], pred_grasp[:,
                                                                            5])
    if len(pred_grasp) > 0:
        pred_6d_gg = GraspGroup(translations=pred_grasp[:, :3],
                                rotations=rotations,
                                widths=pred_grasp[:, 6],
                                depths=alpha * np.ones(pred_grasp.shape[0], ),
                                scores=pred_grasp[:, 7],
                                heights=alpha * np.ones(pred_grasp.shape[0], ))
    else:
        pred_6d_gg = GraspGroup()

    return pred_grasp, pred_6d_gg


def calculate_6d_match(pred_grasp: torch.Tensor,
                       gg_ori_label,
                       threshold_dis,
                       threshold_rot,
                       seperate=False):
    # center distance
    distance = square_distance(pred_grasp, gg_ori_label)
    mask_distance = (distance < threshold_dis**2)
    soft_mask_distance = (distance < (threshold_dis * 2)**2)

    # angle distance
    if seperate:
        # compute angle distance bewteen pred and label seperately
        dis_theta = angle_distance(pred_grasp[:, 3], gg_ori_label[:, 3])
        dis_gamma = angle_distance(pred_grasp[:, 4], gg_ori_label[:, 4])
        dis_beta = angle_distance(pred_grasp[:, 5], gg_ori_label[:, 5])

        # get mask and logical_or along label axis
        mask_theta = (dis_theta < threshold_rot)
        mask_gamma = (dis_gamma < threshold_rot)
        mask_beta = (dis_beta < threshold_rot)

        mask_rot = torch.logical_and(torch.logical_and(mask_theta, mask_beta),
                                     mask_gamma)
    else:
        # using total rotation distance
        dis_rot = rotation_distance(pred_grasp[:, 3:6], gg_ori_label[:, 3:6])
        mask_rot = (dis_rot < threshold_rot)

    mask = torch.logical_and(mask_distance, mask_rot).any(1)

    # cal angle correct rate
    batch_size = mask_distance.size(0)
    correct_dis = mask_distance.any(1).sum()
    correct_rot = torch.logical_and(mask_rot, soft_mask_distance).any(1).sum()

    # get count
    correct_grasp = mask.sum()
    acc = correct_grasp / pred_grasp.shape[0]

    # create T & F array
    r_g = np.array([correct_grasp.cpu(), batch_size - correct_grasp.cpu()])
    r_d = np.array([correct_dis.cpu(), batch_size - correct_dis.cpu()])
    r_r = np.array([correct_rot.cpu(), batch_size - correct_rot.cpu()])
    return r_g, r_d, r_r


def calculate_coverage(pred_grasp: torch.Tensor,
                       gg_ori_label: torch.Tensor,
                       threshold_dis=0.02,
                       threshold_rot=0.1):
    # center distance
    distance = square_distance(gg_ori_label, pred_grasp)
    mask_distance = (distance < threshold_dis**2)

    # angle distance, using total rotation distance
    dis_rot = rotation_distance(gg_ori_label[:, 3:6], pred_grasp[:, 3:6])
    mask_rot = (dis_rot < threshold_rot)

    mask = torch.logical_and(mask_distance, mask_rot).any(1)
    cover_cnt = mask.sum()
    return cover_cnt


def calculate_guidance_accuracy(g, gt_bbs):
    # check center dis and theta dis
    gts = RectGraspGroup()
    for bb in gt_bbs:
        gt = RectGrasp.from_bb(bb)
        gts.append(gt)
    gts_6d = gts.to_6d_grasp_group()

    g_6d = RectGraspGroup()
    g_6d.append(g)
    g_6d = g_6d.to_6d_grasp_group()[0]

    center_dis = np.sqrt(
        np.square(g_6d.translation - gts_6d.translations).sum(1))
    theta_dis = np.abs(g.theta - gts.thetas)
    theta_dis = np.minimum(np.pi - theta_dis, theta_dis)
    center_mask = (center_dis < 0.02)
    theta_mask = (theta_dis < np.pi / 6)
    mask = np.logical_and(center_mask, theta_mask)
    return mask.any()


def calculate_iou_match(gs, gt_bbs, thre=0.25):
    # check iou
    for g in gs:
        for bb in gt_bbs:
            gt = RectGrasp.from_bb(bb)
            if g.iou(gt) > thre:
                return True
    return False


def collision_detect(points_all: torch.Tensor, pred_gg, mode='regnet'):
    # collison detect
    cloud = points_all[:, :3].clone().to(torch.float32)
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=0.01, mode=mode)
    no_collision_mask = mfcdetector.detect(pred_gg, approach_dist=0.05)
    collision_free_gg = pred_gg[no_collision_mask]
    return collision_free_gg, no_collision_mask
