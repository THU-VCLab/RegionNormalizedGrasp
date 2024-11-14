import copy
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numba import njit, prange
from numba.typed import List as typed_List
from torch.autograd import Variable

from dataset.utils import angle_distance, rotation_distance

eps = 1e-6


def ce_loss(pred, targets):
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    loss = -(torch.log(pred) * targets + torch.log(1 - pred) * (1 - targets))
    return loss.sum() / len(pred)


def focal_loss(pred,
               targets,
               thres=0.99,
               alpha=0.5,
               gamma=2,
               neg_suppress=False):
    pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
    pos_inds = targets.ge(thres).float()
    neg_inds = targets.lt(thres).float()

    pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds
    neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred,
                                                             gamma) * neg_inds

    if neg_suppress:
        neg_loss *= torch.pow(1 - targets, 4)

    num_pos = pos_inds.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def loc_loss(pred, targets):
    """Penalty-reduced focal loss for Location Map.

    :param pred: predicted Location Map [B, 1, H, W]
    :param targets: ground truth [B, 1, H, W]
    :return: Location Map loss
    """
    loss = focal_loss(pred, targets, neg_suppress=True)
    return loss


def offset_reg_loss(regs, gt_regs, mask):
    """Regression loss for offsets (z, w, theta) in Multi-Grasp Generator.

    :param regs: offset [B, anchor_k, H/r, W/r]
    :param gt_regs: ground truth [B, anchor_k, H/r, W/r]
    :param mask: classification mask
    :return: regression loss
    """
    mask = (mask > 0)
    regs = [torch.clamp(r, min=-0.5, max=0.5) for r in regs]
    loss = sum(
        F.smooth_l1_loss(r * mask, gt_r * mask, reduction='sum') /
        (mask.sum() + 1e-4) for r, gt_r in zip(regs, gt_regs))
    return loss / len(regs)


def anchor_cls_loss(pred, target, thres):
    return focal_loss(pred, target, thres=thres, alpha=0.25)


def compute_anchor_loss(pred, target, loc_a=1, reg_b=0.5, cls_c=1):
    """Compute Total Loss.

    :param pred: A tuple (Location Map, Cls Mask, Offset theta, Offset h, Offset w)
    :param target: ground truth
    :param reg_a: weight of reg loss
    :param cls_b: weight of cls loss
    :return: total loss
    """
    pos_pred = pred[0]
    pos_target = target[0]
    pos_loss = loc_loss(pos_pred, pos_target)

    angle_q_pred = pred[1]
    angle_q_target = target[1]
    cls_loss = anchor_cls_loss(angle_q_pred, angle_q_target, thres=0.5)
    reg_loss = offset_reg_loss(pred[2:], target[2:], angle_q_target)
    loss = loc_a * pos_loss + reg_b * reg_loss + cls_c * cls_loss
    return {
        'loss': loss,
        'losses': {
            'loc_map_loss': loc_a * pos_loss,
            'reg_loss': reg_b * reg_loss,
            'cls_loss': cls_c * cls_loss
        }
    }


def compute_collision_loss(full_collision_labels, pred_collision_cls):
    # print(torch.sigmoid(pred_collision_cls[0][:10]), full_collision_labels[0][:10])
    # print(torch.sum(full_collision_labels)/(full_collision_labels.shape[0]*full_collision_labels.shape[1]))
    n = pred_collision_cls.shape[0] * pred_collision_cls.shape[1]
    pred_collision_cls = torch.concat([
        pred_collision_cls.reshape(n, 1), 1 - pred_collision_cls.reshape(n, 1)
    ], 1)
    full_collision_labels = torch.concat([
        full_collision_labels.reshape(n, 1),
        1 - full_collision_labels.reshape(n, 1)
    ], 1)

    loss = focal_loss(pred_collision_cls, full_collision_labels)
    return loss


def compute_theta_width_loss(pred_theta_cls,
                             pred_theta_offset,
                             pred_width_reg,
                             gg_labels: list,
                             anchor_w,
                             reg_b=5,
                             cls_c=1,
                             reg_w=1,
                             log_width=True):
    """Get theta and width pred losses.

    Args:
        pred_theta_cls : [B*center_num, 6]
        pred_theta_offset : [B*center_num, 6]
        pred_width_reg : [B*center_num, 6]
        gg_labels (list): labels for each center
    """
    center_num = pred_theta_cls.shape[0]
    anchor_num = pred_theta_cls.shape[1]
    # get theta labels
    cls_labels = np.zeros((center_num, anchor_num), dtype=np.float32)
    offset_labels = np.zeros((center_num, anchor_num), dtype=np.float32)
    width_labels = np.zeros((center_num, anchor_num), dtype=np.float32)
    for i, gg_label in enumerate(gg_labels):  # iteration every center
        if gg_label.shape[0] > 0:
            if torch.is_tensor(gg_label):
                gg_label = gg_label.cpu().numpy()

            ## get theta
            # set anchor for theta angle, need to change to 2 * pi / k
            theta_range = np.pi
            anchor_step = theta_range / anchor_num
            # get theta label, note that -pi / 2 ~ pi / 2, symmetry
            theta_label = np.copy(gg_label[:, 3])
            theta_label_mask = np.logical_or(theta_label >= np.pi / 2,
                                             theta_label <= -np.pi / 2)
            theta_label[theta_label_mask] = (theta_label[theta_label_mask] +
                                             np.pi / 2) % np.pi - np.pi / 2
            # clip to avoid exceeding anchor range
            theta = np.clip(theta_label, -theta_range / 2 + eps,
                            theta_range / 2 - eps)
            g_pos, delta_theta = np.divmod(theta + theta_range / 2,
                                           anchor_step)
            theta_offset = delta_theta / anchor_step - 0.5
            g_pos = g_pos.astype(np.int64)  # (graspnum,)

            # using normalized grasp space
            if log_width:
                width_reg = np.log(gg_label[:, -1] * 1e3 / anchor_w)
            else:
                width_reg = np.clip(gg_label[:, -1] * 1e3 / anchor_w, 0, 1)

            np.add.at(cls_labels[i], g_pos, 1)
            np.add.at(offset_labels[i], g_pos, theta_offset)
            np.add.at(width_labels[i], g_pos, width_reg)

    # average for offset
    count_map = cls_labels + (cls_labels == 0)
    offset_labels = offset_labels / count_map
    width_labels = width_labels / count_map
    # print(width_labels.mean(), width_labels.min(), width_labels.max())

    # sigmoid for cls mask
    cls_labels = cls_labels / (cls_labels.max(-1, keepdims=True) + eps)
    cls_labels = 2 / (1 + np.exp(-cls_labels)) - 1

    cls_labels = torch.from_numpy(cls_labels).cuda()
    offset_labels = torch.from_numpy(offset_labels).cuda()
    width_labels = torch.from_numpy(width_labels).cuda()

    # classification loss
    cls_loss = anchor_cls_loss(pred_theta_cls, cls_labels, thres=0.4)

    # mask smooth l1 loss for valid positions
    mask = (cls_labels > 0)
    reg_theta_loss = F.smooth_l1_loss(pred_theta_offset * mask,
                                      offset_labels * mask,
                                      reduction='sum') / mask.sum()
    reg_width_loss = F.smooth_l1_loss(pred_width_reg * mask,
                                      width_labels * mask,
                                      reduction='sum') / mask.sum()

    loss = reg_b * reg_theta_loss + cls_c * cls_loss + reg_w * reg_width_loss

    return {
        'loss': loss,
        'losses': {
            'reg_loss': reg_b * reg_theta_loss,
            'cls_loss': cls_c * cls_loss,
            'width_loss': reg_w * reg_width_loss
        }
    }


def compute_multicls_loss(pred,
                          offset,
                          gg_labels: list,
                          grasp_info,
                          anchors,
                          label_thres=0.99,
                          offset_thres=0,
                          args=None,
                          return_labels=False,
                          use_pose_dis=False,
                          regress=False):
    anchor_num = len(anchors['gamma'])

    # get q anchors
    anchors_gamma = anchors['gamma'] * torch.pi / 2
    anchors_beta = anchors['beta'] * torch.pi / 2
    # pre calculate anchor_eulers in order: [theta, gamma, beta]
    anchor_eulers = torch.zeros((anchor_num**2, 3), dtype=torch.float32)
    # attention: euler angle is [theta, gamma, beta]
    # but our anchor is gamma + beta * anchor_num
    beta_gamma = torch.cartesian_prod(anchors_beta, anchors_gamma)
    anchor_eulers[:, 1] = beta_gamma[:, 1]
    anchor_eulers[:, 2] = beta_gamma[:, 0]

    # use jit function to speed up
    # prepare input
    thetas = grasp_info[:, 0].cpu().numpy().astype(np.float64)
    if torch.is_tensor(gg_labels[0]):
        label_offsets = typed_List(
            [g[:, :3].cpu().numpy().astype(np.float64) for g in gg_labels])
        label_eulers = typed_List(
            [g[:, 3:6].cpu().numpy().astype(np.float64) for g in gg_labels])
    else:
        label_offsets = typed_List(
            [g[:, :3].astype(np.float64) for g in gg_labels])
        label_eulers = typed_List(
            [g[:, 3:6].astype(np.float64) for g in gg_labels])
    multi_labels, offset_labels = faster_get_local_labels(
        anchor_eulers.numpy().astype(np.float64),
        thetas,
        label_eulers,
        label_offsets,
        use_pose_dis=use_pose_dis,
        offset_thres=offset_thres,
        point_num=6)
    # print(multi_labels.max(), multi_labels[multi_labels > 0].min())
    # for t in [0.9, 0.8, 0.7, 0.6, 0.5]:
    #     print(t, (multi_labels > t).sum() / np.prod(multi_labels.shape))
    # to cuda
    multi_labels = torch.from_numpy(multi_labels).to(device='cuda',
                                                     dtype=torch.float32)
    offset_labels = torch.from_numpy(offset_labels).to(device='cuda',
                                                       dtype=torch.float32)

    # compute focal loss for anchor
    if regress:
        loss_multi = F.smooth_l1_loss(pred, multi_labels,
                                      reduction='mean') * 10
    else:
        # loss_multi = ce_loss(pred, multi_labels)
        loss_multi = focal_loss(pred, multi_labels, thres=label_thres)
    # mask smooth l1 loss for offset
    offset_labels /= args.alpha
    # regress loss for offset
    if regress:
        loss_offset = F.smooth_l1_loss(offset, offset_labels, reduction='mean')
    else:
        # masked loss for valid rotation
        labels_mask = (multi_labels > label_thres)[..., None]
        loss_offset = F.smooth_l1_loss(offset * labels_mask,
                                       offset_labels * labels_mask,
                                       reduction='sum')
        loss_offset /= labels_mask.sum() + eps
    if return_labels:
        return loss_multi, loss_offset * args.offset_coef, (
            multi_labels > label_thres).float()
    return loss_multi, loss_offset * args.offset_coef


@njit
def faster_get_local_labels(eulers,
                            thetas,
                            label_eulers,
                            label_offsets,
                            use_pose_dis=False,
                            offset_thres=0,
                            point_num=6):

    def qmul(q0, q1):
        w0, x0, y0, z0 = q0.T
        w1, x1, y1, z1 = q1.T
        return np.stack(
            (-x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1, x0 * w1 + y0 * z1 -
             z0 * y1 + w0 * x1, -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
             x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1), 1)

    def euler2quaternion(euler):
        N = len(euler)
        qx = np.stack((np.cos(euler[:, 0] / 2), np.sin(
            euler[:, 0] / 2), np.zeros(
                (N, ), dtype=np.float64), np.zeros((N, ), dtype=np.float64)),
                      1)
        qy = np.stack(
            (np.cos(euler[:, 1] / 2), np.zeros(
                (N, ), dtype=np.float64), np.sin(
                    euler[:, 1] / 2), np.zeros((N, ), dtype=np.float64)), 1)
        qz = np.stack(
            (np.cos(euler[:, 2] / 2), np.zeros(
                (N, ), dtype=np.float64), np.zeros(
                    (N, ), dtype=np.float64), np.sin(euler[:, 2] / 2)), 1)
        q = qmul(qmul(qx, qy), qz)
        return q

    def euler2rotation(thetas, gammas, betas):
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
        for i in prange(len(thetas)):
            rotations[i] = rotations[i] @ R[i]  # batch mm

        # 绕夹具本身坐标系y轴，旋转beta角, 右乘！！！
        R = np.zeros((len(thetas), 3, 3))
        R[:, 0, 0] = np.cos(betas)
        R[:, 0, 2] = np.sin(betas)
        R[:, 1, 1] = 1
        R[:, 2, 0] = -np.sin(betas)
        R[:, 2, 2] = np.cos(betas)
        for i in prange(len(thetas)):
            rotations[i] = rotations[i] @ R[i]  # batch mm
        return rotations

    def control_point_distance(a, b, offset, point_num):

        def sample_gripper_points(gripper_width=0.085):
            # depth == 0.02
            y_offset = gripper_width / 2
            if point_num == 6:
                gripper = np.array([[-0.10, 0, 0], [-0.04, 0, 0],
                                    [-0.04, y_offset, 0],
                                    [-0.04, -y_offset, 0], [0.02, y_offset, 0],
                                    [0.02, -y_offset, 0]])
            else:
                gripper = np.array([[-0.10, 0, 0], [-0.07, 0,
                                                    0], [-0.04, 0, 0],
                                    [-0.04, y_offset / 2, 0],
                                    [-0.04, -y_offset / 2, 0],
                                    [-0.04, y_offset,
                                     0], [-0.04, -y_offset, 0],
                                    [-0.01, y_offset, 0],
                                    [-0.01, -y_offset, 0], [0.02, y_offset, 0],
                                    [0.02, -y_offset, 0]])
            return gripper

        def pose2points(euler, offset=None):
            # euler: Mx3 offset: Mx3
            gripper = sample_gripper_points()  # point_num, 3
            points = np.zeros((len(euler), len(gripper), 3))  # M, point_num, 3
            rot = euler2rotation(euler[:, 0], euler[:, 1], euler[:,
                                                                 2])  # Mx3x3
            for i in prange(len(points)):
                # left multiply
                points[i] = gripper @ rot[i].T
            if offset is not None:
                points = points + offset[:, None]
            return points

        # symmetry for gripper
        a_r = a * np.array([[1, -1, -1]])
        a_r[:, 0] += np.pi
        # convert to control points
        anchor_dis = 0.1  # dis >= 0.1 -> score == 0, dis == 0 -> score = 1
        p_a = pose2points(a)  # M, point_num, 3
        p_a_r = pose2points(a_r)  # M, point_num, 3
        p_b = pose2points(b, offset)  # M, point_num, 3

        # tile dis matrix
        p_a = p_a.repeat(len(p_b)).reshape((len(a), point_num, 3, len(p_b)))
        p_a = p_a.transpose(0, 3, 1, 2)  # M, grasp_num, point_num, 3
        p_a_r = p_a_r.repeat(len(p_b)).reshape(
            (len(a), point_num, 3, len(p_b)))
        p_a_r = p_a_r.transpose(0, 3, 1, 2)  # M, grasp_num, point_num, 3
        p_b = np.expand_dims(p_b, 0)  # 1, grasp_num, point_num, 3

        # cal add
        dis = np.sqrt(((p_a - p_b)**2).sum(-1))  # M, grasp_num, point_num
        dis = dis.sum(-1) / point_num  # M, grasp_num
        dis_r = np.sqrt(((p_a_r - p_b)**2).sum(-1))  # M, grasp_num, point_num
        dis_r = dis_r.sum(-1) / point_num  # M, grasp_num

        return np.clip(np.minimum(dis, dis_r) / anchor_dis, 0, 1)

    def rotation_distance(a, b):
        q_a = euler2quaternion(a)
        q_b = euler2quaternion(b)
        # symmetry for gripper
        a_r = a * np.array([[1, -1, -1]])
        a_r[:, 0] += np.pi
        q_a_r = euler2quaternion(a_r)
        # q_a (grasp_cnt, 4) q_b (label_cnt, 4)
        dis = 1 - np.maximum(np.abs(np.dot(q_a, q_b.T)),
                             np.abs(np.dot(q_a_r, q_b.T)))
        return dis

    # get size
    M, N = len(eulers), len(label_eulers)
    # get multi labels
    multi_labels = np.zeros((N, M), dtype=np.float64)
    offset_labels = np.zeros((N, M, 3), dtype=np.float64)
    # set up eulers for N centers
    for i in range(N):
        if len(label_eulers[i]) > 0:
            # set current anchors
            eulers[:, 0] = thetas[i]
            # cal labels dis to nearest anchors
            if use_pose_dis:
                # M * local_grasp_num
                rot_dis = control_point_distance(eulers, label_eulers[i],
                                                 label_offsets[i], point_num)
            else:
                # M * local_grasp_num
                rot_dis = rotation_distance(eulers, label_eulers[i])
            # get labels to earest anchors
            for j in prange(M):
                # soft dis label
                multi_labels[i, j] = 1 - rot_dis[j].min()
                # set offset labels with thres
                if multi_labels[i, j] > offset_thres:
                    idx = np.argmin(rot_dis[j])
                    offset_labels[i, j] = label_offsets[i][idx]
    return multi_labels, offset_labels


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network."""

    def __init__(self, T=4):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        # attention: normalize is necessary here
        p_s = torch.sigmoid(y_s / self.T)
        p_s = torch.log(p_s / p_s.sum(1, keepdim=True))
        p_t = torch.sigmoid(y_t / self.T)
        p_t = p_t / p_t.sum(1, keepdim=True)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
        return loss
