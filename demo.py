import argparse
import os
import random
from time import time

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image
from thop import clever_format, profile
import cv2

from dataset.config import get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, collision_detect,
                                detect_2d_grasp, detect_6d_grasp_multi,
                                get_thetas_widths)
from dataset.pc_dataset_tools import center2dtopc
from dataset.grasp import RectGraspGroup
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PatchMultiGraspNet

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint-path', default=None)

# image input
parser.add_argument('--rgb-path')
parser.add_argument('--depth-path')

# paras from hggd (some are useless but not deleted for convinence)
parser.add_argument('--input-h', type=int, default=360)
parser.add_argument('--input-w', type=int, default=640)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--hggd-anchor-w', type=float, default=75.0)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=int, default=12)

# pc
parser.add_argument('--all-points-num', type=int, default=25600)
parser.add_argument('--center-num', type=int)
parser.add_argument('--group-num', type=int)

# patch
parser.add_argument('--patch-size',
                    type=int,
                    default=64,
                    help='local patch grid size')
parser.add_argument('--alpha',
                    type=float,
                    default=0.02,
                    help='grasp center crop range')

# net
parser.add_argument('--embed-dim', type=int)
parser.add_argument('--anchor-w', type=float, default=60.0)
parser.add_argument('--anchor-num', type=int, default=7)

# grasp detection
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# others
parser.add_argument('--random-seed', type=int, default=123, help='Random seed')

args = parser.parse_args()

eps = 1e-6


class PointCloudHelper:

    def __init__(self, all_points_num) -> None:
        # precalculate x,y map
        self.all_points_num = all_points_num
        # get intrinsics
        intrinsics = get_camera_intrinsic()
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        # cal x, y
        ymap, xmap = np.meshgrid(np.arange(720), np.arange(1280))
        points_x = (xmap - cx) / fx
        points_y = (ymap - cy) / fy
        self.points_x = torch.from_numpy(points_x).float()
        self.points_y = torch.from_numpy(points_y).float()

    def to_scene_points(self,
                        rgbs: torch.Tensor,
                        depths: torch.Tensor,
                        include_rgb=True):
        batch_size = rgbs.shape[0]
        feature_len = 3 + 3 * include_rgb
        points_all = -torch.ones(
            (batch_size, self.all_points_num, feature_len),
            dtype=torch.float32).cuda()
        # cal z
        idxs = []
        masks = (depths > 0)
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        for i in range(batch_size):
            # convert point cloud to xyz maps
            points = torch.stack([cur_xs[i], cur_ys[i], cur_zs[i]], axis=-1)
            # remove zero depth
            mask = masks[i]
            points = points[mask]
            colors = rgbs[i][:, mask].T

            # random sample if points more than required
            if len(points) >= self.all_points_num:
                cur_idxs = random.sample(range(len(points)),
                                         self.all_points_num)
                points = points[cur_idxs]
                colors = colors[cur_idxs]
                # save idxs for concat fusion
                idxs.append(cur_idxs)

            # concat rgb and features after translation
            if include_rgb:
                points_all[i] = torch.concat([points, colors], axis=1)
            else:
                points_all[i] = points
        return points_all, idxs, masks

    def to_xyz_maps(self, depths):
        # convert xyzs
        cur_zs = depths / 1000.0
        cur_xs = self.points_x.cuda() * cur_zs
        cur_ys = self.points_y.cuda() * cur_zs
        xyzs = torch.stack([cur_xs, cur_ys, cur_zs], axis=-1)
        return xyzs.permute(0, 3, 1, 2)


def inference(view_points,
              rgbd,
              x,
              ori_rgb,
              ori_depth,
              use_heatmap=False,
              vis_heatmap=False,
              vis_grasp=True):
    with torch.no_grad():
        if use_heatmap:
            # 2d prediction
            pred_2d, _ = anchornet(x)

            loc_map, cls_mask, theta_offset, height_offset, width_offset = \
                anchor_output_process(*pred_2d, sigma=args.sigma)

            # detect 2d grasp (x, y, theta)
            rect_gg = detect_2d_grasp(loc_map,
                                      cls_mask,
                                      theta_offset,
                                      height_offset,
                                      width_offset,
                                      ratio=args.ratio,
                                      anchor_k=args.anchor_k,
                                      anchor_w=args.hggd_anchor_w,
                                      anchor_z=args.anchor_z,
                                      mask_thre=args.heatmap_thres,
                                      center_num=args.center_num,
                                      grid_size=args.grid_size,
                                      grasp_nms=args.grid_size,
                                      reduce='max')

            # check 2d result
            if rect_gg.size == 0:
                print('No 2d grasp found')
                return

            # show heatmap
            if vis_heatmap:
                rgb_t = x[0, 1:].cpu().numpy().squeeze().transpose(2, 1, 0)
                resized_rgb = Image.fromarray((rgb_t * 255.0).astype(np.uint8))
                resized_rgb = np.array(
                    resized_rgb.resize((args.input_w, args.input_h))) / 255.0
                depth_t = ori_depth.cpu().numpy().squeeze().T
                plt.subplot(131)
                plt.imshow(rgb_t)
                plt.subplot(132)
                plt.imshow(depth_t)
                plt.subplot(133)
                plt.imshow(loc_map.squeeze().T, cmap='jet')
                plt.tight_layout()
                plt.show()
        else:
            init_center = []
            grid_x = [0, 0, 0, 4, 4, 4, -4, -4, -4]
            grid_y = [0, 4, -4, 0, 4, -4, 0, 4, -4]
            ori_rgb = ori_rgb.squeeze().permute(2, 1, 0).cpu().numpy().astype(
                np.float32)
            ori_rgb = cv2.resize(ori_rgb, (640, 360),
                                 interpolation=cv2.INTER_AREA)
            window_name = 'click to choose target'

            def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
                if event == cv2.EVENT_LBUTTONDOWN:
                    xy = '%d,%d' % (x, y)
                    init_center.append([x, y])
                    for i in range(9):
                        init_center.append([x + grid_x[i], y + grid_y[i]])
                    cv2.circle(ori_rgb, (x, y), 7, (255, 0, 0), thickness=-1)
                    cv2.putText(ori_rgb,
                                xy, (x, y),
                                cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 0, 0),
                                thickness=1)
                    cv2.imshow(window_name,
                               cv2.cvtColor(ori_rgb, cv2.COLOR_RGB2BGR))
                    print(x, y)

            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(window_name, on_EVENT_LBUTTONDOWN)
            cv2.imshow(window_name, cv2.cvtColor(ori_rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            init_center = np.array(init_center)
            print('choice center list: ', init_center)

            rect_gg = RectGraspGroup(centers=init_center,
                                     heights=np.full((len(init_center), ), 25),
                                     depths=np.full((len(init_center), ), 0))

        # only need to convert to 3d centers
        valid_local_centers, _ = center2dtopc([rect_gg],
                                              args.center_num,
                                              ori_depth,
                                              (args.input_w, args.input_h),
                                              append_random_center=False,
                                              is_training=False)

        # seg local patches
        # using grid sample to downsample and get patches
        _, w, h = rgbd.shape
        # construct standard grid
        x = torch.linspace(0,
                           1,
                           args.patch_size,
                           device='cuda',
                           dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x, x)
        grid_idxs = torch.stack([grid_x, grid_y], -1) - 0.5  # centering
        # move to corresponding centers
        ratio = w / args.input_w
        centers_t = ratio * torch.from_numpy(rect_gg.centers).cuda()  # N, 2
        # calculate grid pos in original image
        grid_idxs = grid_idxs[None].expand(len(centers_t), -1, -1, -1)
        # adaptive radius
        intrinsics = get_camera_intrinsic()
        fx = intrinsics[0, 0]
        radius = torch.full((len(centers_t), ), 0.10, device='cuda')
        # radius = 0.06 * torch.rand(len(centers_t), device='cuda') + 0.06
        radius *= 2 * fx / valid_local_centers[0][:, 2]  # in ori image
        # fit to different grippers
        radius *= args.anchor_w / 60.0
        grid_idxs = grid_idxs * radius[:, None, None,
                                       None]  # B, S, S, 2 * B, 1, 1, 1
        # move to coresponding centers
        grid_idxs = grid_idxs + torch.flip(centers_t[:, None, None],
                                           [-1])  # B, S, S, 2 + B, 1, 1, 2
        # normalize to [-1, 1]
        grid_idxs = grid_idxs / torch.FloatTensor([(h - 1),
                                                   (w - 1)]).cuda() * 2 - 1
        local_patches = F.grid_sample(rgbd[None].expand(
            len(centers_t), -1, -1, -1),
                                      grid_idxs,
                                      mode='nearest',
                                      align_corners=False)
        local_patches = local_patches.permute(0, 3, 2, 1).contiguous()

        # norm space
        # move to (0, 0, 0)
        mask = (local_patches[..., -1:] > 0)
        patch_centers = valid_local_centers[0][:, None, None]
        patch_centers = patch_centers.expand(-1, args.patch_size,
                                             args.patch_size, -1)
        local_patches[..., 3:] -= mask * patch_centers
        local_patches[..., 3:] /= args.anchor_w / 1e3

        # get gamma and beta classification result
        _, pred, offset, theta_cls, theta_offset, width_reg = localnet(
            local_patches)
        theta_cls = theta_cls.sigmoid().clip(
            eps, 1 - eps).detach().cpu().numpy().squeeze()
        theta_offset = theta_offset.clip(-0.5,
                                         0.5).detach().cpu().numpy().squeeze()
        width_reg = width_reg.detach().cpu().numpy().squeeze()

        # get theta
        thetas, widths_6d = get_thetas_widths(theta_cls,
                                              theta_offset,
                                              width_reg,
                                              anchor_w=args.anchor_w,
                                              rotation_num=1)

        # detect 6d grasp from 2d output and 6d output
        pred_grasp, pred_6d_gg = detect_6d_grasp_multi(thetas,
                                                       widths_6d,
                                                       pred,
                                                       offset,
                                                       valid_local_centers,
                                                       anchors,
                                                       alpha=args.alpha *
                                                       args.anchor_w / 60.0,
                                                       k=args.local_k)

        # collision detect
        pred_gg, valid_mask = collision_detect(
            view_points[..., :3].squeeze(),  # batch_size == 1 when valid
            pred_6d_gg,
            mode='graspnet')
        pred_grasp = pred_grasp[valid_mask]

        # nms
        mask = (pred_gg.scores > 0.5)
        pred_gg = pred_gg[mask]
        pred_gg = pred_gg.nms()[:50]

        # show grasp
        if vis_grasp:
            print('pred grasp num ==', len(pred_gg))
            grasp_geo = pred_gg.to_open3d_geometry_list(scale=args.anchor_w /
                                                        60)
            points = view_points[..., :3].cpu().numpy().squeeze()
            colors = view_points[..., 3:6].cpu().numpy().squeeze()
            vispc = o3d.geometry.PointCloud()
            vispc.points = o3d.utility.Vector3dVector(points)
            vispc.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([vispc] + grasp_geo)
        return pred_gg


if __name__ == '__main__':
    # set up pc transform helper
    pc_helper = PointCloudHelper(all_points_num=args.all_points_num)

    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    # Init the model
    anchornet = AnchorGraspNet(in_dim=4,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k)
    localnet = PatchMultiGraspNet(args.anchor_num**2,
                                  theta_k_cls=6,
                                  feat_dim=args.embed_dim,
                                  anchor_w=args.anchor_w)
    x = torch.randn((48, 64, 64, 6), device='cuda')
    params_heat = sum(p.numel() for p in anchornet.parameters()
                      if p.requires_grad)
    print(f'Heatmap Model params == {params_heat}')
    macs, params = clever_format(profile(localnet.cuda(), inputs=(x, )),
                                 '%.3f')
    print(f'RNGNet macs == {macs}  params == {params}')

    # gpu
    anchornet = anchornet.cuda()
    localnet = localnet.cuda()

    # Load checkpoint
    check_point = torch.load(args.checkpoint_path)
    anchornet.load_state_dict(check_point['anchor'])
    localnet.load_state_dict(check_point['local'])
    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = check_point['gamma']
    anchors['beta'] = check_point['beta']
    print('Using saved anchors')
    print('-> loaded checkpoint %s ' % (args.checkpoint_path))

    # network eval mode
    anchornet.eval()
    localnet.eval()

    # read image and conver to tensor
    ori_depth = np.array(Image.open(args.depth_path)).astype(np.float32)
    ori_rgb = np.array(Image.open(args.rgb_path)) / 255.0
    ori_depth = np.clip(ori_depth, 0, 1000)
    ori_rgb = torch.from_numpy(ori_rgb).permute(2, 1, 0)[None]
    ori_rgb = ori_rgb.to(device='cuda', dtype=torch.float32)
    ori_depth = torch.from_numpy(ori_depth).T[None]
    ori_depth = ori_depth.to(device='cuda', dtype=torch.float32)

    # get scene points
    view_points, masks, idxs = pc_helper.to_scene_points(ori_rgb,
                                                         ori_depth,
                                                         include_rgb=True)
    points = view_points[..., :3]
    view_points = view_points.squeeze()
    # get xyz maps
    xyzs = pc_helper.to_xyz_maps(ori_depth)
    rgbd = torch.cat([ori_rgb.squeeze(), xyzs.squeeze()], 0)

    # pre-process
    rgb = F.interpolate(ori_rgb, (args.input_w, args.input_h))
    depth = F.interpolate(ori_depth[None], (args.input_w, args.input_h))[0]
    depth = depth / 1000.0
    depth = torch.clip((depth - depth.mean()), -1, 1)
    # generate 2d input
    x = torch.concat([depth[None], rgb], 1)
    x = x.to(device='cuda', dtype=torch.float32)

    # inference
    pred_gg = inference(view_points,
                        rgbd,
                        x,
                        ori_rgb,
                        ori_depth,
                        use_heatmap=True,
                        vis_heatmap=True,
                        vis_grasp=True)

    # time test
    start = time()
    T = 100
    for _ in range(T):
        pred_gg = inference(view_points,
                            rgbd,
                            x,
                            ori_depth,
                            ori_depth,
                            use_heatmap=True,
                            vis_heatmap=False,
                            vis_grasp=False)
        torch.cuda.synchronize()
    print('avg time ==', (time() - start) / T * 1e3, 'ms')
