import argparse
import os
import random
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm
from thop import clever_format, profile

from customgraspnetAPI import GraspGroup as GraspNetGraspGroup
from customgraspnetAPI import GraspNetEval
from dataset.config import camera, get_camera_intrinsic
from dataset.evaluation import (anchor_output_process, collision_detect,
                                sample_2d_centers, detect_6d_grasp_multi,
                                get_thetas_widths)
from dataset.graspnet_dataset import GraspnetPointDataset
from dataset.pc_dataset_tools import center2dtopc
from models.anchornet import AnchorGraspNet
from models.localgraspnet import PatchMultiGraspNet
from train_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None)

# dataset
parser.add_argument('--dataset-path')
parser.add_argument('--scene-path')
parser.add_argument('--scene-l', type=int)
parser.add_argument('--scene-r', type=int)
parser.add_argument('--grasp-count', type=int, default=5000)
parser.add_argument('--dump-dir',
                    help='Dump dir to save outputs',
                    default='./pred/test')
parser.add_argument('--num-workers', type=int, default=4)

# hggd
parser.add_argument('--feature-dim',
                    type=int,
                    default=128,
                    help='Feature dim for anchornet')
parser.add_argument('--input-h', type=int, default=360)
parser.add_argument('--input-w', type=int, default=640)
parser.add_argument('--sigma', type=int, default=10)
parser.add_argument('--ratio', type=int, default=8)
parser.add_argument('--anchor-k', type=int, default=6)
parser.add_argument('--anchor-w-hggd', type=float, default=0.06)
parser.add_argument('--anchor-z', type=float, default=20.0)
parser.add_argument('--grid-size', type=float, default=0.01)
parser.add_argument('--min-dis', type=float, default=0.01)

# net
parser.add_argument('--embed-dim', type=int)
parser.add_argument('--anchor-w', type=float, default=100.0)
parser.add_argument('--anchor-num', type=int)

# patch
parser.add_argument('--patch-size',
                    type=int,
                    default=64,
                    help='local patch grid size')
parser.add_argument('--alpha',
                    type=float,
                    default=0.02,
                    help='grasp center crop range')

# pc
parser.add_argument('--all-points-num', type=int, default=25600)
parser.add_argument('--center-num', type=int)
parser.add_argument('--group-num', type=int, default=512)
parser.add_argument('--local-grasp-num', type=int, default=1000)

# grasp detection
parser.add_argument('--collision-detect',
                    action='store_true',
                    help='wheter use collision detection')
parser.add_argument('--heatmap-thres', type=float, default=0.01)
parser.add_argument('--local-k', type=int, default=10)
parser.add_argument('--local-thres', type=float, default=0.01)
parser.add_argument('--rotation-num', type=int, default=1)

# evaluate
parser.add_argument(
    '--multi-scale',
    action='store_true',
    help=
    'whether to conduct evaluation on different sizes as scaled-balanced grasp'
)

# others
parser.add_argument('--logdir',
                    type=str,
                    default='./logs/',
                    help='Log directory')
parser.add_argument('--random-seed', type=int, default=123, help='Random seed')
parser.add_argument('--description',
                    type=str,
                    default='',
                    help='Logging description')

args = parser.parse_args()

# camera-invariant args
# assume z == 0.4 m
z = 0.4
fx = get_camera_intrinsic()[0, 0]
scale_x = 1280 // args.input_w
args.anchor_w_hggd = np.ceil(args.anchor_w_hggd * fx / z / scale_x)
args.grid_size = np.ceil(args.grid_size * fx / z / scale_x)
args.min_dis = np.ceil(args.min_dis * fx / z / scale_x)
# adjust to mask, w % 5 == 0, grid_size is int
args.grid_size = int(args.grid_size)
args.min_dis = int(args.min_dis)
args.anchor_w_hggd = (args.anchor_w_hggd // 5 + 1) * 5

eps = 1e-6


def inference():
    sceneIds = list(range(args.scene_l, args.scene_r))
    # Create Dataset and Dataloader
    test_dataset = GraspnetPointDataset(args.all_points_num,
                                        args.dataset_path,
                                        args.scene_path,
                                        sceneIds,
                                        sigma=args.sigma,
                                        ratio=args.ratio,
                                        anchor_k=args.anchor_k,
                                        anchor_z=args.anchor_z,
                                        anchor_w=args.anchor_w_hggd,
                                        grasp_count=args.grasp_count,
                                        output_size=(args.input_w,
                                                     args.input_h),
                                        random_rotate=False,
                                        random_zoom=False)

    SCENE_LIST = test_dataset.scene_list()
    test_data = DataLoader(test_dataset,
                           batch_size=1,
                           pin_memory=True,
                           num_workers=4)

    # stop rot and zoom for validation
    test_data.dataset.unaug()
    test_data.dataset.eval()

    # Init the model
    input_channels = 4
    anchornet = AnchorGraspNet(in_dim=input_channels,
                               ratio=args.ratio,
                               anchor_k=args.anchor_k,
                               feature_dim=args.feature_dim)

    localnet = PatchMultiGraspNet(args.anchor_num**2,
                                  theta_k_cls=6,
                                  feat_dim=args.embed_dim,
                                  anchor_w=args.anchor_w)

    x = torch.randn((48, 64, 64, 6), device='cuda')
    macs, params = clever_format(profile(localnet.cuda(), inputs=(x, )),
                                 '%.3f')
    logging.info(f'macs == {macs}  params == {params}')

    localnet_params = sum(p.numel() for p in localnet.parameters())
    logging.info(f'localnet paras == {localnet_params}')

    # cuda
    anchornet = anchornet.cuda()
    localnet = localnet.cuda()

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    anchornet.load_state_dict(checkpoint['anchor'])
    localnet.load_state_dict(checkpoint['local'])

    # load checkpoint
    basic_ranges = torch.linspace(-1, 1, args.anchor_num + 1).cuda()
    basic_anchors = (basic_ranges[1:] + basic_ranges[:-1]) / 2
    anchors = {'gamma': basic_anchors, 'beta': basic_anchors}
    anchors['gamma'] = checkpoint['gamma']
    anchors['beta'] = checkpoint['beta']
    logging.info('Using saved anchors')
    logging.info('-> loaded checkpoint %s ' % (args.checkpoint))
    # network eval mode
    anchornet.eval()
    localnet.eval()

    anchor_ws = np.array([60])
    batch_idx = -1
    sum_grasp_num = 0
    sum_grasp_width = 0
    with torch.no_grad():
        for anchor_data, rgb, depth, grasppaths in tqdm(test_data,
                                                        desc='test',
                                                        ncols=80):
            batch_idx += 1

            rgb, depth = rgb.cuda(), depth.cuda()
            # get scene points
            view_points, masks, idxs = test_data.dataset.helper.to_scene_points(
                rgb, depth, include_rgb=True)
            points = view_points[..., :3]
            view_points = view_points.squeeze()
            # get xyz maps
            xyzs = test_data.dataset.helper.to_xyz_maps(depth)
            rgbd = torch.cat([rgb.squeeze(), xyzs.squeeze()], 0)

            # 2d prediction
            x, _, _, _, _ = anchor_data
            x = x.to(device='cuda', dtype=torch.float32)
            pred_2d, _ = anchornet(x)
            # sample 2d grasp centers
            loc_map = anchor_output_process(*pred_2d, sigma=args.sigma)[0]
            rect_gg = sample_2d_centers(loc_map, args.grid_size,
                                        args.center_num, args.grid_size)

            # check 2d result
            if rect_gg.size == 0:
                print('No 2d grasp found')
                continue

            # seg local pcs
            total_gg = GraspNetGraspGroup()
            # iteration different scale of width and offset
            for anchor_w in anchor_ws:
                # get patch
                rect_gg.depths = np.zeros(rect_gg.size)
                # only need to convert to 3d centers
                valid_local_centers, _ = center2dtopc(
                    [rect_gg],
                    args.center_num,
                    depth, (args.input_w, args.input_h),
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
                grid_idxs = torch.stack([grid_x, grid_y],
                                        -1) - 0.5  # centering
                # move to corresponding centers
                ratio = w / args.input_w
                centers_t = ratio * torch.from_numpy(
                    rect_gg.centers).cuda()  # N, 2
                # calculate grid pos in original image
                grid_idxs = grid_idxs[None].expand(len(centers_t), -1, -1, -1)
                # adaptive radius
                intrinsics = get_camera_intrinsic()
                fx = intrinsics[0, 0]
                radius = torch.full((len(centers_t), ), 0.10, device='cuda')
                radius *= anchor_w / args.anchor_w
                radius *= 2 * fx / valid_local_centers[0][:, 2]  # in ori image
                grid_idxs = grid_idxs * radius[:, None, None,
                                               None]  # B, S, S, 2 * B, 1, 1, 1
                # move to coresponding centers
                grid_idxs = grid_idxs + torch.flip(
                    centers_t[:, None, None], [-1])  # B, S, S, 2 + B, 1, 1, 2
                # normalize to [-1, 1]
                grid_idxs = grid_idxs / torch.FloatTensor([(h - 1), (w - 1)
                                                           ]).cuda() * 2 - 1
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
                local_patches[..., 3:] /= anchor_w / 1e3

                # get gamma and beta classification result
                features, pred, offset, theta_cls, theta_offset, width_reg = localnet(
                    local_patches)

                theta_cls = theta_cls.sigmoid().clip(
                    eps, 1 - eps).detach().cpu().numpy().squeeze()
                theta_offset = theta_offset.clip(
                    -0.5, 0.5).detach().cpu().numpy().squeeze()
                width_reg = width_reg.detach().cpu().numpy().squeeze()

                # get theta
                thetas, widths_6d = get_thetas_widths(theta_cls,
                                                      theta_offset,
                                                      width_reg,
                                                      anchor_w=anchor_w,
                                                      rotation_num=1)

                # detect 6d grasp from 2d output and 6d output
                pred_grasp, pred_6d_gg = detect_6d_grasp_multi(
                    thetas,
                    widths_6d,
                    pred,
                    offset,
                    valid_local_centers,
                    anchors,
                    alpha=args.alpha * anchor_w / args.anchor_w,
                    k=args.local_k)

                # collision detect
                if args.collision_detect:
                    pred_gg, valid_mask = collision_detect(
                        points.squeeze(),  # batch_size == 1 when valid
                        pred_6d_gg,
                        mode='graspnet')
                    pred_grasp = pred_grasp[valid_mask]
                else:
                    pred_gg = pred_6d_gg

                # Dump results for evaluation
                gg = GraspNetGraspGroup(np.zeros((len(pred_gg), 17)))
                gg.scores = pred_gg.scores
                gg.widths = pred_gg.widths
                gg.heights = pred_gg.heights / args.alpha * 0.02
                gg.depths = pred_gg.depths / args.alpha * 0.02
                gg.rotation_matrices = pred_gg.rotations
                gg.translations = pred_gg.translations
                gg.object_ids = np.full((len(pred_gg), ), -1)
                total_gg.add(gg)

            sum_grasp_num += len(total_gg)
            sum_grasp_width += total_gg.widths.sum()
            if batch_idx % 256 == 255:
                sum_grasp_width /= sum_grasp_num
                sum_grasp_num /= 256
                if batch_idx == len(test_data) - 1:
                    logging.info(f'Scene: {args.scene_r - 1}:')
                else:
                    logging.info(f'Scene: {batch_idx // 256 + args.scene_l}:')
                logging.info(f'avg grasp num: {sum_grasp_num:.3f}')
                logging.info(f'avg grasp width: {sum_grasp_width:.3f}')
                sum_grasp_num = 0
                sum_grasp_width = 0

            # save grasps
            save_dir = os.path.join(args.dump_dir, SCENE_LIST[batch_idx])
            save_dir = os.path.join(save_dir, camera)
            save_path = os.path.join(save_dir,
                                     str(batch_idx % 256).zfill(4) + '.npy')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            total_gg.save_npy(save_path)


def evaluate(save_folder, multi_scale=False):
    ge = GraspNetEval(root=args.scene_path,
                      camera=camera,
                      split=(args.scene_l, args.scene_r))
    if multi_scale:
        # eval different scales
        for scale in ['small', 'medium', 'large']:
            logging.info(f'Current evaluating scale: {scale}')
            res, ap, colli = ge.eval_scene_lr(args.dump_dir,
                                              args.scene_l,
                                              args.scene_r,
                                              scale=scale,
                                              proc=args.num_workers)
            np.save(os.path.join(save_folder, f'temp_result_{scale}.npy'), res)
            # get ap 0.8 and ap 0.4
            aps = res.mean(0).mean(0).mean(0)
            logging.info(f'colli == {colli}')
            logging.info(f'ap == {ap}  ap0.8 == {aps[3]}  ap0.4 == {aps[1]}')
        # eval all scale
        res, ap, colli = ge.eval_scene_lr(args.dump_dir,
                                          args.scene_l,
                                          args.scene_r,
                                          proc=args.num_workers)
        np.save(os.path.join(save_folder, f'temp_result.npy'), res)
        # get ap 0.8 and ap 0.4
        aps = res.mean(0).mean(0).mean(0)
        logging.info(f'Scene: {args.scene_l} ~ {args.scene_r}')
        logging.info(f'colli == {colli}')
        logging.info(f'ap == {ap}  ap0.8 == {aps[3]}  ap0.4 == {aps[1]}')
    else:
        res, ap, colli = ge.eval_scene_lr(args.dump_dir,
                                          args.scene_l,
                                          args.scene_r,
                                          proc=args.num_workers)
        np.save(os.path.join(save_folder, 'temp_result.npy'), res)
        # get ap 0.8 and ap 0.4
        aps = res.mean(0).mean(0).mean(0)
        logging.info(f'Scene: {args.scene_l} ~ {args.scene_r}')
        logging.info(f'colli == {colli}')
        logging.info(f'ap == {ap}')
        logging.info(f'ap0.8 == {aps[3]}')
        logging.info(f'ap0.4 == {aps[1]}')
        # log results for each part
        if args.scene_l == 100 and args.scene_r == 190:
            scene_aps = res.mean(-1).mean(-1).mean(-1)
            logging.info(f'seen == {scene_aps[:30].mean()}')
            logging.info(f'similar == {scene_aps[30:60].mean()}')
            logging.info(f'novel == {scene_aps[60:].mean()}')


if __name__ == '__main__':
    # multiprocess
    # mp.set_start_method('spawn')
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

    # Set-up output directories
    net_desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = net_desc + '_' + args.description
    net_desc = 'graspnet_test' + net_desc

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Save commandline args
    if args is not None:
        params_path = os.path.join(save_folder, 'commandline_args.json')
        with open(params_path, 'w') as f:
            json.dump(vars(args), f)

    # Initialize logging
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        filename='{0}/{1}.log'.format(save_folder, 'log'),
        format=
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S')
    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    inference()
    evaluate(save_folder, multi_scale=args.multi_scale)
