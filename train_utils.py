import argparse
import datetime
import json
import logging
import os
import random
import sys

import numpy as np
import tensorboardX
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torchsummary import summary

from customgraspnetAPI import Grasp, GraspGroup
from dataset.config import get_camera_intrinsic

eval_scale = np.linspace(0.2, 1, 5)


def log_acc_str(name, T, F):
    T, F = int(T), int(F)
    if T + F == 0:
        return f'{name} 0/0 = 0'
    return f'{name} {T}/{T + F} = {T / (T + F):.3f}'


def prepare_torch(args):
    # multiprocess
    # mp.set_start_method('spawn')
    # set torch and gpu setting
    np.set_printoptions(precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    torch.set_num_threads(8)
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
    else:
        raise RuntimeError('CUDA not available')

    # random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


def prepare_logger(args, mode='train'):
    # Set-up output directories
    net_desc = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    net_desc = net_desc + '_' + args.description
    if mode == 'test':
        net_desc = 'test' + net_desc

    save_folder = os.path.join(args.logdir, net_desc)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    tb = tensorboardX.SummaryWriter(save_folder)

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

    return tb, save_folder


def get_optimizer(args, params):
    # get optimizer
    if args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    elif args.optim.lower() == 'sgd':
        optimizer = optim.SGD(params,
                              lr=args.lr,
                              momentum=0.9,
                              weight_decay=2e-4)
    else:
        raise NotImplementedError('Optimizer {} is not implemented'.format(
            args.optim))
    return optimizer


def print_model(args, input_channels, model, save_folder):
    summary(model, (input_channels, args.input_w, args.input_h), device='cpu')
    with open(os.path.join(save_folder, 'arch.txt'), 'w') as f:
        sys.stdout = f
        summary(model, (input_channels, args.input_w, args.input_h),
                device='cpu')
        sys.stdout = sys.__stdout__


def log_match_result(results, dis_criterion, rot_criterion):
    for scale_factor in eval_scale:
        # get threshold from criterion and factor
        thre_dis = dis_criterion * scale_factor
        thre_rot = rot_criterion * scale_factor

        t_trans, f_trans = results[f'trans_{thre_dis}']
        t_rot, f_rot = results[f'rot_{thre_rot}']
        t_grasp, f_grasp = results[f'grasp_{scale_factor}']

        t_str = log_acc_str(f'trans_{thre_dis:.2f}', t_trans, f_trans)
        r_str = log_acc_str(f'rot_{thre_rot:.2f}', t_rot, f_rot)
        g_str = log_acc_str(f'grasp_{scale_factor:.2f}', t_grasp, f_grasp)

        logging.info(f'{t_str}  {r_str}  {g_str}')


def log_and_save(args,
                 tb,
                 results,
                 epoch,
                 anchornet,
                 localnet,
                 optimizer,
                 anchors,
                 save_folder,
                 mode='regnet'):
    # Log validation results to tensorbaord
    # loss
    tb.add_scalar('val_loss/loss', results['loss'], epoch)
    tb.add_scalar('val_loss/theta_loss', results['theta_loss'], epoch)
    for n, l in results['losses'].items():
        tb.add_scalar('val_loss/' + n, l, epoch)

    logging.info('Validation Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'theta loss: {results["theta_loss"]:.3f}')
    if 'cls_loss' in results['losses']:
        logging.info(
            f'reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
        )
        logging.info(f'width: {results["losses"]["width_loss"]:.3f}')
    tb.add_scalar('val_loss/multi_cls_loss', results['multi_cls_loss'], epoch)
    tb.add_scalar('val_loss/offset_loss', results['offset_loss'], epoch)
    logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')
    logging.info(f'offset_loss: {results["offset_loss"]:.3f}')

    # coverage
    cover_cnt = results['cover_cnt']
    label_cnt = results['label_cnt']
    tb.add_scalar('coverage', cover_cnt / label_cnt, epoch)
    logging.info(
        f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
    )

    # 2d iou
    if results['total'] > 0:
        iou = results['correct'] / results['total']
        tb.add_scalar('IOU', iou, epoch)
        logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if mode == 'regnet':
        view_num = results['grasp_nocoll_view_num']
        vgr = results['vgr']
        score = results['score']
        if view_num > 0:
            tb.add_scalar('collision_free_ratio', vgr / view_num, epoch)
            tb.add_scalar('score', score / view_num, epoch)

            logging.info('REGNet validation:')
            logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
            logging.info(
                f'score: {score:.3f} / {view_num} = {score / view_num:.3f}')
        else:
            logging.info('No collision-free grasp')
    elif mode == 'graspnet':
        logging.info('please run test_graspnet.py for graspnet result')

    # Save best performing network
    if epoch % args.save_freq == 0 and optimizer is not None:
        if mode == 'regnet':
            torch.save(
                {
                    'anchor': anchornet.module.state_dict(),
                    'local': localnet.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_score_{score / view_num:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))
        elif mode == 'graspnet':
            t_grasp, f_grasp = results['grasp_0.2']
            acc = t_grasp / (t_grasp + f_grasp)
            tb.add_scalar('score', acc, epoch)
            torch.save(
                {
                    'anchor': anchornet.module.state_dict(),
                    'local': localnet.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'gamma': anchors['gamma'],
                    'beta': anchors['beta']
                },
                os.path.join(
                    save_folder,
                    f'epoch_{epoch}_acc_{acc:.3f}_cover_{cover_cnt / label_cnt:.3f}'
                ))


def log_test_result(args, results, epoch, mode='regnet'):
    # Log validation results to tensorbaord
    # loss
    logging.info('Test Loss:')
    logging.info(f'test loss: {results["loss"]:.3f}')
    logging.info(f'anchor loss: {results["theta_loss"]:.3f}')
    logging.info(
        f'reg: {results["losses"]["reg_loss"]:.3f}, cls: {results["losses"]["cls_loss"]:.3f}'
    )
    logging.info(f'multicls_loss: {results["multi_cls_loss"]:.3f}')

    # coverage
    cover_cnt = results['cover_cnt']
    label_cnt = results['label_cnt']
    logging.info(
        f'coverage rate: {cover_cnt} / {label_cnt} = {cover_cnt / label_cnt:.3f}'
    )

    # 2d iou
    iou = results['correct'] / (results['correct'] + results['failed'])
    logging.info(f'2d iou: {iou:.2f}')

    # regnet validation
    if mode == 'regnet':
        view_num = results['grasp_nocoll_view_num']
        vgr = results['vgr']
        score = results['score']
        if view_num > 0:
            logging.info('REGNet validation:')
            logging.info(f'vgr: {vgr} / {view_num} = {vgr / view_num:.3f}')
            logging.info(
                f'score: {score:.3f} / {view_num} = {score / view_num:.3f}')
        else:
            logging.info('No collision-free grasp')
    elif mode == 'graspnet':
        logging.info('please run test_graspnet.py for graspnet result')


def log_theta_loss(epoch, batch_idx, loss, theta_loss, theta_losses,
                   batch_cnt):
    logging.info('Epoch: {}, Batch: {}, total_loss: {:0.4f}'.format(
        epoch, batch_idx, loss / batch_cnt))
    logging.info('theta_loss: {:0.4f}'.format(theta_loss / batch_cnt))
    logging.info(
        'reg_loss: {:0.4f}, cls_loss: {:0.4f}, width_loss: {:0.4f}'.format(
            theta_losses['reg_loss'] / batch_cnt,
            theta_losses['cls_loss'] / batch_cnt,
            theta_losses['width_loss'] / batch_cnt))


def dump_grasp(epoch, batch_idx, pred_gg, scene_list, dump_dir='./pred'):
    gg = GraspGroup()
    for g in pred_gg:
        g = Grasp(1, g.width, 0.02, 0.02, g.rotation.reshape(9, ),
                  g.translation, -1)
        gg.add(g)

    # save grasps
    save_dir = os.path.join(dump_dir, f'epoch_{epoch}')
    save_dir = os.path.join(save_dir, scene_list[batch_idx])
    save_dir = os.path.join(save_dir, 'realsense')
    save_path = os.path.join(save_dir, str(batch_idx % 256).zfill(4) + '.npy')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gg.save_npy(save_path)
