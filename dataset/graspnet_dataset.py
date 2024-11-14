import logging
import os

import cupoch
import cv2
import numpy as np
from PIL import Image
from torch.utils.dlpack import from_dlpack, to_dlpack
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    PILToTensor, ToPILImage)
from tqdm import tqdm

from .base_grasp_dataset import GraspDataset
from .config import camera
from .utils import PointCloudHelper, fast_sample


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print('Showing outliers (red) and inliers (gray): ')
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    cupoch.visualization.draw_geometries([inlier_cloud, outlier_cloud])


class GraspnetAnchorDataset(GraspDataset):

    def __init__(self,
                 labelroot,
                 graspnetroot,
                 fullpc_path,
                 gtdetph_path,
                 local_data_path,
                 sceneIds,
                 ratio,
                 anchor_k,
                 anchor_z,
                 anchor_w,
                 grasp_count,
                 sigma=10,
                 min_dis=8,
                 random_rotate=False,
                 random_zoom=False,
                 output_size=(640, 360),
                 include_rgb=True,
                 include_depth=True,
                 noise=0,
                 aug=False,
                 view_num=256):
        logging.info('Using Graspnet dataset')
        # basic attributes
        self.trainning = True
        self.labelroot = labelroot
        self.graspnetroot = graspnetroot
        self.fullpc_path = fullpc_path
        self.gtdepth_path = gtdetph_path
        self.local_data_path = local_data_path
        self.sceneIds = sceneIds
        self.random_rotate = random_rotate
        self.random_zoom = random_zoom
        self.include_rgb = include_rgb
        self.include_depth = include_depth
        self.viewnum = view_num
        # anchor size
        self.ratio = ratio
        self.min_dis = min_dis
        self.anchor_k = anchor_k
        self.anchor_z = anchor_z
        self.anchor_w = anchor_w
        # grasp count
        self.grasp_count = grasp_count
        # gaussian kernel size
        self.sigma = sigma

        self.colorpath = []
        self.depthpath = []
        self.cameraposepath = []
        self.alignmatpath = []
        self.metapath = []
        self.grasppath = []
        self.sceneIds_str = ['scene_{}'.format(str(x)) for x in self.sceneIds]
        self.sceneIds_zfill = [
            'scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds
        ]
        self.scenename = []
        self.frameid = []
        self.output_size = output_size
        self.noise = noise
        self.aug = None
        if aug:
            self.aug = Compose(
                [PILToTensor(),
                 ColorJitter(0.5, 0.5, 0.5, 0.3),
                 ToPILImage()])

        for x in tqdm(self.sceneIds_zfill, desc='Loading data path...'):
            self.cameraposepath.append(
                os.path.join(graspnetroot, 'scenes', x, camera,
                             'camera_poses.npy'))
            self.alignmatpath.append(
                os.path.join(graspnetroot, 'scenes', x, camera,
                             'cam0_wrt_table.npy'))
            for img_num in range(self.viewnum):
                self.colorpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'rgb',
                                 str(img_num).zfill(4) + '.png'))
                self.depthpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'depth',
                                 str(img_num).zfill(4) + '.png'))
                self.metapath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'meta',
                                 str(img_num).zfill(4) + '.mat'))
                self.scenename.append(x.strip())
                self.frameid.append(img_num)

        for x in tqdm(self.sceneIds_str,
                      desc='Loading 6d grasp label path...'):
            for ann_num in range(self.viewnum):
                self.grasppath.append(
                    os.path.join(labelroot, '6d_dataset', x, 'grasp_labels',
                                 '{}_view.npz'.format(ann_num)))


class GraspnetPointDataset(GraspnetAnchorDataset):

    def __init__(self, all_points_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # for pc convert
        self.all_points_num = all_points_num
        self.helper = PointCloudHelper(self.all_points_num)

    def get_camera_pose(self, index):
        camera_pose = np.load(self.cameraposepath[index // self.viewnum])
        align_mat = np.load(self.alignmatpath[index // self.viewnum])
        return align_mat @ camera_pose[index % self.viewnum]

    def __getitem__(self, index):
        # get anchor data
        anchor_data = super().__getitem__(index)

        # load image
        rgb = self.cur_rgb.astype(np.float32) / 255.0
        depth = self.cur_depth.astype(np.float32)

        cur_scene = self.sceneIds[0] + index // self.viewnum
        cur_ann = index % self.viewnum
        # get gt_pcs
        camera_pose = self.get_camera_pose(index)
        fullpc = np.load(
            os.path.join(self.fullpc_path, f'pc_{cur_scene}_world.npy'))
        fullpc = np.concatenate([fullpc, np.ones((fullpc.shape[0], 1))], 1)
        fullpc = (np.linalg.inv(camera_pose) @ fullpc.T).T
        idxs = fast_sample(fullpc.shape[0], self.all_points_num)
        fullpc = fullpc[idxs, :3]

        # get full_depth in camera frame
        gt_depth = cv2.imread(
            os.path.join(self.gtdepth_path,
                         f'scene_{cur_scene}/depth_{cur_ann}.png'),
            cv2.IMREAD_UNCHANGED)
        gt_depth = np.array(gt_depth, np.float32).T

        # get grasp path
        grasp_path = self.grasppath[index]
        return anchor_data, rgb, depth, grasp_path, gt_depth, fullpc


class GraspnetLocalDataset(GraspnetAnchorDataset):

    def __init__(self, all_points_num, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.all_points_num = all_points_num
        self.helper = PointCloudHelper(self.all_points_num)

        self.localdatapath = []
        self.collisionpath = []

        for x in tqdm(self.sceneIds_str, desc='Loading local data path...'):
            for img_num in range(self.viewnum):
                self.localdatapath.append(
                    os.path.join(self.local_data_path, x,
                                 str(img_num) + '.npz'))
                # self.localdatapath.append(
                #     os.path.join('/data1/robotics/local_data_randomwd_gth', x,
                #                  str(img_num) + '.npz'))
                # self.collisionpath.append(
                #     os.path.join(self.local_data_path, 'collision_labels', x,
                #                  f'{img_num}_collision_label.npz'))

    def __getitem__(self, index):

        local_path = self.localdatapath[index]
        # collision_path = self.collisionpath[index]

        # anchor_data = super().__getitem__(index)
        # color_img = self.cur_rgb.astype(np.float32) / 255.0
        # depth_img = self.cur_depth.astype(np.float32)
        # grasp_path = self.grasppath[index]
        return local_path  #, anchor_data, color_img, depth_img, grasp_path


class PartGraspDataset():

    def __init__(self,
                 graspnetroot,
                 local_data_path,
                 sceneIds,
                 all_points_num,
                 view_interval=1):
        self.local_data_path = local_data_path
        self.all_points_num = all_points_num
        self.view_interval = view_interval
        self.view_num = 256 // view_interval
        self.sceneIds = sceneIds
        self.sceneIds_str = ['scene_{}'.format(str(x)) for x in self.sceneIds]
        self.sceneIds_zfill = [
            'scene_{}'.format(str(x).zfill(4)) for x in self.sceneIds
        ]
        self.helper = PointCloudHelper(self.all_points_num)

        self.localdatapath = []
        self.colorpath = []
        self.depthpath = []
        self.maskpath = []
        self.scenename = []

        for x in tqdm(self.sceneIds_str, desc='Loading local data path...'):
            for img_num in range(0, 256, view_interval):
                print(img_num)
                self.localdatapath.append(
                    os.path.join(self.local_data_path, x,
                                 str(img_num) + '.npz'))

        for x in tqdm(self.sceneIds_zfill, desc='Loading data path...'):
            for img_num in range(0, 256, view_interval):
                self.colorpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'rgb',
                                 str(img_num).zfill(4) + '.png'))
                self.depthpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'depth',
                                 str(img_num).zfill(4) + '.png'))
                self.maskpath.append(
                    os.path.join(graspnetroot, 'scenes', x, camera, 'label',
                                 str(img_num).zfill(4) + '.png'))
                self.scenename.append(x.strip())

    def get_camera_pose(self, index):
        camera_pose = np.load(self.cameraposepath[index // self.view_num])
        align_mat = np.load(self.alignmatpath[index // self.view_num])
        return align_mat @ camera_pose[index % self.view_num]

    def scene_list(self):
        return self.scenename

    def __len__(self):
        return len(self.localdatapath)

    def __getitem__(self, index):

        # load image
        depth = Image.open(self.depthpath[index])
        depth = np.array(depth, dtype=np.float32).T

        rgb = Image.open(self.colorpath[index])
        rgb = np.array(rgb).transpose(2, 1, 0).astype(np.float32) / 255.0

        mask = cv2.imread(self.maskpath[index], cv2.IMREAD_UNCHANGED)
        maskx = np.any(mask, axis=0)
        masky = np.any(mask, axis=1)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1])
        workspace = np.array([x1, y1, x2, y2])

        local_path = self.localdatapath[index]
        local_data = dict(np.load(local_path, allow_pickle=True))
        return index, rgb, depth, workspace, local_data
