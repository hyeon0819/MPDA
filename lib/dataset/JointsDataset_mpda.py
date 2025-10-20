# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import copy
import logging

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import os

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale
from utils.generate_heatmap import check_occlusion_camera, occlusion_level

logger = logging.getLogger(__name__)


class JointsDataset_mpda(Dataset):

    def __init__(self, cfg, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = 0
        self.pixel_std = 200
        self.flip_pairs = []
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        self.num_views = cfg.DATASET.CAMERA_NUM
        # self.cam_sim = cfg.DATASET.CAM_SIMILARITY
        self.data_num = cfg.DATASET.DATA_NUM
        self.aug_ratio = cfg.DATASET.AUG_RATIO
        self.aug_type = cfg.DATASET.AUG_TYPE
        self.heatmap_generation = cfg.DATASET.HEATMAP_GENERATION

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP
        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)


    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_file = db_rec['image']

        # if self.data_format == 'zip':
        #     from utils import zipreader
        #     data_numpy = zipreader.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # else:
        #     data_numpy = cv2.imread(
        #         image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # if data_numpy is None:
        #     # logger.error('=> fail to read {}'.format(image_file))
        #     # raise ValueError('Fail to read {}'.format(image_file))
        #     return None, None, None, None, None, None

        # if self.color_rgb:
        #     data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

        joints = db_rec['joints_2d']
        joints_vis = db_rec['joints_2d_vis']
        joints_3d = db_rec['joints_3d']
        joints_3d_vis = db_rec['joints_3d_vis']
        
        camera_t = db_rec['camera']['T'].reshape(3)
        joints_occ = check_occlusion_camera(joints_3d, camera_t)
        joints_occ_level = occlusion_level(joints_occ)
        # print('joints_occ', joints_occ)
        # print('joints_occ_level', joints_occ_level)

        nposes = len(joints)
        assert nposes <= self.maximum_person, 'too many persons'

        height, width = 1080, 1920
        c = np.array([width / 2.0, height / 2.0])
        s = get_scale((width, height), self.image_size)
        r = 0

        trans = get_affine_transform(c, s, r, self.image_size)
        # input = cv2.warpAffine(
        #     data_numpy,
        #     trans, (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)

        # if self.transform:
        #     input = self.transform(input)

        # resize한 image size에 맞게 2D 좌표 조정
        for n in range(nposes):     
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        # if 'pred_pose2d' in db_rec and db_rec['pred_pose2d'] != None:
        #     # For convenience, we use predicted poses and corresponding values at the original heatmaps
        #     # to generate 2d heatmaps for Campus and Shelf dataset.
        #     # You can also use other 2d backbone trained on COCO to generate 2d heatmaps directly.
        #     pred_pose2d = db_rec['pred_pose2d']
        #     for n in range(len(pred_pose2d)):
        #         for i in range(len(pred_pose2d[n])):
        #             pred_pose2d[n][i, 0:2] = affine_transform(pred_pose2d[n][i, 0:2], trans)

        #     input_heatmap = self.generate_input_heatmap(pred_pose2d)
        #     input_heatmap = torch.from_numpy(input_heatmap)
        # else:
        #     input_heatmap = torch.zeros(self.cfg.NETWORK.NUM_JOINTS, self.heatmap_size[1], self.heatmap_size[0])
        
        if self.heatmap_generation == 'target':
            # occlusion 고려 X
            input_heatmap = self.generate_target_heatmap(joints, joints_vis)[0]
        elif self.heatmap_generation == 'input_occ_level':
            # occlusion 고려 O + occlusion level 고려 O
            input_heatmap = self.generate_input_heatmap(joints, joints_vis, joints_occ, joints_occ_level)
        elif self.heatmap_generation == 'input_random_noise':
            # random으로 생성한 occlusion (other model)
            input_heatmap = self.generate_input_heatmap_random_noise(joints, joints_vis)
        
        target_heatmap, target_weight = self.generate_target_heatmap(
            joints, joints_vis)
        target_heatmap = torch.from_numpy(target_heatmap)
        target_weight = torch.from_numpy(target_weight)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, self.num_joints-4, 2))
        joints_vis_u = np.zeros((self.maximum_person, self.num_joints-4, 2))
        for i in range(nposes):
            joints_u[i] = joints[i][:15]
            joints_vis_u[i] = joints_vis[i][:15]

        joints_3d_u = np.zeros((self.maximum_person, self.num_joints-4, 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints-4, 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:15, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:15, 0:3]

        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        if isinstance(self.root_id, int):
            roots_3d = joints_3d_u[:, self.root_id]
        elif isinstance(self.root_id, list):
            roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
        meta = {
            'image': image_file,
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'joints_3d_vis': joints_3d_vis_u,
            'roots_3d': roots_3d,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': db_rec['camera']
        }

        return [], target_heatmap, target_weight, target_3d, meta, input_heatmap

    def compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx)**2,  1.0 / 4 * 96**2, 4 * 96**2)

    def generate_target_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        nposes = len(joints)
        num_joints = self.num_joints-4
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if joints_vis[n][joint_id, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [np.searchsorted(grid1Dx,  mu_x - 3 * cur_sigma),
                       np.searchsorted(grid1Dx,  mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy,  mu_y - 3 * cur_sigma),
                       np.searchsorted(grid1Dy,  mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz,  mu_z - 3 * cur_sigma),
                       np.searchsorted(grid1Dz,  mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]], indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target
            
    # # version 1
    # def generate_input_heatmap(self, joints, joints_vis, joints_occ, joints_occ_level=None):
    #     if joints_occ_level is None:
    #         occ_levels = {1:0.5, 2:0.5, 3:0.5, 4:0.5}
    #         occ_lower_intensity = {1:[0.4,0.6], 2:[0.4,0.6], 3:[0.4,0.6], 4:[0.4,0.6]}
    #     else:
    #         occ_levels = {1:0.2, 2:0.4, 3:0.6, 4:0.8}
    #         occ_lower_intensity = {1:[0.8,1.0], 2:[0.6,0.8], 3:[0.4,0.6], 4:[0.2,0.4]}
            
    #     nposes = len(joints)
    #     num_joints = self.num_joints-4
    #     target_weight = np.zeros((num_joints, 1), dtype=np.float32)
    #     for i in range(num_joints):
    #         for n in range(nposes):
    #             if joints_vis[n][i, 0] == 1:
    #                 target_weight[i, 0] = 1

    #     assert self.target_type == 'gaussian', \
    #         'Only support gaussian map now!'

    #     if self.target_type == 'gaussian':
    #         target = np.zeros(      # empty heatmap 생성
    #             (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
    #             dtype=np.float32)
    #         feat_stride = self.image_size / self.heatmap_size

    #         for n in range(nposes):
    #             human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
    #             if human_scale == 0:
    #                 continue
    #             all_occlusion = False   # 모든 관절이 occlusion 되었는지 여부 확인
    #             if np.all(joints_occ[n]):
    #                 all_occlusion = True

    #             for joint_id in range(num_joints):
    #                 feat_stride = self.image_size / self.heatmap_size
    #                 mu_x = int(joints[n][joint_id][0] / feat_stride[0])         # heatmap 중앙 좌표
    #                 mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    
    #                 cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
    #                 tmp_size = cur_sigma * 3       # 하나의 Gaussian heatmap의 크기 (반지름)
    #                 ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # upper lower (경계 좌표)
    #                 br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # bottom right
    #                 if joints_vis[n][joint_id, 0] == 0 or \
    #                             ul[0] >= self.heatmap_size[0] or \
    #                             ul[1] >= self.heatmap_size[1] \
    #                             or br[0] < 0 or br[1] < 0:
    #                         continue
    #                 # 전체 joint occlusion에 대한 처리
    #                 if all_occlusion:
    #                     expand_ratio = np.random.uniform(low=0.5, high=1.5, size=1)
    #                     lower_intensity = np.random.uniform(low=0.0, high=0.3)
    #                     # (1) heatmap 크기 random scaling
    #                     cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0))) * expand_ratio
    #                     tmp_size = cur_sigma * 3
    #                     size = 2 * tmp_size + 1
    #                     x = np.arange(0, size, 1, np.float32)
    #                     y = x[:, np.newaxis]
    #                     x0 = y0 = size // 2
    #                     g = np.exp(         
    #                         -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))  # 중간값으로부터 heatmap intensity 값 정의
    #                     # (2) 전체적으로 intensity 줄이기
    #                     g = g * lower_intensity
    #                     # (3) heatmap peak 위치 -> random shift
    #                     while True:
    #                         mu_occ = np.random.normal(loc=0.0, scale=2, size=2)
    #                         mu_x, mu_y = int(mu_x + mu_occ[0]), int(mu_y + mu_occ[1])
    #                         ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    #                         br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    #                         if joints_vis[n][joint_id, 0] == 0 or not (ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] 
    #                             or br[0] < 0 or br[1]) < 0:
    #                             break
    #                 # occlusion joint에 대한 처리
    #                 elif joints_occ[n][joint_id, 0]:
    #                     if joints_occ_level is None:
    #                         occ_level = 1
    #                     else:
    #                         occ_level = joints_occ_level[n][joint_id, 0]    # occlusion의 강도
    #                     expand_ratio = np.random.uniform(low=1.0-occ_levels[occ_level]*0.5, high=1.0+occ_levels[occ_level]*0.5)
    #                     lower_intensity = np.random.uniform(low=occ_lower_intensity[occ_level][0], high=occ_lower_intensity[occ_level][1])
    #                     cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0))) * expand_ratio
    #                     tmp_size = cur_sigma * 3
    #                     size = 2 * tmp_size + 1
    #                     x = np.arange(0, size, 1, np.float32)
    #                     y = x[:, np.newaxis]
    #                     x0 = y0 = size // 2
    #                     g = np.exp(         
    #                         -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))
    #                     g = g * lower_intensity
    #                     while True:
    #                         mu_occ = np.random.normal(loc=0.0, scale=2*occ_levels[occ_level], size=2)
    #                         mu_x, mu_y = int(mu_x + mu_occ[0]), int(mu_y + mu_occ[1])
    #                         ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    #                         br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    #                         if joints_vis[n][joint_id, 0] == 0 or not (ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] 
    #                             or br[0] < 0 or br[1]) < 0:
    #                             break
    #                 # not occlusion joint에 대한 처리
    #                 else:
    #                     cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
    #                     tmp_size = cur_sigma * 3       # 하나의 Gaussian heatmap의 크기 (반지름)
    #                     ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # upper lower (경계 좌표)
    #                     br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # bottom right
    #                     # heatmap의 크기를 벗어나는 경우 continue
    #                     if joints_vis[n][joint_id, 0] == 0 or \
    #                             ul[0] >= self.heatmap_size[0] or \
    #                             ul[1] >= self.heatmap_size[1] \
    #                             or br[0] < 0 or br[1] < 0:
    #                         continue
    #                     size = 2 * tmp_size + 1
    #                     x = np.arange(0, size, 1, np.float32)
    #                     y = x[:, np.newaxis]
    #                     x0 = y0 = size // 2
    #                     g = np.exp(         
    #                         -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))  # 중간값으로부터 heatmap intensity 값 정의
                        
    #                 # Usable gaussian range
    #                 g_x = max(0,-ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
    #                 g_y = max(0,-ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
    #                 # Image range
    #                 img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
    #                 img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
                    
    #                 target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
    #                     g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                    
    #             target = np.clip(target, 0, 1)      # 0~1 으로 값 제한
    #     if self.use_different_joints_weight:
    #         target_weight = np.multiply(target_weight, self.joints_weight)

    #     return target
    
    
    # version 2
    def generate_input_heatmap(self, joints, joints_vis, joints_occ, joints_occ_level=None):
        nposes = len(joints)
        num_joints = self.num_joints-4
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(      # empty heatmap 생성
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue
                all_occlusion = False   # 모든 관절이 occlusion 되었는지 여부 확인
                if np.all(joints_occ[n]):
                    all_occlusion = True

                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])         # 관절의 heatmap 중앙 좌표
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    
                    cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                    tmp_size = cur_sigma * 3       # 하나의 Gaussian heatmap의 크기 (반지름)
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # upper lower (경계 좌표)
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # bottom right
                    if joints_vis[n][joint_id, 0] == 0 or \
                                ul[0] >= self.heatmap_size[0] or \
                                ul[1] >= self.heatmap_size[1] \
                                or br[0] < 0 or br[1] < 0:
                            continue
                    # 전체 joint occlusion에 대한 처리
                    if all_occlusion:
                        continue
                    
                    # occlusion joint에 대한 처리
                    elif joints_occ[n][joint_id, 0]:
                        cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                        tmp_size = cur_sigma * 3
                        size = 2 * tmp_size + 1
                        x = np.arange(0, size, 1, np.float32)
                        y = x[:, np.newaxis]
                        x0 = y0 = size // 2
                        g = np.exp(         
                            -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))
                        
                        # (1) intensity 낮추기
                        if joints_occ_level is None:
                            lower_intensity = 0.5
                        else:
                            # lower_intensity = 1 - joints_occ_level[n][joint_id, 0] / 340  # v2
                            lower_intensity = 1 - joints_occ_level[n][joint_id, 0] / 170
                        g = g * lower_intensity
                        while True:
                            # (2) random shift
                            if joints_occ_level is None:
                                mu_occ = np.random.normal(loc=0.0, scale=1, size=2)
                            else:
                                # mu_occ = np.random.normal(loc=0.0, scale=joints_occ_level[n][joint_id, 0]/100, size=2)  # v2
                                mu_occ = np.random.normal(loc=0.0, scale=joints_occ_level[n][joint_id, 0]/50, size=2)  # v2
                            mu_x, mu_y = int(mu_x + mu_occ[0]), int(mu_y + mu_occ[1])
                            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                            
                            # if joints_vis[n][joint_id, 0] == 0 or not (ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] 
                            #     or br[0] < 0 or br[1]) < 0:
                            #     break                            
                            if not (joints_vis[n][joint_id, 0] == 0 or \
                                    ul[0] >= self.heatmap_size[0] or \
                                    ul[1] >= self.heatmap_size[1] \
                                    or br[0] < 0 or br[1] < 0):
                                break
                            
                    # not occlusion joint에 대한 처리
                    else:
                        cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                        tmp_size = cur_sigma * 3       # 하나의 Gaussian heatmap의 크기 (반지름)
                        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # upper lower (경계 좌표)
                        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # bottom right
                        # heatmap의 크기를 벗어나는 경우 continue
                        if joints_vis[n][joint_id, 0] == 0 or \
                                ul[0] >= self.heatmap_size[0] or \
                                ul[1] >= self.heatmap_size[1] \
                                or br[0] < 0 or br[1] < 0:
                            continue
                        size = 2 * tmp_size + 1
                        x = np.arange(0, size, 1, np.float32)
                        y = x[:, np.newaxis]
                        x0 = y0 = size // 2
                        g = np.exp(         
                            -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))  # 중간값으로부터 heatmap intensity 값 정의
                        
                    # Usable gaussian range
                    g_x = max(0,-ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,-ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
                    
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                    
                target = np.clip(target, 0, 1)      # 0~1 으로 값 제한
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target
    
    
    # heatmap 이미지의 크기에 기반하여 임의의 mask region 생성
    def generate_random_mask_points(self, width, height):
        """
        image_size: 이미지의 크기 (너비, 높이)
        return: (가로 시작 픽셀, 가로 끝 픽셀, 세로 시작 픽셀, 세로 끝 픽셀)
        """
        min_mask_size, max_mask_size = 0.1, 0.5
        mask_width = np.random.randint(int(width * min_mask_size), int(width * max_mask_size))
        mask_height = np.random.randint(int(height * min_mask_size), int(height * max_mask_size))

        start_row = np.random.randint(0, width - mask_width)
        start_col = np.random.randint(0, height - mask_height)
        end_row = start_row + mask_width
        end_col = start_col + mask_height

        return start_row, end_row, start_col, end_col
    
    
    def generate_input_heatmap_random_noise(self, joints, joints_vis):
        keypoint_occlusion = 0.1
        region_occlusion = 0.05
        shift_noise = 0.1
        nposes = len(joints)
        num_joints = self.num_joints-4
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(      # empty heatmap 생성
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])         # heatmap 중앙 좌표
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    
                    cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                    tmp_size = cur_sigma * 3       # 하나의 Gaussian heatmap의 크기 (반지름)
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]           # upper lower (경계 좌표)
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   # bottom right
                    if joints_vis[n][joint_id, 0] == 0 or \
                                ul[0] >= self.heatmap_size[0] or \
                                ul[1] >= self.heatmap_size[1] \
                                or br[0] < 0 or br[1] < 0:
                            continue
                        
                    # (1) keypoint 별 occlusion 적용 (random keypoint가 occlusion 되면 0으로 설정)
                    if np.random.rand() <= keypoint_occlusion:
                        continue
                    
                    # (2) region 별 occlusion 적용 (random occlusion region 내에 keypoint가 있으면 0으로 설정)
                    if np.random.rand() <= region_occlusion:
                        start_row, end_row, start_col, end_col = self.generate_random_mask_points(self.heatmap_size[0], self.heatmap_size[1])
                        if mu_x > start_row and mu_x < end_row and mu_y > start_col and mu_y < end_col:
                            continue
                    
                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(         
                        -((x - x0)**2 + (y - y0)**2) / (2 * cur_sigma**2))
                    
                    # (3) randomly shift pixel 적용
                    if np.random.rand() <= shift_noise:
                        while True:
                            mu_occ = np.random.normal(loc=0.0, scale=3, size=2)
                            mu_x, mu_y = int(mu_x + mu_occ[0]), int(mu_y + mu_occ[1])
                            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]   
                            if not (joints_vis[n][joint_id, 0] == 0 or \
                                    ul[0] >= self.heatmap_size[0] or \
                                    ul[1] >= self.heatmap_size[1] \
                                    or br[0] < 0 or br[1] < 0):
                                break
                        
                    # Usable gaussian range
                    g_x = max(0,-ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,-ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])
                    
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                    
                target = np.clip(target, 0, 1)      # 0~1 으로 값 제한
        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target