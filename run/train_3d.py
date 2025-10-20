# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset

from tensorboardX import SummaryWriter
import argparse
import os
import pprint
import logging
import json
import numpy as np

import _init_paths
from core.config import config
from core.config import update_config
from core.function import train_3d, validate_3d
from utils.utils import create_logger
from utils.utils import save_checkpoint, load_checkpoint, load_model_state
from utils.utils import load_backbone_panoptic
import dataset
import models

# random.seed(0)
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    
    parser.add_argument('--camera_num', default=None, type=int, help='Set DATASET.CAMERA_NUM value')
    parser.add_argument('--aug_ratio', default=None, type=int, help='Set DATASET.AUG_RATIO value')
    parser.add_argument('--heatmap_generation', default=None, type=str, help='Set DATASET.HEATMAP_GENERATION value')
    parser.add_argument('--data_num', default=None, type=int, help='Set DATASET.DATA_NUM value')

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    return args


def get_optimizer(model):   # optimizer 얻는 함수
    lr = config.TRAIN.LR
    if model.module.backbone is not None:
        for params in model.module.backbone.parameters():
            params.requires_grad = False   # If you want to train the whole model jointly, set it to be True.
    for params in model.module.root_net.parameters():
        params.requires_grad = True
    for params in model.module.pose_net.parameters():
        params.requires_grad = True
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.module.parameters()), lr=lr)
    # optimizer = optim.Adam(model.module.parameters(), lr=lr)

    return model, optimizer


def main():
    args = parse_args()
    
    # 인자를 기반으로 config 값 업데이트
    if args.camera_num is not None:
        config.DATASET.CAMERA_NUM = args.camera_num
    if args.aug_ratio is not None:
        config.DATASET.AUG_RATIO = args.aug_ratio
    if args.heatmap_generation is not None:
        config.DATASET.HEATMAP_GENERATION = args.heatmap_generation
    if args.data_num is not None:
        config.DATASET.DATA_NUM = args.data_num
        
    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    gpus = [int(i) for i in config.GPUS.split(',')]

    # datasets 구축
    print('=> Loading data ..')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    
    # augmentation data 포함해서 학습
    if config.DATASET.AUG_TYPE == 'mpda':
        train_augmented_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET + '_mpda')(
            config, config.DATASET.TRAIN_SUBSET, True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        combined_train_dataset = ConcatDataset([train_dataset, train_augmented_dataset])
        train_loader = torch.utils.data.DataLoader(
            combined_train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True)
    elif config.DATASET.AUG_TYPE == 'original':
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True)

    valid_dataset = eval('dataset.' + config.DATASET.VALID_DATASET)(
        config, config.DATASET.VALID_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)
    
    if 'panoptic' in config.DATASET.TEST_DATASET:
        test_occ_dataset = eval('dataset.' + config.DATASET.TEST_OCC_DATASET)(
            config, config.DATASET.TEST_OCC_SUBSET, False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))    
        test_occ_loader = torch.utils.data.DataLoader(
            test_occ_dataset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=config.WORKERS,
            pin_memory=True)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # model, optimizer 구축
    print('=> Constructing models ..')
    model = eval('models.' + config.MODEL + '.get_multi_person_pose_net')(
        config, is_train=True)
    
    with torch.no_grad():
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
  
    model, optimizer = get_optimizer(model)

    start_epoch = config.TRAIN.BEGIN_EPOCH
    end_epoch = config.TRAIN.END_EPOCH

    best_mpjpe = np.inf
    best_avg_pcp = 0
    if config.NETWORK.PRETRAINED_BACKBONE:
        model = load_backbone_panoptic(model, config.NETWORK.PRETRAINED_BACKBONE)
    if config.TRAIN.RESUME:
        if 'panoptic' in config.DATASET.TEST_DATASET or 'etri' in config.DATASET.TEST_DATASET \
                or 'hospital' in config.DATASET.TEST_DATASET:
            start_epoch, model, optimizer, best_mpjpe, _ = load_checkpoint(model, optimizer, final_output_dir)
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            start_epoch, model, optimizer, _, best_avg_pcp = load_checkpoint(model, optimizer, final_output_dir)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 학습 시작
    print('=> Training...')
    for epoch in range(start_epoch, end_epoch):
        print('Epoch: {}'.format(epoch))

        # lr_scheduler.step()
        # with torch.autograd.set_detect_anomaly(True):   # 오류 때문에 수정
        train_3d(config, model, optimizer, train_loader, epoch, final_output_dir, writer_dict)
        if 'panoptic' in config.DATASET.TEST_DATASET or 'etri' in config.DATASET.TEST_DATASET \
                or 'hospital' in config.DATASET.TEST_DATASET:
            mpjpe = validate_3d(config, model, valid_loader, final_output_dir)
            if mpjpe < best_mpjpe:
                best_mpjpe = mpjpe
                best_model = True
            else:
                best_model = False
            logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'mpjpe': best_mpjpe,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)
        
        elif 'campus' in config.DATASET.TEST_DATASET or 'shelf' in config.DATASET.TEST_DATASET:
            avg_pcp = validate_3d(config, model, valid_loader, final_output_dir)
            if avg_pcp > best_avg_pcp:
                best_avg_pcp = avg_pcp
                best_model = True
            else:
                best_model = False
            logger.info('=> saving checkpoint to {} (Best: {})'.format(final_output_dir, best_model))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'avg_pcp': best_avg_pcp,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth.tar')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)

    # TEST
    test_model_file = os.path.join(final_output_dir, config.TEST.MODEL_FILE)
    if config.TEST.MODEL_FILE and os.path.isfile(test_model_file):
        logger.info('=> load best models state {}'.format(test_model_file))
        model.module.load_state_dict(torch.load(test_model_file))
    else:
        raise ValueError('Check the model file for testing!')
    validate_3d(config, model, test_loader, final_output_dir)
    
    if 'panoptic' in config.DATASET.TEST_DATASET:
        print('------------------------------------------')
        print(f'Samples with heavy occlusion')
        validate_3d(config, model, test_occ_loader, final_output_dir)
    
    writer_dict['writer'].close()
    
    
if __name__ == '__main__':
    main()
