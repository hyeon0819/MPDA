# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))  ##
    root_output_dir = (this_dir / '..' / '..' / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (this_dir / '..' / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]
    
    aug_type = cfg.DATASET.AUG_TYPE
    cam_num = cfg.DATASET.CAMERA_NUM
    # cam_similarity = cfg.DATASET.CAM_SIMILARITY
    data_num = cfg.DATASET.DATA_NUM
    aug_ratio = cfg.DATASET.AUG_RATIO
    heatmap_generation = cfg.DATASET.HEATMAP_GENERATION

    # if phase == 'validate':
    #     valid_final_output_dir = root_output_dir / 'validate' / dataset / aug_type / f"cam{cam_num}_{cam_similarity}_data{data_num}_{aug_ratio}_{heatmap_generation}"
    #     final_output_dir = root_output_dir / dataset / aug_type / f"cam{cam_num}_{cam_similarity}_data{data_num}_{aug_ratio}_{heatmap_generation}" 
    #     print('=> creating {}'.format(valid_final_output_dir))
    #     valid_final_output_dir.mkdir(parents=True, exist_ok=True)

    #     time_str = time.strftime('%Y-%m-%d-%H-%M')
    #     log_file = f"{aug_type}_cam{cam_num}_{cam_similarity}_data{data_num}_{aug_ratio}_{heatmap_generation}.log"
    #     final_log_file = valid_final_output_dir / log_file
    #     head = '%(asctime)-15s %(message)s'
    #     logging.basicConfig(filename=str(final_log_file),
    #                         format=head)
    #     logger = logging.getLogger()
    #     logger.setLevel(logging.INFO)
    #     console = logging.StreamHandler()
    #     logging.getLogger('').addHandler(console)
        
    #     return logger, valid_final_output_dir, final_output_dir
    
    if aug_type == 'original':
        final_output_dir = root_output_dir / dataset / phase / f"{aug_type}_cam{cam_num}_data{data_num}"
        log_file = f"{aug_type}_cam{cam_num}_data{data_num}.log"
    else:
        final_output_dir = root_output_dir / dataset / phase / f"{aug_type}_cam{cam_num}_data{data_num}_aug{aug_ratio}_{heatmap_generation}"
        log_file = f"{aug_type}_cam{cam_num}_data{data_num}_aug{aug_ratio}_{heatmap_generation}.log"

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = tensorboard_log_dir / dataset / phase / f"{aug_type}_cam{cam_num}_data{data_num}_aug{aug_ratio}_{heatmap_generation}"
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def load_model_state(model, output_dir, epoch):
    file = os.path.join(output_dir, 'checkpoint_3d_epoch'+str(epoch)+'.pth.tar')
    if os.path.isfile(file):
        model.module.load_state_dict(torch.load(file))
        print('=> load models state {} (epoch {})'
              .format(file, epoch))
        return model
    else:
        print('=> no checkpoint found at {}'.format(file))
        return model


def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        mpjpe = checkpoint['mpjpe'] if 'mpjpe' in checkpoint else np.inf
        avg_pcp = checkpoint['avg_pcp'] if 'avg_pcp' in checkpoint else 0
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, mpjpe, avg_pcp

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, np.inf, 0


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def load_backbone_panoptic(model, pretrained_file):
    this_dir = os.path.dirname(__file__)
    pretrained_file = os.path.abspath(os.path.join(this_dir, '../..', pretrained_file))
    pretrained_state_dict = torch.load(pretrained_file)
    model_state_dict = model.module.backbone.state_dict()

    prefix = "module."
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
    logging.info("load backbone statedict from {}".format(pretrained_file))
    model.module.backbone.load_state_dict(new_pretrained_state_dict)

    return model
