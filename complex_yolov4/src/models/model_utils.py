# -*- coding: utf-8 -*-
"""
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.06.18
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: utils functions that use for model
"""

import sys
import torch

sys.path.append('../')

from models.darknet2pytorch import Darknet


def create_model(configs):
    """Create model based on architecture name"""
    if (configs.arch == 'darknet') and (configs.cfgfile is not None):
        print('Using darknet backbone')
        model = Darknet(cfgfile=configs.cfgfile,
                        use_giou_loss=configs.use_giou_loss)
    else:
        raise ValueError('Undefined model backbone')

    return model


def get_num_parameters(model):
    """Count number of trainable parameters"""
    if hasattr(model, 'module'):
        num_parameters = sum(
            p.numel() for p in model.module.parameters() if p.requires_grad
        )
    else:
        num_parameters = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

    return num_parameters


def make_data_parallel(model, configs):
    if getattr(configs, 'distributed', False):
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    elif not getattr(configs, 'no_cuda', False):
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = model.cpu()

    return model


if __name__ == '__main__':
    import argparse
    from easydict import EasyDict as edict

    parser = argparse.ArgumentParser(
        description='Complex YOLOv4 Implementation'
    )
    parser.add_argument('-a', '--arch', type=str,
                        default='darknet', metavar='ARCH',
                        help='Model architecture name')
    parser.add_argument('--cfgfile', type=str,
                        default='../config/cfg/complex_yolov4.cfg',
                        metavar='PATH',
                        help='Path to cfg file')

    configs = edict(vars(parser.parse_args()))

    # FORCE CPU
    configs.device = torch.device('cpu')

    model = create_model(configs).to(configs.device)

    sample_input = torch.randn((1, 3, 608, 608)).to(configs.device)

    output = model(sample_input, targets=None)

    print("Output size:", output.size())