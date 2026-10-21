"""
STEP 1: Export SECOND (OpenPCDet) to ONNX
=========================================
Run this on your HOST machine (Linux GPU server or Windows with CUDA).

This script:
  - Loads checkpoint_epoch_80.pth
  - Traces only the DPU-mappable portion (VFE is excluded – it is a sparse
    scatter op that the DPU cannot run; we handle it in pre-processing)
  - Exports the BEV backbone + detection head as a single ONNX graph

Usage:
    python 1_export_onnx.py \
        --cfg  ../SECOND_model/tools/cfgs/kitti_models/second.yaml \
        --ckpt ../SECOND_model/checkpoints/checkpoint_epoch_80.pth \
        --output ./second_bev.onnx

NOTE: VFE (MeanVFE) + voxelization are handled by the custom pre-processor
      (see 4_run_on_board.py).  Only the dense BEV branch is exported.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'SECOND_model', 'tools'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                '..', 'SECOND_model'))

import argparse
import numpy as np
import torch
import torch.nn as nn

# ── OpenPCDet imports ──────────────────────────────────────────────────────────
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.utils import common_utils

# ── Wrapper that exposes only the dense BEV portion ──────────────────────────
class SECONDBEVWrapper(nn.Module):
    """
    Wraps the SECONDNet dense pipeline:
        spatial_features  →  backbone_2d  →  dense_head  →  (cls, box, dir)
    Input  : BEV pseudo-image  [B, C_bev, H, W]  (float32)
    Outputs: cls_preds  [B, num_anchors, num_classes]
             box_preds  [B, num_anchors, 7]
             dir_preds  [B, num_anchors, 2]
    """
    def __init__(self, model):
        super().__init__()
        self.backbone_2d = model.backbone_2d
        self.dense_head  = model.dense_head

    def forward(self, spatial_features):
        data_dict = {'spatial_features': spatial_features}
        data_dict = self.backbone_2d(data_dict)
        data_dict = self.dense_head(data_dict)
        cls_preds = data_dict['batch_cls_preds']   # [B, N, num_cls]
        box_preds = data_dict['batch_box_preds']   # [B, N, 7]
        dir_preds = data_dict.get('batch_dir_cls_preds',
                                  torch.zeros(cls_preds.shape[0],
                                              cls_preds.shape[1], 2,
                                              device=cls_preds.device))
        return cls_preds, box_preds, dir_preds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg',    required=True,
                   help='Path to second.yaml')
    p.add_argument('--ckpt',   required=True,
                   help='Path to checkpoint_epoch_80.pth')
    p.add_argument('--output', default='second_bev.onnx',
                   help='Output ONNX path')
    p.add_argument('--opset',  type=int, default=11,
                   help='ONNX opset version (11 is safe for Vitis AI 3.0)')
    return p.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg, cfg)
    logger = common_utils.create_logger()

    # ── Build model ──────────────────────────────────────────────────────────
    # We need a dummy dataset for build_network; create a minimal stub.
    class _DummyDataset:
        class point_cloud_range:
            pass
        point_cloud_range = [0, -40, -3, 70.4, 40, 1]
        class voxel_size:
            pass
        voxel_size = [0.05, 0.05, 0.1]
        grid_size  = np.array([1408, 1600, 40], dtype=np.int64)
        class_names = cfg.CLASS_NAMES

    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=_DummyDataset())

    logger.info(f'Loading checkpoint: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model_state = ckpt.get('model_state', ckpt)
    model.load_state_dict(model_state, strict=False)
    model.eval()

    # ── Build BEV wrapper ────────────────────────────────────────────────────
    bev_model = SECONDBEVWrapper(model).eval()

    # BEV feature map shape: [B=1, 256, H, W]
    # KITTI grid: 70.4/0.05=1408 (x) × 80/0.05=1600 (y)
    # After VoxelBackBone8x (8× downsampling): 176 × 200
    H_bev, W_bev = 200, 176   # height-compressed after 8× sparse backbone
    C_bev = 256                # NUM_BEV_FEATURES from second.yaml
    dummy_bev = torch.zeros(1, C_bev, H_bev, W_bev)

    logger.info('Running a forward pass to verify shapes …')
    with torch.no_grad():
        cls_p, box_p, dir_p = bev_model(dummy_bev)
    logger.info(f'  cls_preds : {cls_p.shape}')
    logger.info(f'  box_preds : {box_p.shape}')
    logger.info(f'  dir_preds : {dir_p.shape}')

    # ── Export ───────────────────────────────────────────────────────────────
    logger.info(f'Exporting ONNX → {args.output}')
    torch.onnx.export(
        bev_model,
        dummy_bev,
        args.output,
        opset_version=args.opset,
        input_names=['bev_input'],
        output_names=['cls_preds', 'box_preds', 'dir_preds'],
        dynamic_axes={
            'bev_input':  {0: 'batch'},
            'cls_preds':  {0: 'batch'},
            'box_preds':  {0: 'batch'},
            'dir_preds':  {0: 'batch'},
        },
        do_constant_folding=True,
        verbose=False,
    )
    logger.info('ONNX export complete.')

    # ── Quick validation ─────────────────────────────────────────────────────
    try:
        import onnx
        m = onnx.load(args.output)
        onnx.checker.check_model(m)
        logger.info('ONNX model check: PASSED')
    except ImportError:
        logger.warning('onnx package not found – skipping validation.')


if __name__ == '__main__':
    main()
