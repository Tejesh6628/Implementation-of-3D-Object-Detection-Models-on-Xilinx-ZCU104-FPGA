"""
finetune_leakyrelu.py - Fine-tune Complex-YOLOv4 with LeakyReLU
================================================================
Loads the Mish-trained 300-epoch checkpoint and fine-tunes for 20-30
epochs with LeakyReLU activations (already changed in darknet2pytorch.py).

This re-aligns the conv/BN weights to work with LeakyReLU, fixing the
false positive / missed detection issues on the DPU.

USAGE (on GPU machine):
    cd complex_yolov4/src
    python finetune_leakyrelu.py --dataset_dir /path/to/KITTI/kitti_medium_split

    The script will:
    1. Load Model_complexer_yolo_epoch_300.pth (Mish-trained weights)
    2. Fine-tune for 30 epochs with LeakyReLU
    3. Save checkpoints to ../checkpoints/complexer_yolo_leakyrelu/

AFTER TRAINING:
    Copy the best checkpoint to the Vitis AI Docker and re-quantize:
        python quantize_for_dpu.py
        vai_c_xir -x quantize_result/DarknetBackbone_int.xmodel -a arch.json -o compiled/ -n complex_yolov4
"""

import time
import math
import numpy as np
import sys
import random
import os
import warnings
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append('./')

from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model
from utils.misc import AverageMeter
from evaluate import evaluate_mAP
from easydict import EasyDict as edict


def main():  
    parser = argparse.ArgumentParser(description='Fine-tune Complex-YOLOv4 with LeakyReLU')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Path to KITTI dataset (kitti_medium_split)')
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/Model_complexer_yolo_epoch_300.pth',
                        help='Path to Mish-trained checkpoint')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of fine-tuning epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate (default: 1e-5, keep low!)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Data loader workers')
    parser.add_argument('--save_dir', type=str,
                        default='../checkpoints/complexer_yolo_leakyrelu',
                        help='Directory to save fine-tuned checkpoints')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--no_val', action='store_true',
                        help='Skip validation (faster training)')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='GPU index')
    args = parser.parse_args()

    # Setup
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Build config (mimics train_config.py) ──
    configs = edict({
        'arch': 'darknet',
        'cfgfile': 'config/cfg/complex_yolov4.cfg',
        'use_giou_loss': False,
        'img_size': 608,
        'hflip_prob': 0.5,
        'cutout_prob': 0.0,
        'cutout_nholes': 1,
        'cutout_ratio': 0.3,
        'cutout_fill_value': 0.0,
        'multiscale_training': False,
        'mosaic': False,
        'random_padding': False,
        'num_samples': None,
        'num_workers': args.num_workers,
        'batch_size': args.batch_size,
        'subdivisions': max(1, 64 // args.batch_size),
        'dataset_dir': args.dataset_dir,
        'device': device,
        'gpu_idx': args.gpu_idx,
        'distributed': False,
        'pin_memory': True,
        'no_val': args.no_val,
        'conf_thresh': 0.5,
        'nms_thresh': 0.5,
        'iou_thresh': 0.5,
        'img_size': 608,
    })

    # ── Create model (now uses LeakyReLU thanks to darknet2pytorch.py change) ──
    print("\n[1/4] Creating model with LeakyReLU activations...")
    model = create_model(configs)

    # ── Load Mish-trained checkpoint ──
    print(f"[2/4] Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    state_dict = torch.load(args.checkpoint, map_location='cpu')
    # Handle both DataParallel and regular checkpoints
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"  Loaded: {len(state_dict)} keys")
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing keys: {missing[:5]}...")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {num_params:,}")

    # ── Optimizer (low LR for fine-tuning) ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    print(f"  Optimizer: Adam, LR={args.lr}")

    # ── Dataloader ──
    print(f"\n[3/4] Loading dataset from: {args.dataset_dir}")
    train_dataloader, _ = create_train_dataloader(configs)
    print(f"  Training batches: {len(train_dataloader)}")

    # ── Training loop ──
    print(f"\n[4/4] Fine-tuning for {args.num_epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = AverageMeter('Loss', ':.4e')
        start_time = time.time()

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.num_epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            _, imgs, targets = batch_data
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            total_loss, outputs = model(imgs, targets)

            if not math.isfinite(total_loss.item()):
                print(f'  WARNING: NaN/Inf loss at batch {batch_idx}, skipping')
                optimizer.zero_grad()
                continue

            total_loss.backward()

            # Accumulate gradients (subdivisions)
            if (batch_idx + 1) % configs.subdivisions == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss.update(total_loss.item(), imgs.size(0))
            pbar.set_postfix({'loss': f'{epoch_loss.avg:.4f}'})

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch}: loss={epoch_loss.avg:.4f}, time={elapsed:.1f}s")

        # Validation
        if not args.no_val and epoch % args.save_freq == 0:
            try:
                val_dataloader = create_val_dataloader(configs)
                precision, recall, AP, f1, ap_class = evaluate_mAP(
                    val_dataloader, model, configs, None)
                print(f"  Val: mAP={AP.mean():.4f}, P={precision.mean():.4f}, "
                      f"R={recall.mean():.4f}, F1={f1.mean():.4f}")
            except Exception as e:
                print(f"  Val error: {e}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            save_path = os.path.join(args.save_dir,
                                      f'Model_complexer_yolo_leakyrelu_epoch_{epoch}.pth')
            # Save raw state_dict (no DataParallel wrapper)
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  Saved: {save_path}")

    print("=" * 70)
    print("Fine-tuning complete!")
    print(f"\nCheckpoints saved to: {args.save_dir}")
    print(f"\nNEXT STEPS:")
    print(f"  1. Copy the best checkpoint to the Vitis AI Docker workspace")
    print(f"  2. Update CHECKPOINT path in quantize_for_dpu.py")
    print(f"  3. Re-run: python quantize_for_dpu.py")
    print(f"  4. Compile: vai_c_xir -x quantize_result/DarknetBackbone_int.xmodel "
          f"-a arch.json -o compiled/ -n complex_yolov4")
    print(f"  5. Deploy the new .xmodel to ZCU104")


if __name__ == '__main__':
    main()
