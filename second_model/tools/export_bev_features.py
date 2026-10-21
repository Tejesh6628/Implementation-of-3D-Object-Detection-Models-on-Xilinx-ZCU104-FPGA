import _init_path
import argparse
import os
import torch
import numpy as np
from pathlib import Path
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from demo import DemoDataset

def parse_config():
    parser = argparse.ArgumentParser(description='Export BEV features from VoxelBackBone8x')
    parser.add_argument('--cfg_file', type=str, required=True, help='Path to SECOND config')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to SECOND checkpoint')
    parser.add_argument('--out_dir', type=str, required=True, help='Output dir for .npy files')
    parser.add_argument('--data_path', type=str, default='/media/vlsi/Expansion/kitti_np/testing/velodyne', help='Path to the raw KITTI testing velodyne .bin files')
    parser.add_argument('--num_frames', type=int, default=50, help='Number of frames to extract')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()
    
    cfg_from_yaml_file(args.cfg_file, cfg)
    if args.set_cfgs is not None:
        from pcdet.config import cfg_from_list
        cfg_from_list(args.set_cfgs, cfg)
        
    return args, cfg

def main():
    args, cfg = parse_config()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = common_utils.create_logger()

    # Use DemoDataset to directly load .bin files without needing info.pkl files
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, 
        class_names=cfg.CLASS_NAMES, 
        training=False,
        root_path=Path(args.data_path), 
        ext='.bin', 
        logger=logger
    )
    
    logger.info(f'Total number of samples found: {len(demo_dataset)}')

    test_loader = torch.utils.data.DataLoader(
        demo_dataset, batch_size=1, pin_memory=True, num_workers=4,
        shuffle=False, collate_fn=demo_dataset.collate_batch,
        drop_last=False, timeout=0
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    print(f"Exporting the first {args.num_frames} BEV features to {out_dir}...")
    with torch.no_grad():
        for i, batch_dict in enumerate(test_loader):
            if i >= args.num_frames:
                break
                
            # Load batch to GPU
            load_data_to_gpu(batch_dict)

            # Pass through 3D backbone
            batch_dict = model.vfe(batch_dict)
            batch_dict = model.backbone_3d(batch_dict)
            batch_dict = model.map_to_bev_module(batch_dict)
            
            # The output of HeightCompression is spatial_features
            spatial_features = batch_dict['spatial_features'].cpu().numpy() # [B, C, H, W]

            # Save each frame
            frame_ids = batch_dict['frame_id']
            for b_idx in range(len(frame_ids)):
                # Since we don't have frame IDs from pkl, we use the original filename
                original_file_path = Path(demo_dataset.sample_file_list[i]).stem
                feat = spatial_features[b_idx] # [C, H, W]
                save_path = out_dir / f"bev_{original_file_path}.npy"
                np.save(str(save_path), feat)
                
            if i % 10 == 0:
                print(f"Processed {i+1} / {args.num_frames} frames")

    print(f"Done! Saved {args.num_frames} .npy files.")

if __name__ == '__main__':
    main()
