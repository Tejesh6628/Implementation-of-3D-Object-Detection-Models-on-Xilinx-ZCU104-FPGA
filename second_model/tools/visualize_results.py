"""
Visualize SECOND model predictions on KITTI images.
Draws 3D bounding boxes projected onto 2D camera images and saves them.

Usage:
    python visualize_results.py \
        --cfg_file cfgs/kitti_models/second.yaml \
        --ckpt ../output/kitti_models/second/kitti_standrad/ckpt/checkpoint_epoch_80.pth \
        --data_path /media/vlsi/Expansion/kitti_np/ \
        --output_dir ../output/kitti_models/second/kitti_standrad/visualizations \
        --num_samples 20 \
        --score_thresh 0.3
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import torch
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


# ──────────────── Color palette per class ──────────────────
CLASS_COLORS = {
    'Car':            (0,   255, 0),    # Green
    'Van':            (0,   200, 200),  # Cyan
    'Truck':          (255, 165, 0),    # Orange
    'Pedestrian':     (255, 0,   0),    # Red
    'Person_sitting': (255, 100, 100),  # Light Red
    'Cyclist':        (255, 255, 0),    # Yellow
    'Tram':           (148, 0,   211),  # Purple
    'Misc':           (128, 128, 128),  # Gray
}


def read_calib(calib_path):
    """Read KITTI calibration file and return projection matrices."""
    with open(calib_path, 'r') as f:
        lines = f.readlines()

    calib = {}
    for line in lines:
        line = line.strip()
        if len(line) == 0 or ':' not in line:
            continue
        key, value = line.split(':', 1)
        calib[key.strip()] = np.array([float(x) for x in value.strip().split()])

    # P2 is the left color camera projection matrix (3x4)
    P2 = calib['P2'].reshape(3, 4)

    # R0_rect: rotation to rectified camera coords (3x3)
    R0 = np.eye(4)
    R0[:3, :3] = calib['R0_rect'].reshape(3, 3)

    # Tr_velo_to_cam: transform from velodyne to camera (3x4)
    Tr = np.eye(4)
    Tr[:3, :4] = calib['Tr_velo_to_cam'].reshape(3, 4)

    return P2, R0, Tr


def boxes3d_lidar_to_camera(boxes3d_lidar, R0, Tr):
    """
    Convert 3D boxes from LiDAR coordinates to camera coordinates.
    boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading] in LiDAR
    Returns: (N, 7) [x, y, z, h, w, l, ry] in camera
    """
    xyz_lidar = boxes3d_lidar[:, :3]
    # Convert to homogeneous
    xyz_homo = np.concatenate([xyz_lidar, np.ones((xyz_lidar.shape[0], 1))], axis=1)
    # LiDAR -> Camera
    xyz_cam = (R0 @ Tr @ xyz_homo.T).T[:, :3]

    # Convert dimensions: LiDAR (dx, dy, dz) -> Camera (h, w, l)
    dx, dy, dz = boxes3d_lidar[:, 3], boxes3d_lidar[:, 4], boxes3d_lidar[:, 5]
    h, w, l = dz, dy, dx  # height=dz, width=dy, length=dx

    # Convert heading: LiDAR heading -> camera ry
    # In KITTI, ry = -(heading + pi/2) approximately
    ry = -boxes3d_lidar[:, 6] - np.pi / 2

    return np.stack([xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2], h, w, l, ry], axis=1)


def corners3d_from_camera_box(boxes_camera):
    """
    Get 8 corners of 3D boxes in camera coordinates.
    boxes_camera: (N, 7) [x, y, z, h, w, l, ry] in camera coords
    Returns: (N, 8, 3)
    """
    N = boxes_camera.shape[0]
    x, y, z = boxes_camera[:, 0], boxes_camera[:, 1], boxes_camera[:, 2]
    h, w, l = boxes_camera[:, 3], boxes_camera[:, 4], boxes_camera[:, 5]
    ry = boxes_camera[:, 6]

    # 3D box corners in object coordinate (centered at bottom)
    #    3 -------- 0
    #   /|         /|
    #  2 -------- 1 .
    #  | |        | |
    #  . 7 -------- 4
    #  |/         |/
    #  6 -------- 5
    x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]).T  # (N, 8)
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h]).T  # bottom at y, top at y-h
    z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]).T

    corners = np.zeros((N, 8, 3))
    for i in range(N):
        cos_ry = np.cos(ry[i])
        sin_ry = np.sin(ry[i])
        R = np.array([[cos_ry, 0, sin_ry],
                       [0, 1, 0],
                       [-sin_ry, 0, cos_ry]])
        corners_obj = np.stack([x_corners[i], y_corners[i], z_corners[i]], axis=1)  # (8, 3)
        corners[i] = corners_obj @ R.T + np.array([x[i], y[i], z[i]])

    return corners


def project_to_image(pts_3d, P2):
    """
    Project 3D points (in camera coords) to 2D image plane.
    pts_3d: (N, 3) 
    P2: (3, 4) projection matrix
    Returns: (N, 2) pixel coordinates
    """
    pts_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1))], axis=1)  # (N, 4)
    pts_2d = (P2 @ pts_homo.T).T  # (N, 3)
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, :2]


def draw_3d_box_on_image(img, corners_2d, color, thickness=2):
    """
    Draw a 3D bounding box on an image given the 2D projected corners.
    corners_2d: (8, 2) projected corner coordinates
    """
    corners = corners_2d.astype(int)

    # Draw bottom face (0-1-2-3)
    for i in range(4):
        j = (i + 1) % 4
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)

    # Draw top face (4-5-6-7)
    for i in range(4, 8):
        j = 4 + (i - 4 + 1) % 4
        cv2.line(img, tuple(corners[i]), tuple(corners[j]), color, thickness)

    # Draw vertical pillars
    for i in range(4):
        cv2.line(img, tuple(corners[i]), tuple(corners[i + 4]), color, thickness)

    # Draw front face cross for orientation
    cv2.line(img, tuple(corners[0]), tuple(corners[5]), color, 1)
    cv2.line(img, tuple(corners[1]), tuple(corners[4]), color, 1)


def draw_bev(points, boxes_lidar, labels, class_names, scores, score_thresh=0.3,
             bev_range=(-40, -40, 80, 40), resolution=0.1):
    """
    Draw bird's eye view of point cloud with 3D boxes.
    """
    x_min, y_min, x_max, y_max = bev_range
    width = int((y_max - y_min) / resolution)
    height = int((x_max - x_min) / resolution)

    bev_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw points
    mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & \
           (points[:, 1] >= y_min) & (points[:, 1] < y_max)
    pts = points[mask]
    px = ((pts[:, 1] - y_min) / resolution).astype(int)
    py = (height - 1 - (pts[:, 0] - x_min) / resolution).astype(int)
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    bev_img[py, px] = [255, 255, 255]

    # Draw boxes
    for i in range(boxes_lidar.shape[0]):
        if scores[i] < score_thresh:
            continue

        cls_name = class_names[labels[i] - 1] if labels[i] <= len(class_names) else 'Unknown'
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))

        x, y, z, dx, dy, dz, heading = boxes_lidar[i]

        # Box corners in BEV
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        corners = np.array([
            [dx / 2, dy / 2],
            [dx / 2, -dy / 2],
            [-dx / 2, -dy / 2],
            [-dx / 2, dy / 2],
        ])

        # Rotate
        R = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        corners = (R @ corners.T).T + np.array([x, y])

        # Convert to pixel
        corners_px = ((corners[:, 1] - y_min) / resolution).astype(int)
        corners_py = (height - 1 - (corners[:, 0] - x_min) / resolution).astype(int)
        corners_pixel = np.stack([corners_px, corners_py], axis=1)

        # Draw box edges
        for j in range(4):
            k = (j + 1) % 4
            cv2.line(bev_img, tuple(corners_pixel[j]), tuple(corners_pixel[k]), color, 2)

        # Draw heading direction (front of box)
        front_center = ((corners_pixel[0] + corners_pixel[1]) / 2).astype(int)
        back_center = ((corners_pixel[2] + corners_pixel[3]) / 2).astype(int)
        cv2.arrowedLine(bev_img, tuple(back_center), tuple(front_center), color, 1, tipLength=0.3)

        # Label
        cv2.putText(bev_img, f'{cls_name} {scores[i]:.2f}',
                    (corners_pixel[0][0], corners_pixel[0][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return bev_img


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize SECOND detection results on KITTI images')
    parser.add_argument('--cfg_file', type=str, required=True, help='Config file')
    parser.add_argument('--ckpt', type=str, required=True, help='Checkpoint file')
    parser.add_argument('--data_path', type=str, default=None, help='KITTI data path')
    parser.add_argument('--output_dir', type=str, default='../output/visualizations', help='Output directory')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples to visualize')
    parser.add_argument('--score_thresh', type=float, default=0.3, help='Score threshold for drawing boxes')
    parser.add_argument('--start_idx', type=int, default=0, help='Start sample index')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine data path
    if args.data_path is not None:
        data_root = Path(args.data_path)
    else:
        data_root = Path(cfg.DATA_CONFIG.DATA_PATH)

    image_dir = data_root / 'training' / 'images'
    if not image_dir.exists():
        image_dir = data_root / 'training' / 'image_2'
    calib_dir = data_root / 'training' / 'calib'
    velodyne_dir = data_root / 'training' / 'velodyne'

    logger.info(f'Image dir: {image_dir}')
    logger.info(f'Calib dir: {calib_dir}')

    # Build dataset and dataloader (validation split)
    test_set, test_loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=2, logger=logger, training=False
    )

    # Build model
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    class_names = cfg.CLASS_NAMES
    logger.info(f'Class names: {class_names}')
    logger.info(f'Visualizing {args.num_samples} samples starting from index {args.start_idx}...')

    count = 0
    with torch.no_grad():
        for idx, data_dict in enumerate(test_loader):
            if idx < args.start_idx:
                continue
            if count >= args.num_samples:
                break

            # Get frame ID
            frame_id = data_dict['frame_id'][0]
            logger.info(f'[{count+1}/{args.num_samples}] Processing frame: {frame_id}')

            # Run inference
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()    # (N, 7) in LiDAR
            pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()  # (N,)
            pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy()  # (N,)

            # Filter by score
            mask = pred_scores >= args.score_thresh
            pred_boxes = pred_boxes[mask]
            pred_scores = pred_scores[mask]
            pred_labels = pred_labels[mask]

            # ─── Camera Image Visualization ───
            img_path = image_dir / f'{frame_id}.png'
            if not img_path.exists():
                img_path = image_dir / f'{frame_id}.jpg'
            calib_path = calib_dir / f'{frame_id}.txt'

            if img_path.exists() and calib_path.exists() and len(pred_boxes) > 0:
                img = cv2.imread(str(img_path))
                P2, R0, Tr = read_calib(str(calib_path))

                # Convert boxes to camera coords
                boxes_cam = boxes3d_lidar_to_camera(pred_boxes, R0, Tr)
                corners_3d = corners3d_from_camera_box(boxes_cam)  # (N, 8, 3)

                for i in range(len(pred_boxes)):
                    cls_name = class_names[pred_labels[i] - 1] if pred_labels[i] <= len(class_names) else 'Unknown'
                    color = CLASS_COLORS.get(cls_name, (255, 255, 255))
                    # BGR for OpenCV
                    color_bgr = (color[2], color[1], color[0])

                    # Project corners to image
                    corners_2d = project_to_image(corners_3d[i], P2)  # (8, 2)

                    # Check if the box is in front of the camera and within image bounds
                    if np.any(corners_3d[i][:, 2] < 0.1):
                        continue
                    if np.any(corners_2d[:, 0] < -500) or np.any(corners_2d[:, 0] > img.shape[1] + 500):
                        continue
                    if np.any(corners_2d[:, 1] < -500) or np.any(corners_2d[:, 1] > img.shape[0] + 500):
                        continue

                    draw_3d_box_on_image(img, corners_2d, color_bgr, thickness=2)

                    # Draw label
                    label_pos = (int(corners_2d[:, 0].min()), int(corners_2d[:, 1].min()) - 5)
                    cv2.putText(img, f'{cls_name} {pred_scores[i]:.2f}',
                                label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 2)

                # Save camera image
                out_path = output_dir / f'{frame_id}_camera.png'
                cv2.imwrite(str(out_path), img)
                logger.info(f'  Saved camera view: {out_path}')
            elif not img_path.exists():
                logger.warning(f'  Image not found: {img_path}')

            # ─── Bird's Eye View ───
            points = data_dict['points'][:, 1:4].cpu().numpy()  # (N, 3) xyz
            bev_img = draw_bev(points, pred_boxes, pred_labels, class_names, pred_scores,
                               score_thresh=args.score_thresh)
            bev_path = output_dir / f'{frame_id}_bev.png'
            cv2.imwrite(str(bev_path), bev_img)
            logger.info(f'  Saved BEV view: {bev_path}')

            count += 1

    logger.info(f'\nDone! {count} visualizations saved to: {output_dir}')
    logger.info(f'Legend:')
    for cls, color in CLASS_COLORS.items():
        logger.info(f'  {cls}: RGB{color}')


if __name__ == '__main__':
    main()
