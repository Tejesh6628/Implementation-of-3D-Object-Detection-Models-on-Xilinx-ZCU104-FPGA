"""
STEP 2a: Prepare Calibration Data
==================================
Run on HOST inside Vitis AI 3.0 Docker container.

Generates NumPy BEV pseudo-images from KITTI .bin point clouds for
calibration.  The voxelization + MeanVFE + HeightCompression steps that
SECOND normally does in GPU are replicated here in pure NumPy so we produce
the same float32 tensor that the DPU will receive.

Output: ./calib_data/bev_*.npy   (shape [1, 256, 200, 176])

Usage (inside Vitis AI Docker):
    python 2a_prepare_calibration.py \
        --kitti_root /path/to/kitti/training \
        --num_samples 200 \
        --output_dir ./calib_data

The calibration images are consumed by 2b_quantize_pytorch.py.
"""

import argparse, glob, os
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Voxelization parameters (must match kitti_dataset.yaml / second.yaml) ────
POINT_CLOUD_RANGE = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 1.0])
VOXEL_SIZE        = np.array([0.05, 0.05, 0.1])
MAX_VOXELS        = 40000
MAX_PTS_PER_VOXEL = 5
NUM_POINT_FEATURES = 4      # x, y, z, intensity
C_BEV             = 256     # NUM_BEV_FEATURES (after HeightCompression)
# Grid dimensions after 8× sparse backbone:
# x_range=70.4/0.05=1408  → /8 = 176
# y_range=80/0.05 =1600  → /8 = 200
H_BEV, W_BEV = 200, 176


def compute_grid_size(pc_range, voxel_size):
    return np.round((pc_range[3:] - pc_range[:3]) / voxel_size).astype(np.int32)


GRID_SIZE = compute_grid_size(POINT_CLOUD_RANGE, VOXEL_SIZE)  # [1408, 1600, 40]


def voxelize(points, pc_range, voxel_size, grid_size,
             max_voxels, max_pts):
    """Scatter points to voxels (CPU, matches OpenPCDet VoxelGeneratorV2)."""
    mask = np.all(
        (points[:, :3] >= pc_range[:3]) &
        (points[:, :3] <  pc_range[3:]), axis=1)
    points = points[mask]

    coords = np.floor((points[:, :3] - pc_range[:3]) / voxel_size
                      ).astype(np.int32)           # [N, 3] → (x, y, z)
    coords = np.clip(coords, 0, grid_size - 1)

    # Build voxel dictionary
    voxel_dict = {}
    for i, c in enumerate(coords):
        key = (c[2], c[1], c[0])   # (z, y, x) → matches scatter ordering
        if key not in voxel_dict:
            voxel_dict[key] = []
        if len(voxel_dict[key]) < max_pts:
            voxel_dict[key].append(points[i])
        if len(voxel_dict) >= max_voxels:
            break

    keys   = list(voxel_dict.keys())[:max_voxels]
    n_vox  = len(keys)
    voxels = np.zeros((n_vox, max_pts, NUM_POINT_FEATURES), dtype=np.float32)
    coords_out = np.zeros((n_vox, 3), dtype=np.int32)
    num_pts    = np.zeros(n_vox, dtype=np.int32)

    for i, k in enumerate(keys):
        pts = np.stack(voxel_dict[k])
        n   = pts.shape[0]
        voxels[i, :n] = pts
        coords_out[i] = k      # (z, y, x)
        num_pts[i]    = n

    return voxels, coords_out, num_pts


def mean_vfe(voxels, num_pts):
    """MeanVFE: average non-empty points per voxel."""
    # voxels: [N, max_pts, 4]
    pts_mean = voxels[:, :, :4].sum(axis=1)
    mask_sum = np.maximum(num_pts[:, None].astype(np.float32), 1.0)
    return pts_mean / mask_sum   # [N, 4]


def height_compression(voxel_features, coords, grid_size):
    """
    Scatter voxel features into a dense BEV map.
    Mimics HeightCompression (uses z-axis for channel).
    grid_size = [Nx=1408, Ny=1600, Nz=40]
    BEV output: [Nz*C_feat, Ny, Nx] → after backbone conv to [256, 200, 176]

    Since we want the *input* to the 2D backbone (which is 256-ch after
    HeightCompression+VoxelBackBone8x), we produce the intermediate
    dense tensor and note that the DPU will start from AFTER the sparse
    3D backbone. Therefore here we produce the full spatial_features [256, 200, 176].

    NOTE: Because we cannot run the sparse VoxelBackBone8x in NumPy easily,
    we approximate it by building a 40-channel BEV then zero-padding to 256.
    For REAL calibration you should use the GPU model's intermediate output.
    See the comment below for the preferred approach.
    """
    Nx, Ny, Nz = grid_size   # 1408, 1600, 40
    bev = np.zeros((Nz, Ny, Nx), dtype=np.float32)

    z_idx = coords[:, 0]
    y_idx = coords[:, 1]
    x_idx = coords[:, 2]

    bev[z_idx, y_idx, x_idx] = voxel_features[:, 0]  # use intensity as proxy

    # Downsample 8× (mimics VoxelBackBone8x stride-8 output)
    from skimage.transform import downscale_local_mean
    bev_ds = downscale_local_mean(bev, (1, 8, 8))    # [40, 200, 176]

    # Pad channels to 256 (the rest are learned by the sparse backbone)
    pad = np.zeros((256 - Nz, bev_ds.shape[1], bev_ds.shape[2]),
                   dtype=np.float32)
    bev_256 = np.concatenate([bev_ds, pad], axis=0)  # [256, 200, 176]
    return bev_256


def load_point_cloud(file_path):
    """
    Load a single point cloud file.
    Supports:
      .bin  -> KITTI binary float32 format  (x, y, z, intensity per point)
      .npy  -> numpy array, shape [N, 3] or [N, 4]
    Returns float32 array of shape [N, 4] (pads intensity=0 if only 3 cols).
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.bin':
        pts = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    elif ext == '.npy':
        pts = np.load(file_path).astype(np.float32)
        if pts.ndim == 1:
            # Some datasets flatten: reshape assuming 4 cols
            try:
                pts = pts.reshape(-1, 4)
            except ValueError:
                pts = pts.reshape(-1, 3)
        if pts.shape[1] == 3:              # x,y,z only — add dummy intensity
            pts = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1)
        elif pts.shape[1] > 4:            # keep only first 4 columns
            pts = pts[:, :4]
    else:
        raise ValueError(f'Unsupported point cloud format: {ext}')
    return pts


def process_point_cloud(file_path):
    """Full pre-processing pipeline for one point cloud file → BEV [1,256,200,176]."""
    points = load_point_cloud(file_path)

    # ── FOV filter (KITTI front-camera cone) ───────────────────────────────
    mask = np.all((points[:, :3] >= POINT_CLOUD_RANGE[:3]) &
                  (points[:, :3] <  POINT_CLOUD_RANGE[3:]), axis=1)
    points = points[mask]

    if len(points) == 0:
        print(f'  [WARN] {os.path.basename(file_path)}: 0 points after FOV filter — saving zeros')
        return np.zeros((1, C_BEV, H_BEV, W_BEV), dtype=np.float32)

    voxels, coords, num_pts = voxelize(
        points, POINT_CLOUD_RANGE, VOXEL_SIZE, GRID_SIZE,
        MAX_VOXELS, MAX_PTS_PER_VOXEL)

    vf  = mean_vfe(voxels, num_pts)                   # [N, 4]
    bev = height_compression(vf, coords, GRID_SIZE)   # [256, 200, 176]
    return bev[np.newaxis]                             # [1, 256, 200, 176]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--kitti_root', required=True,
                   help='Path to kitti/training directory')
    p.add_argument('--num_samples', type=int, default=200,
                   help='Number of frames for calibration (200 is enough)')
    p.add_argument('--output_dir',  default='./calib_data')
    return p.parse_args()


def find_point_cloud_files(kitti_root, num_samples):
    """
    Auto-detect point cloud files under kitti_root.
    Search order:
      1. <kitti_root>/velodyne/*.bin   (standard KITTI layout)
      2. <kitti_root>/velodyne/*.npy   (numpy KITTI layout)
      3. <kitti_root>/*.npy            (flat numpy layout)
      4. <kitti_root>/*.bin            (flat binary layout)
    Prints a diagnostic summary so you can see exactly what was found.
    """
    root = kitti_root
    search_patterns = [
        (os.path.join(root, 'velodyne', '*.bin'), '.bin in velodyne/'),
        (os.path.join(root, 'velodyne', '*.npy'), '.npy in velodyne/'),
        (os.path.join(root, '*.npy'),             '.npy in root'),
        (os.path.join(root, '*.bin'),             '.bin in root'),
    ]

    print(f'\nSearching for point cloud files under: {root}')
    for pattern, label in search_patterns:
        matches = sorted(glob.glob(pattern))
        print(f'  {label:25s}: {len(matches)} files found')
        if matches:
            chosen = matches[:num_samples]
            print(f'  -> Using {len(chosen)} files (pattern: {pattern})')
            print(f'     First file : {chosen[0]}')
            if len(chosen) > 1:
                print(f'     Last file  : {chosen[-1]}')
            return chosen

    # Nothing found — print directory contents for debugging
    print('\n[ERROR] No point cloud files found!')
    print('Directory contents:')
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.replace(root, '').count(os.sep)
        if depth > 2:
            break
        indent = '  ' * depth
        print(f'{indent}{os.path.basename(dirpath)}/')
        subindent = '  ' * (depth + 1)
        exts = {}
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            exts[ext] = exts.get(ext, 0) + 1
        for ext, count in sorted(exts.items()):
            print(f'{subindent}{count} x {ext} files')
    return []


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = find_point_cloud_files(args.kitti_root, args.num_samples)

    if not files:
        print('\n[HINT] Common fixes:')
        print('  1. Check --kitti_root points to the training/ directory')
        print('     e.g. --kitti_root /data/kitti/training')
        print('  2. velodyne/ subfolder should contain .bin or .npy files')
        print('  3. For kitti_np datasets the files may be directly in the root')
        return

    print(f'\nProcessing {len(files)} frames ...')
    ok = 0
    for i, fpath in enumerate(tqdm(files)):
        try:
            bev = process_point_cloud(fpath)
            np.save(str(out / f'bev_{i:04d}.npy'), bev)
            ok += 1
        except Exception as e:
            print(f'  [ERROR] {os.path.basename(fpath)}: {e}')

    print(f'\nDone. Saved {ok}/{len(files)} BEV images to {out}')
    if ok > 0:
        print(f'Output files: bev_0000.npy ... bev_{ok-1:04d}.npy  shape [1,{C_BEV},{H_BEV},{W_BEV}]')
    print('\n[IMPORTANT] For accurate quantization, prefer running the GPU')
    print('forward pass through backbone_3d first and capturing its output.')
    print('See the comment in height_compression() for details.')


if __name__ == '__main__':
    main()
