"""
SECOND on ZCU104 - Board Inference Script
==========================================
Mirrors the exact flow of test.py / eval_utils.eval_one_epoch:
  1. Load raw .bin → voxelize (CPU, matches DataProcessor)
  2. Build batch_dict with same keys the model expects
  3. Run DPU (backbone_2d + dense_head)
  4. Decode outputs → pred_boxes / pred_scores / pred_labels  (same as model.post_processing)
  5. Save per-frame KITTI .txt files in final_result/data/  (same as save_to_file)

INPUT  : directory of KITTI .bin LiDAR files
OUTPUT : per-frame .txt files with 3D bounding boxes (KITTI label format)

Copy to board:
    scp second_bev.xmodel  root@<BOARD_IP>:/home/root/second/
    scp 4_run_on_board.py  root@<BOARD_IP>:/home/root/second/

Run on board (Vitis AI 3.0 image, Python 3.8):
    python 4_run_on_board.py \
        --xmodel   /home/root/second/second_bev.xmodel \
        --bin_dir  /home/root/kitti/testing/velodyne \
        --output   /home/root/second/results \
        --score_thresh 0.3
"""

import faulthandler
faulthandler.enable()   # print C-level traceback on SIGSEGV / SIGBUS

import argparse, glob, os, time
import numpy as np
from pathlib import Path

# ── vart / xir (on Vitis AI 3.0 board image) ─────────────────────────────────
try:
    import vart, xir
    VART_OK = True
except ImportError:
    print("[WARN] vart/xir not found – CPU simulation mode")
    VART_OK = False

# =============================================================================
#  Constants (must match second.yaml + kitti_dataset.yaml exactly)
# =============================================================================
PC_RANGE      = np.array([0.0, -40.0, -3.0, 70.4, 40.0, 1.0], np.float32)
VOXEL_SIZE    = np.array([0.05, 0.05, 0.1],                    np.float32)
MAX_VOXELS    = 40000          # test split value from kitti_dataset.yaml
MAX_PTS       = 5              # MAX_POINTS_PER_VOXEL
N_FEAT        = 4              # x, y, z, intensity
GRID_SIZE     = np.round((PC_RANGE[3:] - PC_RANGE[:3]) / VOXEL_SIZE).astype(np.int32)
# GRID_SIZE = [1408, 1600, 40]  (x, y, z)

# After VoxelBackBone8x (stride-8 downsample) + HeightCompression:
C_BEV, H_BEV, W_BEV = 256, 200, 176   # channels, height(y), width(x)

# Classes from second.yaml CLASS_NAMES (index 1-based in pred_labels)
CLASS_NAMES = ['Car', 'Van', 'Truck', 'Pedestrian',
               'Person_sitting', 'Cyclist', 'Tram', 'Misc']
NUM_CLS     = len(CLASS_NAMES)

# Anchor config: (l, w, h, z_center) per class, 2 rotations each
# Matches ANCHOR_GENERATOR_CONFIG in second.yaml
ANCHOR_CFG = [
    (3.9,   1.6,  1.56, -1.78 + 1.56/2),   # Car
    (5.08,  1.87, 2.02, -1.78 + 2.02/2),   # Van
    (10.13, 2.58, 3.07, -1.78 + 3.07/2),   # Truck
    (0.8,   0.6,  1.73, -0.6  + 1.73/2),   # Pedestrian
    (0.8,   0.6,  1.27, -0.6  + 1.27/2),   # Person_sitting
    (1.76,  0.6,  1.73, -0.6  + 1.73/2),   # Cyclist
    (16.17, 2.53, 3.53, -1.78 + 3.53/2),   # Tram
    (3.0,   1.5,  1.7,  -1.0  + 1.7 /2),   # Misc
]
ROTATIONS   = [0.0, np.pi / 2]
DIR_OFFSET  = 0.78539   # π/4, from second.yaml
NUM_DIR     = 2
SCORE_THRESH_DEFAULT = 0.3
NMS_IOU     = 0.01
NMS_MAX     = 500

# =============================================================================
#  Step 1 – Pre-processing: .bin → voxels  (mirrors DataProcessor)
# =============================================================================

def load_and_filter(bin_path):
    """Load .bin and apply FOV / range filter (FOV_POINTS_ONLY=True in cfg)."""
    pts = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    mask = (pts[:, 0] >= PC_RANGE[0]) & (pts[:, 0] < PC_RANGE[3]) & \
           (pts[:, 1] >= PC_RANGE[1]) & (pts[:, 1] < PC_RANGE[4]) & \
           (pts[:, 2] >= PC_RANGE[2]) & (pts[:, 2] < PC_RANGE[5])
    return pts[mask]


def voxelize(points):
    """
    Mirrors transform_points_to_voxels (VoxelGeneratorV2 in OpenPCDet).
    Returns:
        voxels      [N, MAX_PTS, 4]
        coords      [N, 3]  in (z, y, x) order  ← matches voxel_coords in batch_dict
        num_points  [N]
    """
    coords_all = np.floor(
        (points[:, :3] - PC_RANGE[:3]) / VOXEL_SIZE
    ).astype(np.int32)                            # [M, 3] x,y,z indices
    coords_all = np.clip(coords_all, 0, GRID_SIZE - 1)

    vox_map = {}   # (zi, yi, xi) → list of point indices
    for idx in range(len(points)):
        cx, cy, cz = coords_all[idx]
        key = (cz, cy, cx)
        if key not in vox_map:
            if len(vox_map) >= MAX_VOXELS:
                continue
            vox_map[key] = []
        if len(vox_map[key]) < MAX_PTS:
            vox_map[key].append(idx)

    n = len(vox_map)
    voxels    = np.zeros((n, MAX_PTS, N_FEAT), dtype=np.float32)
    coords    = np.zeros((n, 3),               dtype=np.int32)
    num_pts   = np.zeros(n,                    dtype=np.int32)

    for i, (key, idxs) in enumerate(vox_map.items()):
        m = len(idxs)
        voxels[i, :m] = points[idxs]
        coords[i]     = key          # (z, y, x)
        num_pts[i]    = m

    return voxels, coords, num_pts


def mean_vfe(voxels, num_pts):
    """MeanVFE: average occupied points per voxel → [N, 4]."""
    feat = voxels.sum(axis=1)                              # [N, 4]
    cnt  = np.maximum(num_pts[:, None].astype(np.float32), 1.0)
    return feat / cnt                                       # [N, 4]


def scatter_to_bev(voxel_feats, coords):
    """
    HeightCompression + approximate VoxelBackBone8x output.
    Produces [C_BEV, H_BEV, W_BEV] = [256, 200, 176] for the DPU.

    NOTE: True VoxelBackBone8x uses sparse 3D convolutions (not DPU-able).
    We scatter intensity into a [40, 1600, 1408] volume then average-pool ×8.
    The first 40 channels carry real data; channels 40-255 are zero-padded.
    For full accuracy, export the sparse backbone separately and run it on CPU.
    """
    Nx, Ny, Nz = GRID_SIZE          # 1408, 1600, 40
    volume = np.zeros((Nz, Ny, Nx), dtype=np.float32)
    zi, yi, xi = coords[:, 0], coords[:, 1], coords[:, 2]
    volume[zi, yi, xi] = voxel_feats[:, 3]   # intensity as proxy feature

    # 8× average pool (stride-8 mimics VoxelBackBone8x output resolution)
    vol_ds = volume.reshape(Nz, Ny//8, 8, Nx//8, 8).mean(axis=(2, 4))
    # vol_ds: [40, 200, 176]
    pad    = np.zeros((C_BEV - Nz, H_BEV, W_BEV), dtype=np.float32)
    bev    = np.concatenate([vol_ds, pad], axis=0)   # [256, 200, 176]
    return bev


# =============================================================================
#  Step 2 – DPU runner (vart)
# =============================================================================

def load_runner(xmodel_path):
    """
    Load the DPU runner and print tensor shapes + fix_points so you can
    see them BEFORE inference starts (helps diagnose shape mismatches).
    Returns (runner, in_meta, out_meta) where *_meta are lists of dicts.
    """
    g    = xir.Graph.deserialize(xmodel_path)
    root = g.get_root_subgraph()
    dpu_sgs = [s for s in root.toposort_child_subgraph()
               if s.has_attr('device') and s.get_attr('device') == 'DPU']
    assert len(dpu_sgs) >= 1, (
        "No DPU subgraph found in xmodel!\n"
        "The xmodel was compiled with batch>1 (all ops went to CPU).\n"
        "Re-run 2b_ptq_cpu_quantize.py --mode calib then --mode test,\n"
        "then recompile with vai_c_xir."
    )
    runner = vart.Runner.create_runner(dpu_sgs[0], 'run')
    print(f"DPU runner loaded: {len(dpu_sgs)} DPU subgraph(s) found.")

    # ── Print tensor metadata so any shape mismatch is visible ───────────────
    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()

    in_meta = []
    for t in in_tensors:
        fp   = t.get_attr('fix_point')
        meta = {'name': t.name, 'dims': tuple(t.dims), 'fix_point': fp,
                'scale': float(2 ** fp)}
        in_meta.append(meta)
        print(f"  IN  {t.name}  dims={tuple(t.dims)}  fix_point={fp}  scale=2^{fp}")

    out_meta = []
    for t in out_tensors:
        fp   = t.get_attr('fix_point')
        meta = {'name': t.name, 'dims': tuple(t.dims), 'fix_point': fp,
                'scale': float(2 ** (-fp))}
        out_meta.append(meta)
        print(f"  OUT {t.name}  dims={tuple(t.dims)}  fix_point={fp}")

    return runner, in_meta, out_meta, g   # MUST keep `g` alive — see note below

    # NOTE: The vart.Runner holds internal C++ pointers into the xir.Graph's
    # memory.  If `g` is garbage-collected (e.g. it goes out of scope in the
    # caller), those pointers become dangling and execute_async() segfaults.
    # The working ComplexYOLOv4 script stores `self.graph = g` to prevent this.
    # We return `g` and keep it referenced in main() for the same reason.


def dpu_infer(runner, in_meta, out_meta, bev_chw):
    """
    bev_chw  : float32 [C, H, W]  BEV pseudo-image
    Returns dict: cls_preds, box_preds, dir_preds  (float32)

    Buffer pattern matches the WORKING ComplexYOLOv4 DPURunner exactly:
      - Buffers: np.empty(dims, dtype=np.int8, order='C')  — fresh per call
      - Input:   float32 → round(x * 2^fix_point) → clip → int8
      - Write:   input_buf[0][0, ...] = int8_data
      - Output:  int8 → float32 * 2^(-fix_point)
    """
    # ── Quantize input: float32 → int8 ───────────────────────────────────────
    in_dims  = in_meta[0]['dims']       # (1, H, W, C) NHWC
    in_scale = in_meta[0]['scale']      # 2^fix_point

    bev_nhwc = np.transpose(bev_chw, (1, 2, 0))   # [H, W, C]
    if (1,) + bev_nhwc.shape != in_dims:
        raise RuntimeError(
            f"Input shape mismatch: BEV gives (1, {bev_nhwc.shape}) "
            f"but DPU expects {in_dims}."
        )
    bev_int8 = np.clip(np.round(bev_nhwc * in_scale), -128, 127).astype(np.int8)

    # ── Allocate fresh C-contiguous int8 buffers (matches working YOLOv4) ────
    input_buf  = [np.empty(in_dims, dtype=np.int8, order='C')]
    output_buf = [np.empty(m['dims'], dtype=np.int8, order='C')
                  for m in out_meta]

    # ── Copy data into batch dim 0 ───────────────────────────────────────────
    input_buf[0][0, ...] = bev_int8

    # ── Execute on DPU ───────────────────────────────────────────────────────
    job = runner.execute_async(input_buf, output_buf)
    runner.wait(job)

    # ── Dequantize outputs: int8 → float32 ───────────────────────────────────
    # Map by tensor name (DPU compiler may reorder outputs).
    out = {}
    for i, m in enumerate(out_meta):
        name_lc = m['name'].lower()
        if 'conv_cls' in name_lc or '_cls_' in name_lc:
            key = 'cls_preds'
            feat_dim = NUM_CLS
        elif 'conv_box' in name_lc or '_box_' in name_lc:
            key = 'box_preds'
            feat_dim = 7
        elif 'conv_dir' in name_lc or '_dir_' in name_lc:
            key = 'dir_preds'
            feat_dim = 2
        else:
            key = ['cls_preds', 'box_preds', 'dir_preds'][i] if i < 3 else f'out_{i}'
            feat_dim = [NUM_CLS, 7, 2][i] if i < 3 else output_buf[i].shape[-1]
            print(f"  [WARN] Unknown output '{m['name']}' → mapped as '{key}'")

        scl_out  = m['scale']                                     # 2^(-fix_point)
        arr      = output_buf[i].astype(np.float32) * scl_out     # [1, H, W, C]
        arr      = arr[0]                                         # [H, W, C]
        out[key] = arr.reshape(-1, feat_dim)                      # [N_anchors, feat_dim]

    return out



# =============================================================================
#  Step 3 – Anchor generation (matches AnchorHeadSingle)
# =============================================================================

def build_anchors():
    """
    Returns ANCHORS [total_anchors, 7] = (x, y, z, l, w, h, rot).
    Order: class0_rot0, class0_rot1, class1_rot0, ... (per grid cell, row-major)
    Matches OpenPCDet AnchorGenerator with feature_map_stride=8.
    """
    stride = 8
    xs = np.arange(W_BEV) * VOXEL_SIZE[0] * stride + PC_RANGE[0] + VOXEL_SIZE[0] * stride / 2
    ys = np.arange(H_BEV) * VOXEL_SIZE[1] * stride + PC_RANGE[1] + VOXEL_SIZE[1] * stride / 2
    xg, yg = np.meshgrid(xs, ys)   # [H_BEV, W_BEV]

    parts = []
    for (l, w, h, zc) in ANCHOR_CFG:
        for rot in ROTATIONS:
            a = np.stack([xg, yg,
                          np.full_like(xg, zc),
                          np.full_like(xg, l),
                          np.full_like(xg, w),
                          np.full_like(xg, h),
                          np.full_like(xg, rot)], axis=-1)   # [H, W, 7]
            parts.append(a)

    # Stack along axis=2 -> [H, W, 16, 7]. Then reshape flattens to [H*W*16, 7]
    # This interleaves the 16 anchors per grid cell, exactly matching PyTorch
    return np.stack(parts, axis=2).reshape(-1, 7)    # [N_anchors, 7]


ANCHORS = build_anchors()   # built once at import


# =============================================================================
#  Step 4 – Decode boxes  (mirrors ResidualCoder + dir classifier)
# =============================================================================

def decode_boxes(box_preds, anchors):
    """ResidualCoder inverse – identical to OpenPCDet's implementation."""
    xa, ya, za = anchors[:, 0], anchors[:, 1], anchors[:, 2]
    la, wa, ha = anchors[:, 3], anchors[:, 4], anchors[:, 5]
    ra          = anchors[:, 6]
    da = np.sqrt(la**2 + wa**2)

    x   = box_preds[:, 0] * da + xa
    y   = box_preds[:, 1] * da + ya
    z   = box_preds[:, 2] * ha + za
    l   = np.exp(np.clip(box_preds[:, 3], -6, 6)) * la
    w   = np.exp(np.clip(box_preds[:, 4], -6, 6)) * wa
    h   = np.exp(np.clip(box_preds[:, 5], -6, 6)) * ha
    rot = box_preds[:, 6] + ra
    return np.stack([x, y, z, l, w, h, rot], axis=1)


def apply_dir_classifier(boxes, dir_preds):
    """Fix rotation sign with direction head (mirrors post_processing in SECONDNet)."""
    period    = 2 * np.pi / NUM_DIR
    dir_rot   = (boxes[:, 6] - DIR_OFFSET) % period + DIR_OFFSET
    dir_label = np.argmax(dir_preds, axis=-1)          # 0 or 1
    boxes[:, 6] = dir_rot + dir_label.astype(np.float32) * period
    return boxes


# =============================================================================
#  Step 5 – NMS (BEV axis-aligned approximation for CPU)
# =============================================================================

def nms_cpu(boxes, scores, iou_thresh=NMS_IOU, max_out=NMS_MAX):
    order = np.argsort(-scores)
    keep  = []
    while order.size > 0 and len(keep) < max_out:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        b   = boxes[i]
        rst = boxes[order[1:]]
        ix  = np.minimum(b[0]+b[3]/2, rst[:,0]+rst[:,3]/2) - \
              np.maximum(b[0]-b[3]/2, rst[:,0]-rst[:,3]/2)
        iy  = np.minimum(b[1]+b[4]/2, rst[:,1]+rst[:,4]/2) - \
              np.maximum(b[1]-b[4]/2, rst[:,1]-rst[:,4]/2)
        inter = np.maximum(ix, 0) * np.maximum(iy, 0)
        union = b[3]*b[4] + rst[:,3]*rst[:,4] - inter
        iou   = inter / np.maximum(union, 1e-6)
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=np.int32)


# =============================================================================
#  Step 6 – Save KITTI label file  (mirrors generate_prediction_dicts)
#            Output format identical to test.py --save_to_file
# =============================================================================

def save_prediction_txt(pred_boxes, pred_scores, pred_labels, out_path):
    """
    Writes KITTI format label file.
    pred_boxes  [N, 7]  x,y,z,l,w,h,heading  (LiDAR frame)
    pred_scores [N]
    pred_labels [N]  1-based class index
    """
    lines = []
    for i in range(len(pred_scores)):
        cls = CLASS_NAMES[int(pred_labels[i]) - 1]
        x, y, z, l, w, h, rot = pred_boxes[i]
        score = float(pred_scores[i])
        # KITTI format: type trunc occ alpha  bbox(4)  hwl  xyz  rot_y  score
        lines.append(
            f"{cls} -1 -1 -10 "
            f"0.00 0.00 1.00 1.00 "
            f"{h:.4f} {w:.4f} {l:.4f} "
            f"{x:.4f} {y:.4f} {z:.4f} "
            f"{rot:.4f} {score:.4f}"
        )
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + ('\n' if lines else ''))


# =============================================================================
#  Main inference loop  (structure mirrors eval_one_epoch in eval_utils.py)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--xmodel',      required=True)
    p.add_argument('--bin_dir',     required=True,
                   help='Dir with KITTI .bin files (same as velodyne/ folder)')
    p.add_argument('--output',      default='./results',
                   help='Output root; labels go to output/final_result/data/')
    p.add_argument('--score_thresh',type=float, default=SCORE_THRESH_DEFAULT)
    p.add_argument('--max_frames',  type=int,   default=None)
    p.add_argument('--verbose',     action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # Output directory mirrors test.py → final_result/data/
    out_label_dir = Path(args.output) / 'final_result' / 'data'
    out_label_dir.mkdir(parents=True, exist_ok=True)

    # Support both raw .bin and preprocessed .npy
    data_files = sorted(glob.glob(os.path.join(args.bin_dir, '*.npy')))
    if not data_files:
        data_files = sorted(glob.glob(os.path.join(args.bin_dir, '*.bin')))
        
    if args.max_frames:
        data_files = data_files[:args.max_frames]
    print(f"Found {len(data_files)} files in {args.bin_dir}")

    # ── Load DPU runner ───────────────────────────────────────────────────────
    if VART_OK:
        # `_graph` must be kept alive — the runner holds C++ pointers into it.
        # If it gets garbage-collected, execute_async() segfaults.
        runner, in_meta, out_meta, _graph = load_runner(args.xmodel)
    else:
        runner = in_meta = out_meta = _graph = None
        print("[SIM] No DPU – generating dummy outputs")

    t_pre = t_dpu = t_post = 0.0
    total_dets = 0

    for i, data_path in enumerate(data_files):
        frame_id = Path(data_path).stem
        print(f"[{i+1}/{len(data_files)}] {frame_id}", end="  ", flush=True)

        t0 = time.time()
        ext = Path(data_path).suffix.lower()
        if ext == '.npy':
            # ── Direct Load (Real Features) ──────────────────────────────────
            bev = np.load(data_path).astype(np.float32)
            if bev.ndim == 4:
                bev = bev[0] # remove batch dim
        else:
            # ── Legacy Pre-process (Fake Features) ───────────────────────────
            points       = load_and_filter(data_path)
            voxels, coords, num_pts = voxelize(points)
            voxel_feats  = mean_vfe(voxels, num_pts)
            bev          = scatter_to_bev(voxel_feats, coords)   # [C, H, W]
            
        t1 = time.time(); t_pre += t1 - t0

        # ── DPU inference ─────────────────────────────────────────────────────
        t2 = time.time()
        if runner is not None:
            dpu_out = dpu_infer(runner, in_meta, out_meta, bev)
        else:
            # Simulation – random tensors
            N = len(ANCHORS)
            dpu_out = {
                'cls_preds': np.random.randn(N, NUM_CLS).astype(np.float32) - 4,
                'box_preds': np.random.randn(N, 7).astype(np.float32) * 0.1,
                'dir_preds': np.random.randn(N, NUM_DIR).astype(np.float32),
            }
        t3 = time.time(); t_dpu += t3 - t2

        # ── Post-process: decode → NMS → per-class results ───────────────────
        t4 = time.time()
        cls_preds = dpu_out['cls_preds']          # [N_anchors, 8]
        box_preds = dpu_out['box_preds']          # [N_anchors, 7]
        dir_preds = dpu_out['dir_preds']          # [N_anchors, 2]

        scores_all = 1.0 / (1.0 + np.exp(-cls_preds))   # sigmoid

        all_boxes, all_scores, all_labels = [], [], []

        for cls_i in range(NUM_CLS):
            s = scores_all[:, cls_i]
            if args.verbose and i == 0:
                print(f"    Class {cls_i} max score: {s.max():.4f}")
            mask = s >= args.score_thresh
            if mask.sum() == 0:
                continue
            anc = ANCHORS[mask]
            bp  = box_preds[mask]
            dp  = dir_preds[mask]
            sc  = s[mask]

            boxes_dec = decode_boxes(bp, anc)
            boxes_dec = apply_dir_classifier(boxes_dec, dp)

            keep = nms_cpu(boxes_dec, sc)
            all_boxes.append(boxes_dec[keep])
            all_scores.append(sc[keep])
            all_labels.append(np.full(len(keep), cls_i + 1, dtype=np.int32))  # 1-based

        t5 = time.time(); t_post += t5 - t4

        # ── Save output .txt (KITTI label format) ─────────────────────────────
        if all_boxes:
            pred_boxes   = np.concatenate(all_boxes,  axis=0)
            pred_scores  = np.concatenate(all_scores, axis=0)
            pred_labels  = np.concatenate(all_labels, axis=0)
        else:
            pred_boxes  = np.zeros((0, 7))
            pred_scores = np.zeros(0)
            pred_labels = np.zeros(0, dtype=np.int32)

        label_path = out_label_dir / f'{frame_id}.txt'
        save_prediction_txt(pred_boxes, pred_scores, pred_labels, str(label_path))
        total_dets += len(pred_scores)

        if args.verbose:
            print(f"dets={len(pred_scores):3d} "
                  f"pre={1000*(t1-t0):.0f}ms "
                  f"dpu={1000*(t3-t2):.0f}ms "
                  f"post={1000*(t5-t4):.0f}ms")
        else:
            print(f"dets={len(pred_scores):3d}")

    n = max(len(data_files), 1)
    print(f"\n=== Summary ({n} frames, {total_dets} total detections) ===")
    print(f"  Avg pre-process : {1000*t_pre/n:.1f} ms")
    print(f"  Avg DPU         : {1000*t_dpu/n:.1f} ms")
    print(f"  Avg post-proc   : {1000*t_post/n:.1f} ms")
    print(f"  Avg total       : {1000*(t_pre+t_dpu+t_post)/n:.1f} ms")
    print(f"  Labels saved to : {out_label_dir}")


if __name__ == '__main__':
    main()
