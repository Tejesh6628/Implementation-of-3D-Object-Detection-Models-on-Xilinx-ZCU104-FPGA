#!/usr/bin/env python3
"""
run_on_board.py — Complex-YOLOv4 inference on ZCU104 DPU
=========================================================
DPU target: DPUCZDX8G_ISA1_B4096
SD image:   xilinx-zcu104-dpu-v2022.2-v3.0.0
Vitis AI:   3.0

Usage:
    python3 run_on_board.py --xmodel complex_yolov4.xmodel --image calib_0000.npy
    python3 run_on_board.py --xmodel complex_yolov4.xmodel --image_dir test_images/
"""

import os
import sys
import time
import argparse
import numpy as np

# ═══════════════════════════════════════════════════════
# Model Constants — from complex_yolov4.cfg & kitti_config.py
# ═══════════════════════════════════════════════════════
NUM_CLASSES = 8
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian",
               "Person_sitting", "Cyclist", "Tram", "Misc"]
INPUT_SIZE  = 608

# All 9 anchors (w, h) in pixels at input resolution
ALL_ANCHORS = [
    (11, 15), (10, 24), (11, 25),   # Head 1: stride=8,  mask=0,1,2
    (23, 49), (23, 55), (24, 53),   # Head 2: stride=16, mask=3,4,5
    (24, 60), (27, 63), (29, 74),   # Head 3: stride=32, mask=6,7,8
]

# (stride, anchor_mask_indices, scale_x_y)
HEAD_CONFIG = [
    ( 8, [0, 1, 2], 1.20),   # Head 1 → 76×76 grid
    (16, [3, 4, 5], 1.10),   # Head 2 → 38×38 grid
    (32, [6, 7, 8], 1.05),   # Head 3 → 19×19 grid
]

# Expected output shapes (C, H, W) per head
FM_SHAPES = [(45, 76, 76), (45, 38, 38), (45, 19, 19)]

# BEV coordinate parameters (from kitti_config.py)
BEV_WIDTH       = 608
BEV_HEIGHT      = 608
DISCRETIZATION  = 50.0 / BEV_HEIGHT   # 0.0822 meters per pixel


# ═══════════════════════════════════════════════════════
# CPU Post-processing: Decode YOLO heads
# ═══════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_head(feature_map, stride, anchor_indices, scale_xy):
    """
    Decode one YOLO detection head from raw feature map.

    Input:  feature_map — numpy [45, H, W] (NCHW, batch squeezed)
    Output: detections  — numpy [N, 15]
            columns: [cx, cy, w, h, sin_yaw, cos_yaw, conf, cls0..cls7]
    """
    num_anchors = 3
    num_attrs   = 7 + NUM_CLASSES   # = 15
    _, grid_h, grid_w = feature_map.shape

    # [45, H, W] → [3, 15, H, W] → [3, H, W, 15]
    pred = feature_map.reshape(num_anchors, num_attrs, grid_h, grid_w)
    pred = pred.transpose(0, 2, 3, 1)   # [3, H, W, 15]

    # Grid offsets
    grid_x = np.arange(grid_w).reshape(1, 1, grid_w, 1).astype(np.float32)
    grid_y = np.arange(grid_h).reshape(1, grid_h, 1, 1).astype(np.float32)

    # Decode center (x, y) with scale_x_y
    bx = (sigmoid(pred[..., 0:1]) * scale_xy - (scale_xy - 1) / 2 + grid_x) * stride
    by = (sigmoid(pred[..., 1:2]) * scale_xy - (scale_xy - 1) / 2 + grid_y) * stride

    # Decode size (w, h) from anchors
    anchors = np.array([ALL_ANCHORS[i] for i in anchor_indices], dtype=np.float32)
    aw = anchors[:, 0].reshape(num_anchors, 1, 1, 1)
    ah = anchors[:, 1].reshape(num_anchors, 1, 1, 1)
    bw = np.exp(np.clip(pred[..., 2:3], -10, 10)) * aw
    bh = np.exp(np.clip(pred[..., 3:4], -10, 10)) * ah

    # sin(yaw), cos(yaw) — raw values, no sigmoid
    im = pred[..., 4:5]
    re = pred[..., 5:6]

    # Confidence & class probabilities
    conf = sigmoid(pred[..., 6:7])
    cls  = sigmoid(pred[..., 7:])

    # Stack → [3, H, W, 15] → flatten → [3*H*W, 15]
    dets = np.concatenate([bx, by, bw, bh, im, re, conf, cls], axis=-1)
    return dets.reshape(-1, num_attrs)


def nms(detections, nms_thresh):
    """Simple IoU-based NMS (axis-aligned approximation for rotated boxes)."""
    if len(detections) == 0:
        return detections

    scores = detections[:, 6] * detections[:, 7:].max(axis=1)
    order  = scores.argsort()[::-1]

    cx, cy = detections[:, 0], detections[:, 1]
    w,  h  = detections[:, 2], detections[:, 3]
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    areas  = w * h

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        order = order[np.where(iou <= nms_thresh)[0] + 1]

    return detections[keep]


def postprocess(heads_nchw, conf_thresh, nms_thresh):
    """Decode all 3 heads, filter by confidence, apply per-class NMS."""
    all_dets = []
    for i, (stride, anchors, scale_xy) in enumerate(HEAD_CONFIG):
        fm = heads_nchw[i]
        if fm.ndim == 4:
            fm = fm[0]              # squeeze batch dim: [1,45,H,W] → [45,H,W]
        all_dets.append(decode_head(fm, stride, anchors, scale_xy))

    all_dets = np.concatenate(all_dets, axis=0)
    mask     = all_dets[:, 6] > conf_thresh
    filtered = all_dets[mask]

    if len(filtered) == 0:
        return np.zeros((0, 7 + NUM_CLASSES))

    # Per-class NMS
    cls_ids  = filtered[:, 7:].argmax(axis=1)
    final    = []
    for c in range(NUM_CLASSES):
        cls_dets = filtered[cls_ids == c]
        if len(cls_dets) > 0:
            final.append(nms(cls_dets, nms_thresh))
    return np.concatenate(final, axis=0) if final else np.zeros((0, 7 + NUM_CLASSES))


def pixel_to_bev(cx, cy, w, h):
    """Convert pixel coords to real-world BEV meters (KITTI frame)."""
    real_x = (BEV_HEIGHT - cy) * DISCRETIZATION    # forward (0–50m)
    real_y = (cx - BEV_WIDTH / 2) * DISCRETIZATION # lateral (-25 to +25m)
    real_w =  w * DISCRETIZATION
    real_h =  h * DISCRETIZATION
    return real_x, real_y, real_w, real_h


# ═══════════════════════════════════════════════════════
# DPU Runner — using VART
# ═══════════════════════════════════════════════════════
def run_dpu(xmodel_path, image_nchw):
    """
    Run DPU inference via VART.

    Input:  image_nchw — numpy float32 [1, 3, 608, 608], range [0, 1]
    Output: list of 3 numpy arrays (NCHW float32), one per YOLO head
            sorted largest-to-smallest spatial size
    """
    import xir
    import vart

    # Load graph
    graph    = xir.Graph.deserialize(xmodel_path)
    root     = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()

    dpu_subgraphs = [s for s in children
                     if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]

    if not dpu_subgraphs:
        print("ERROR: No DPU subgraph found in xmodel!")
        for s in children:
            d = s.get_attr("device") if s.has_attr("device") else "?"
            print(f"  {s.get_name()[:70]} → {d}")
        sys.exit(1)

    print(f"  DPU subgraph: {dpu_subgraphs[0].get_name()[:60]}")
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")

    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()

    print(f"  Input  tensor: {in_tensors[0].name}")
    print(f"  Input  shape : {tuple(in_tensors[0].dims)}")
    for ot in out_tensors:
        print(f"  Output tensor: {ot.name}  shape={tuple(ot.dims)}")

    # ── Prepare input ──────────────────────────────────
    in_fp  = in_tensors[0].get_attr("fix_point")
    in_scl = 2.0 ** in_fp
    print(f"  Input fix_point: {in_fp}  → scale: {in_scl}")

    # image_nchw: [1, 3, H, W] → squeeze → [3, H, W] → NHWC → [H, W, 3]
    img = image_nchw[0] if image_nchw.ndim == 4 else image_nchw
    img_nhwc = img.transpose(1, 2, 0)                              # [H, W, 3]

    # Quantize to int8 (round is slightly more accurate than truncation)
    img_int8 = np.clip(np.round(img_nhwc * in_scl), -128, 127).astype(np.int8)
    input_data = [np.expand_dims(img_int8, 0)]                     # [1, H, W, 3]

    # ── Output buffers ──────────────────────────────────
    output_data = [np.empty(tuple(ot.dims), dtype=np.int8) for ot in out_tensors]

    # ── Run DPU ─────────────────────────────────────────
    t0     = time.time()
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    dt_ms  = (time.time() - t0) * 1000

    # ── Dequantize outputs → float32 ────────────────────
    raw_fms = []
    for i, ot in enumerate(out_tensors):
        fp  = ot.get_attr("fix_point")
        scl = 2.0 ** (-fp)
        fm  = output_data[i].astype(np.float32) * scl
        print(f"  Output[{i}]: shape={fm.shape}  fix_pt={fp}  "
              f"range=[{fm.min():.3f}, {fm.max():.3f}]")
        raw_fms.append(fm)

    del runner

    # ── Convert NHWC → NCHW and sort by spatial size ────
    fms_nchw = []
    for fm in raw_fms:
        if fm.ndim == 4:
            # DPU outputs NHWC: [1, H, W, C]
            if fm.shape[-1] == 45:       # last dim is channel → NHWC
                fm = fm.transpose(0, 3, 1, 2)   # → NCHW [1, 45, H, W]
            # else: already NCHW (unlikely but handle)
        elif fm.ndim == 2:               # flat [1, N] — split manually
            pass
        fms_nchw.append(fm)

    # Handle case where DPU gives 1 flat output instead of 3 separate ones
    if len(fms_nchw) == 1 and fms_nchw[0].ndim == 2:
        flat = fms_nchw[0]               # [1, 341145]
        fms_nchw = []
        offset = 0
        for (c, h, w) in FM_SHAPES:
            sz  = c * h * w
            fm  = flat[:, offset:offset + sz].reshape(1, c, h, w)
            fms_nchw.append(fm)
            offset += sz

    # Sort: largest spatial size first (Head1 76×76, Head2 38×38, Head3 19×19)
    fms_nchw.sort(key=lambda x: x.shape[-1] * x.shape[-2], reverse=True)

    return fms_nchw, dt_ms


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='Complex-YOLOv4 ZCU104 Inference')
    p.add_argument('--xmodel',    type=str, required=True,
                   help='Path to compiled complex_yolov4.xmodel')
    p.add_argument('--image',     type=str, default=None,
                   help='Single .npy BEV image [1,3,608,608] or [3,608,608]')
    p.add_argument('--image_dir', type=str, default=None,
                   help='Directory of .npy BEV images')
    p.add_argument('--conf',      type=float, default=0.3,
                   help='Confidence threshold (default: 0.3)')
    p.add_argument('--nms',       type=float, default=0.5,
                   help='NMS IoU threshold (default: 0.5)')
    args = p.parse_args()

    if not args.image and not args.image_dir:
        print("ERROR: Provide --image or --image_dir")
        sys.exit(1)

    # Collect image paths
    if args.image:
        img_paths = [args.image]
    else:
        img_paths = sorted([os.path.join(args.image_dir, f)
                            for f in os.listdir(args.image_dir)
                            if f.endswith('.npy')])

    print("═" * 52)
    print("Complex-YOLOv4 Inference on ZCU104 (B4096 DPU)")
    print(f"  xmodel : {args.xmodel}")
    print(f"  images : {len(img_paths)} file(s)")
    print(f"  classes: {NUM_CLASSES}")
    print(f"  conf_th: {args.conf}  nms_th: {args.nms}")
    print("═" * 52)

    total_time = 0.0
    total_imgs = 0

    for img_path in img_paths:
        print(f"\n── {os.path.basename(img_path)} " + "─" * 30)

        # Load image
        img = np.load(img_path).astype(np.float32)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)   # [3,H,W] → [1,3,H,W]
        print(f"  Input : {img.shape} | range=[{img.min():.3f}, {img.max():.3f}]")

        # DPU inference
        heads, dt = run_dpu(args.xmodel, img)
        print(f"  DPU inference: {dt:.1f} ms")
        total_time += dt
        total_imgs += 1

        # CPU post-processing
        dets = postprocess(heads, args.conf, args.nms)

        if len(dets) == 0:
            print("  No detections above threshold.")
            continue

        # Print results
        W = 58
        print(f"\n  Detections ({len(dets)} found):")
        print("  " + "─" * W)
        print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  {'cx':>6} {'cy':>6} "
              f"{'w':>5} {'h':>5}  {'Yaw':>8}")
        print("  " + "─" * W)
        for j, d in enumerate(dets):
            cx, cy, w, h      = d[0], d[1], d[2], d[3]
            sin_y, cos_y      = d[4], d[5]
            conf              = d[6]
            cls_probs         = d[7:]
            cls_id            = cls_probs.argmax()
            score             = conf * cls_probs[cls_id]
            yaw_deg           = np.degrees(np.arctan2(sin_y, cos_y))
            cls_name          = CLASS_NAMES[cls_id]
            print(f"  {j+1:>3}  {cls_name:<16} {score:>6.3f}  "
                  f"{cx:>6.1f} {cy:>6.1f} {w:>5.1f} {h:>5.1f}  {yaw_deg:>+7.1f}°")

        print(f"\n  Real-world BEV coordinates (meters):")
        for j, d in enumerate(dets):
            cx, cy, w, h   = d[0], d[1], d[2], d[3]
            cls_id         = d[7:].argmax()
            score          = d[6] * d[7 + cls_id]
            yaw_deg        = np.degrees(np.arctan2(d[4], d[5]))
            rx, ry, rw, rh = pixel_to_bev(cx, cy, w, h)
            print(f"   [{j+1}] {CLASS_NAMES[cls_id]}: "
                  f"X={rx:+.1f}m  Y={ry:+.1f}m  "
                  f"W={rw:.1f}m  L={rh:.1f}m  Yaw={yaw_deg:+.1f}°")

    # Summary
    print(f"\n{'═' * 52}")
    if total_imgs > 0:
        print(f"Processed {total_imgs} image(s)")
        print(f"Avg DPU time: {total_time/total_imgs:.1f} ms/frame")
        print(f"Throughput:   {1000/(total_time/total_imgs):.1f} FPS (DPU only)")
    print("Inference complete.")
    print("═" * 52)


if __name__ == '__main__':
    main()