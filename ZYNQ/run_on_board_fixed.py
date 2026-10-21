#!/usr/bin/env python3
"""
run_on_board.py — Complex-YOLOv4 inference on ZCU104 DPU
=========================================================
DPU target: DPUCZDX8G_ISA1_B4096
Usage:
    python3 run_on_board.py --xmodel complex_yolov4.xmodel --image bev_000001.npy
    python3 run_on_board.py --xmodel complex_yolov4.xmodel --image_dir bev_images_test/
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
# Triplet format from cfg: (w, h, angle) - angle is not used in size decode
ALL_ANCHORS = [
    (11, 15), (10, 24), (11, 25),   # Head 1: stride=8,  mask=0,1,2
    (23, 49), (23, 55), (24, 53),   # Head 2: stride=16, mask=3,4,5
    (24, 60), (27, 63), (29, 74),   # Head 3: stride=32, mask=6,7,8
]
# (stride, anchor_mask_indices)
HEAD_CONFIG = [
    ( 8, [0, 1, 2]),   # Head 1 → 76×76 grid
    (16, [3, 4, 5]),   # Head 2 → 38×38 grid
    (32, [6, 7, 8]),   # Head 3 → 19×19 grid
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
def decode_head(feature_map, stride, anchor_indices):
    """
    Decode one YOLO detection head.
    Matches yolo_layer.py forward() EXACTLY — no scale_x_y applied.
    Input:  feature_map — numpy [45, H, W] (NCHW, batch squeezed)
    Output: detections  — numpy [N, 15]
    """
    num_anchors = 3
    num_attrs   = 7 + NUM_CLASSES   # = 15
    _, grid_h, grid_w = feature_map.shape
    # [45, H, W] → [3, 15, H, W] → [3, H, W, 15]  (matches yolo_layer permute)
    pred = feature_map.reshape(num_anchors, num_attrs, grid_h, grid_w)
    pred = pred.transpose(0, 2, 3, 1)   # [3, H, W, 15]
    # Grid offsets — matches yolo_layer.py compute_grid_offsets
    grid_x = np.arange(grid_w, dtype=np.float32).reshape(1, 1, grid_w, 1)
    grid_y = np.arange(grid_h, dtype=np.float32).reshape(1, grid_h, 1, 1)
    # Decode x, y — plain sigmoid + grid offset, then scale by stride
    bx = (sigmoid(pred[..., 0:1]) + grid_x) * stride
    by = (sigmoid(pred[..., 1:2]) + grid_y) * stride
    # Decode w, h from anchors
    anchors = np.array([ALL_ANCHORS[i] for i in anchor_indices], dtype=np.float32)
    aw = (anchors[:, 0] / stride).reshape(num_anchors, 1, 1, 1)
    ah = (anchors[:, 1] / stride).reshape(num_anchors, 1, 1, 1)
    
    bw = np.exp(np.clip(pred[..., 2:3], -10, 10)) * aw * stride
    bh = np.exp(np.clip(pred[..., 3:4], -10, 10)) * ah * stride
    # im, re — raw values, no activation
    im = pred[..., 4:5]
    re = pred[..., 5:6]
    # Confidence & classes
    conf = sigmoid(pred[..., 6:7])
    cls  = sigmoid(pred[..., 7:])
    # Stack and flatten
    dets = np.concatenate([bx, by, bw, bh, im, re, conf, cls], axis=-1)
    return dets.reshape(-1, num_attrs)
def nms(detections, nms_thresh):
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
    all_dets = []
    for i, (stride, anchors) in enumerate(HEAD_CONFIG):
        fm = heads_nchw[i]
        if fm.ndim == 4:
            fm = fm[0]              # squeeze batch dim
        all_dets.append(decode_head(fm, stride, anchors))
    all_dets = np.concatenate(all_dets, axis=0)
    mask     = all_dets[:, 6] > conf_thresh
    filtered = all_dets[mask]
    if len(filtered) == 0:
        return np.zeros((0, 7 + NUM_CLASSES))
    cls_ids  = filtered[:, 7:].argmax(axis=1)
    final    = []
    for c in range(NUM_CLASSES):
        cls_dets = filtered[cls_ids == c]
        if len(cls_dets) > 0:
            final.append(nms(cls_dets, nms_thresh))
    return np.concatenate(final, axis=0) if final else np.zeros((0, 7 + NUM_CLASSES))
def pixel_to_bev(cx, cy, w, h):
    real_x = (BEV_HEIGHT - cy) * DISCRETIZATION
    real_y = (cx - BEV_WIDTH / 2) * DISCRETIZATION
    real_w =  w * DISCRETIZATION
    real_h =  h * DISCRETIZATION
    return real_x, real_y, real_w, real_h
# ═══════════════════════════════════════════════════════
# DPU Runner
# ═══════════════════════════════════════════════════════
def run_dpu(xmodel_path, image_nchw):
    import xir
    import vart
    graph    = xir.Graph.deserialize(xmodel_path)
    root     = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_subgraphs = [s for s in children if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    runner = vart.Runner.create_runner(dpu_subgraphs[0], "run")
    in_tensors  = runner.get_input_tensors()
    out_tensors = runner.get_output_tensors()
    in_fp  = in_tensors[0].get_attr("fix_point")
    in_scl = 2.0 ** in_fp
    img = image_nchw[0] if image_nchw.ndim == 4 else image_nchw
    img_nhwc = img.transpose(1, 2, 0)                              
    img_int8 = np.clip(np.round(img_nhwc * in_scl), -128, 127).astype(np.int8)
    input_data = [np.expand_dims(img_int8, 0)]                     
    output_data = [np.empty(tuple(ot.dims), dtype=np.int8) for ot in out_tensors]
    t0     = time.time()
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    dt_ms  = (time.time() - t0) * 1000
    raw_fms = []
    for i, ot in enumerate(out_tensors):
        fp  = ot.get_attr("fix_point")
        scl = 2.0 ** (-fp)
        fm  = output_data[i].astype(np.float32) * scl
        raw_fms.append(fm)
    del runner
    fms_nchw = []
    for fm in raw_fms:
        if fm.ndim == 4:
            if fm.shape[-1] == 45:       
                fm = fm.transpose(0, 3, 1, 2)   
        fms_nchw.append(fm)
    if len(fms_nchw) == 1 and fms_nchw[0].ndim == 2:
        flat = fms_nchw[0]               
        fms_nchw = []
        offset = 0
        for (c, h, w) in FM_SHAPES:
            sz  = c * h * w
            fm  = flat[:, offset:offset + sz].reshape(1, c, h, w)
            fms_nchw.append(fm)
            offset += sz
    fms_nchw.sort(key=lambda x: x.shape[-1] * x.shape[-2], reverse=True)
    return fms_nchw, dt_ms
# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='Complex-YOLOv4 ZCU104 Inference')
    p.add_argument('--xmodel',    type=str, required=True)
    p.add_argument('--image',     type=str, default=None)
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--conf',      type=float, default=0.3)
    p.add_argument('--nms',       type=float, default=0.5)
    args = p.parse_args()
    if not args.image and not args.image_dir:
        print("ERROR: Provide --image or --image_dir")
        sys.exit(1)
    if args.image:
        img_paths = [args.image]
    else:
        img_paths = sorted([os.path.join(args.image_dir, f) for f in os.listdir(args.image_dir) if f.endswith('.npy')])
    print("═" * 52)
    print("Complex-YOLOv4 Inference on ZCU104 (B4096 DPU)")
    print(f"  conf_th: {args.conf}  nms_th: {args.nms}")
    print("═" * 52)
    for img_path in img_paths:
        print(f"\n── {os.path.basename(img_path)} " + "─" * 30)
        img = np.load(img_path).astype(np.float32)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)   
        print(f"  Input : {img.shape} | range=[{img.min():.3f}, {img.max():.3f}]")
        heads, dt = run_dpu(args.xmodel, img)
        print(f"  DPU inference: {dt:.1f} ms")
        dets = postprocess(heads, args.conf, args.nms)
        if len(dets) == 0:
            print("  No detections above threshold.")
            continue
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
if __name__ == '__main__':
    main()
