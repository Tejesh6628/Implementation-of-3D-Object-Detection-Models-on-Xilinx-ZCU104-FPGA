#!/usr/bin/env python3
"""
run_on_board_fixed.py — Complex-YOLOv4 inference on ZCU104 DPU
=================================================================
DPU target: DPUCZDX8G_ISA1_B4096

Usage:
    python3 run_on_board_fixed.py --xmodel complex_yolov4.xmodel \
        --image bev_000001.npy --conf 0.3 --nms 0.5 --save_img

    python3 run_on_board_fixed.py --xmodel complex_yolov4.xmodel \
        --image_dir bev_images_test/ --conf 0.3 --nms 0.5 --save_img
"""

import os, sys, time, argparse
import numpy as np

# ════════════════════════════════════════════════════
# Model Constants  (from complex_yolov4.cfg)
# ════════════════════════════════════════════════════
NUM_CLASSES = 8
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian",
               "Person_sitting", "Cyclist", "Tram", "Misc"]

# 9 anchors in (w, h) pixels — read as triplets from cfg, angle ignored here
ALL_ANCHORS = [
    (11, 15), (10, 24), (11, 25),   # Head-1 stride=8   mask=0,1,2
    (23, 49), (23, 55), (24, 53),   # Head-2 stride=16  mask=3,4,5
    (24, 60), (27, 63), (29, 74),   # Head-3 stride=32  mask=6,7,8
]

HEAD_CONFIG = [
    ( 8, [0, 1, 2]),   # 76×76 grid
    (16, [3, 4, 5]),   # 38×38 grid
    (32, [6, 7, 8]),   # 19×19 grid
]

FM_SHAPES = [(45, 76, 76), (45, 38, 38), (45, 19, 19)]

# BEV coordinate params (kitti_config.py)
BEV_WIDTH      = 608
BEV_HEIGHT     = 608
DISCRETIZATION = 50.0 / BEV_HEIGHT   # metres per pixel

# Per-class colours (BGR for OpenCV)
CLASS_COLORS = [
    (0,   255,   0),   # Car           — green
    (0,   255, 255),   # Van           — yellow
    (0,   165, 255),   # Truck         — orange
    (255,  50,  50),   # Pedestrian    — blue
    (255, 100, 180),   # Person_sitting— pink
    (180,   0, 255),   # Cyclist       — purple
    (255, 255,   0),   # Tram          — cyan
    (180, 180, 180),   # Misc          — grey
]


# ════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_head(feature_map, stride, anchor_indices):
    """
    Decode one YOLO head.
    Matches yolo_layer.py exactly — no scale_x_y applied.
    Input:  feature_map [45, H, W]  (NCHW, batch squeezed, float32)
    Output: [N, 15]
    """
    num_anchors = 3
    num_attrs   = 7 + NUM_CLASSES    # 15
    _, grid_h, grid_w = feature_map.shape

    # Make contiguous before reshape (critical after NHWC→NCHW transpose)
    pred = np.ascontiguousarray(feature_map)
    pred = pred.reshape(num_anchors, num_attrs, grid_h, grid_w)
    pred = pred.transpose(0, 2, 3, 1)          # [3, H, W, 15]

    grid_x = np.arange(grid_w, dtype=np.float32).reshape(1, 1, grid_w, 1)
    grid_y = np.arange(grid_h, dtype=np.float32).reshape(1, grid_h, 1, 1)

    bx = (sigmoid(pred[..., 0:1]) + grid_x) * stride
    by = (sigmoid(pred[..., 1:2]) + grid_y) * stride

    anchors = np.array([ALL_ANCHORS[i] for i in anchor_indices], dtype=np.float32)
    aw = (anchors[:, 0] / stride).reshape(num_anchors, 1, 1, 1)
    ah = (anchors[:, 1] / stride).reshape(num_anchors, 1, 1, 1)
    bw  = np.exp(np.clip(pred[..., 2:3], -10, 10)) * aw * stride
    bh  = np.exp(np.clip(pred[..., 3:4], -10, 10)) * ah * stride

    im   = pred[..., 4:5]
    re   = pred[..., 5:6]
    conf = sigmoid(pred[..., 6:7])
    cls  = sigmoid(pred[..., 7:])

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
    areas  = (w * h).clip(min=1e-6)
    keep   = []
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
            fm = fm[0]
        all_dets.append(decode_head(fm, stride, anchors))

    all_dets = np.concatenate(all_dets, axis=0)
    mask     = all_dets[:, 6] > conf_thresh
    filtered = all_dets[mask]
    if len(filtered) == 0:
        return np.zeros((0, 7 + NUM_CLASSES))

    cls_ids = filtered[:, 7:].argmax(axis=1)
    final   = []
    for c in range(NUM_CLASSES):
        cls_dets = filtered[cls_ids == c]
        if len(cls_dets) > 0:
            final.append(nms(cls_dets, nms_thresh))
    return np.concatenate(final, axis=0) if final else np.zeros((0, 7 + NUM_CLASSES))


def pixel_to_bev(cx, cy, w, h):
    real_x = (BEV_HEIGHT - cy) * DISCRETIZATION
    real_y = (cx - BEV_WIDTH / 2) * DISCRETIZATION
    return real_x, real_y, w * DISCRETIZATION, h * DISCRETIZATION


# ════════════════════════════════════════════════════
# Visualisation
# ════════════════════════════════════════════════════
def visualize_detections(img_nchw, detections, save_path, class_names):
    """
    Draw rotated bboxes on a BEV image and save as PNG.
    img_nchw: numpy (1, 3, H, W) float32 [0..1]
    detections: numpy (N, 15)
    """
    try:
        import cv2
    except ImportError:
        print("  [WARN] cv2 not found — skipping visualisation")
        return

    # (1,3,H,W) → (H,W,3) uint8 BGR
    img = img_nchw[0] if img_nchw.ndim == 4 else img_nchw
    img_bgr = np.ascontiguousarray((img.transpose(1, 2, 0) * 255).astype(np.uint8))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    for d in detections:
        cx,  cy  = float(d[0]), float(d[1])
        w_b, h_b = float(d[2]), float(d[3])
        sin_y, cos_y = float(d[4]), float(d[5])
        conf    = float(d[6])
        cls_id  = int(d[7:].argmax())
        score   = conf * float(d[7 + cls_id])
        yaw_deg = float(np.degrees(np.arctan2(sin_y, cos_y)))
        color   = CLASS_COLORS[cls_id]

        # Rotated bounding box
        rect = ((cx, cy), (w_b, h_b), yaw_deg)
        box  = cv2.boxPoints(rect)
        box  = np.intp(box)
        cv2.drawContours(img_bgr, [box], 0, color, 2)

        # Label
        label = f"{class_names[cls_id]} {score:.2f}"
        tx = max(int(cx) - 30, 2)
        ty = max(int(cy) - int(h_b / 2) - 4, 12)
        cv2.putText(img_bgr, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, img_bgr)
    print(f"  Image saved: {save_path}")


# ════════════════════════════════════════════════════
# DPU Runner
# ════════════════════════════════════════════════════
def run_dpu(xmodel_path, image_nchw):
    import xir, vart

    graph    = xir.Graph.deserialize(xmodel_path)
    root     = graph.get_root_subgraph()
    children = root.toposort_child_subgraph()
    dpu_sgs  = [s for s in children
                if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
    if not dpu_sgs:
        raise RuntimeError("No DPU subgraph found in xmodel!")
    runner   = vart.Runner.create_runner(dpu_sgs[0], "run")

    in_t     = runner.get_input_tensors()
    out_t    = runner.get_output_tensors()

    in_fp    = in_t[0].get_attr("fix_point")
    in_scl   = 2.0 ** in_fp

    img      = image_nchw[0] if image_nchw.ndim == 4 else image_nchw
    img_nhwc = np.ascontiguousarray(img.transpose(1, 2, 0))
    img_int8 = np.clip(np.round(img_nhwc * in_scl), -128, 127).astype(np.int8)
    input_data = [np.expand_dims(img_int8, 0)]

    output_data = [np.empty(tuple(ot.dims), dtype=np.int8) for ot in out_t]

    t0     = time.time()
    job_id = runner.execute_async(input_data, output_data)
    runner.wait(job_id)
    dt_ms  = (time.time() - t0) * 1000

    raw_fms = []
    for i, ot in enumerate(out_t):
        fp  = ot.get_attr("fix_point")
        scl = 2.0 ** (-fp)
        raw_fms.append(output_data[i].astype(np.float32) * scl)

    del runner

    fms_nchw = []
    for fm in raw_fms:
        if fm.ndim == 4 and fm.shape[-1] == 45:   # NHWC → NCHW
            fm = fm.transpose(0, 3, 1, 2)
        fms_nchw.append(fm)

    # Handle flat output (some xmodel variants)
    if len(fms_nchw) == 1 and fms_nchw[0].ndim == 2:
        flat = fms_nchw[0]
        fms_nchw = []
        offset = 0
        for (c, h, w) in FM_SHAPES:
            sz = c * h * w
            fms_nchw.append(flat[:, offset:offset+sz].reshape(1, c, h, w))
            offset += sz

    # Sort largest spatial → smallest (76×76, 38×38, 19×19)
    fms_nchw.sort(key=lambda x: x.shape[-1] * x.shape[-2], reverse=True)

    # Debug: print fix_points so we can verify quantization quality
    for i, (ot, fm) in enumerate(zip(out_t, fms_nchw)):
        fp = ot.get_attr("fix_point")
        print(f"  Output[{i}]: shape={fm.shape}  fix_pt={fp}  "
              f"range=[{fm.min():.3f}, {fm.max():.3f}]")

    return fms_nchw, dt_ms


# ════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='Complex-YOLOv4 ZCU104 Inference')
    p.add_argument('--xmodel',    type=str, required=True)
    p.add_argument('--image',     type=str, default=None)
    p.add_argument('--image_dir', type=str, default=None)
    p.add_argument('--conf',      type=float, default=0.3)
    p.add_argument('--nms',       type=float, default=0.5)
    p.add_argument('--save_img',  action='store_true',
                   help='Save detection visualisation as PNG (use cv2)')
    args = p.parse_args()

    if not args.image and not args.image_dir:
        print("ERROR: provide --image or --image_dir"); sys.exit(1)

    if args.image:
        img_paths = [args.image]
    else:
        img_paths = sorted([
            os.path.join(args.image_dir, f)
            for f in os.listdir(args.image_dir) if f.endswith('.npy')
        ])

    print("═" * 52)
    print("Complex-YOLOv4 Inference on ZCU104 (B4096 DPU)")
    print(f"  conf_th: {args.conf}  nms_th: {args.nms}")
    print("═" * 52)

    total_ms = 0.0
    for img_path in img_paths:
        print(f"\n── {os.path.basename(img_path)} " + "─" * 28)

        img = np.load(img_path).astype(np.float32)
        if img.ndim == 3:
            img = np.expand_dims(img, 0)   # add batch dim
        print(f"  Input : {img.shape} | range=[{img.min():.3f}, {img.max():.3f}]")

        heads, dt = run_dpu(args.xmodel, img)
        total_ms += dt
        print(f"  DPU inference: {dt:.1f} ms")

        dets = postprocess(heads, args.conf, args.nms)

        if len(dets) == 0:
            print("  No detections above threshold.")
            if args.save_img:
                stem = os.path.splitext(os.path.basename(img_path))[0]
                visualize_detections(img, dets, f"{stem}_det.png", CLASS_NAMES)
            continue

        W = 60
        print(f"\n  Detections ({len(dets)} found):")
        print("  " + "─" * W)
        print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  "
              f"{'cx':>6} {'cy':>6} {'w':>5} {'h':>5}  {'Yaw':>8}")
        print("  " + "─" * W)
        for j, d in enumerate(dets):
            cx, cy, w, h = d[0], d[1], d[2], d[3]
            sin_y, cos_y = d[4], d[5]
            conf         = d[6]
            cls_id       = int(d[7:].argmax())
            score        = conf * d[7 + cls_id]
            yaw_deg      = np.degrees(np.arctan2(sin_y, cos_y))
            print(f"  {j+1:>3}  {CLASS_NAMES[cls_id]:<16} {score:>6.3f}  "
                  f"{cx:>6.1f} {cy:>6.1f} {w:>5.1f} {h:>5.1f}  {yaw_deg:>+7.1f}°")

        print(f"\n  Real-world BEV coordinates (metres):")
        for j, d in enumerate(dets):
            cls_id = int(d[7:].argmax())
            rx, ry, rw, rh = pixel_to_bev(d[0], d[1], d[2], d[3])
            yaw_deg = np.degrees(np.arctan2(d[4], d[5]))
            print(f"   [{j+1}] {CLASS_NAMES[cls_id]}: "
                  f"X={rx:+.1f}m  Y={ry:+.1f}m  "
                  f"W={rw:.1f}m  L={rh:.1f}m  Yaw={yaw_deg:+.1f}°")

        if args.save_img:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            visualize_detections(img, dets, f"{stem}_det.png", CLASS_NAMES)

    n = len(img_paths)
    avg = total_ms / n if n else 0
    print(f"\n{'═'*52}")
    print(f"Processed {n} image(s)")
    print(f"Avg DPU time : {avg:.1f} ms/frame")
    print(f"Throughput   : {1000/avg:.1f} FPS (DPU only)" if avg else "")
    print(f"{'═'*52}")


if __name__ == '__main__':
    main()
