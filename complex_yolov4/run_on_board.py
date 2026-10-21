#!/usr/bin/env python3
"""
run_on_board.py  —  Complex-YOLOv4 inference on ZCU104 DPU
============================================================
DPU target : DPUCZDX8G_ISA1_B4096

KEY CHANGES vs previous version:
  - xmodel loaded ONCE at startup (not per-image) → stable runner
  - faulthandler enabled → C-level traceback on segfault
  - Buffer allocation follows official Xilinx example pattern exactly
  - fix_point read with fallback (some VART versions lack get_attr)

Supports two input modes:
  --image  path.npy      Pre-generated BEV as float32 numpy [3,608,608]
  --velodyne path.bin    Raw LIDAR point cloud (BEV generated on board)

Usage:
  python3 run_on_board.py --xmodel complex_yolov4.xmodel \\
      --image bev_images_test/bev_000001.npy --conf 0.3 --save_img
"""

import faulthandler
faulthandler.enable()   # print C-level traceback on SIGSEGV

import os, sys, time, argparse, math
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Optional: use source config / BEV utils if copied alongside this script
# ══════════════════════════════════════════════════════════════════════
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, 'src'))

try:
    import config.kitti_config as cnf
    BEV_HEIGHT     = cnf.BEV_HEIGHT
    BEV_WIDTH      = cnf.BEV_WIDTH
    DISCRETIZATION = cnf.DISCRETIZATION
    BOUNDARY       = cnf.boundary
    CLASS_NAMES    = cnf.class_list
    CLASS_COLORS   = [tuple(reversed(c)) for c in cnf.colors]
    print("[INFO] Using config/kitti_config.py from source.")
except ImportError:
    BEV_HEIGHT     = 608
    BEV_WIDTH      = 608
    DISCRETIZATION = 50.0 / 608
    BOUNDARY = {"minX":0,"maxX":50,"minY":-25,"maxY":25,"minZ":-2.73,"maxZ":1.27}
    CLASS_NAMES  = ["Car","Van","Truck","Pedestrian",
                    "Person_sitting","Cyclist","Tram","Misc"]
    CLASS_COLORS = [
        (255,255,0),(255,0,0),(0,0,255),(0,255,255),
        (255,0,255),(0,255,0),(255,0,128),(0,128,255),
    ]

try:
    from data_process.kitti_bev_utils import makeBVFeature, drawRotatedBox
    _have_bev = True
    print("[INFO] Using data_process/kitti_bev_utils.py from source.")
except ImportError:
    _have_bev = False


# ══════════════════════════════════════════════════════════════════════
# Anchors from complex_yolov4.cfg  (w, h, sin(angle), cos(angle))
# angle=0 for all → sin=0, cos=1
# ══════════════════════════════════════════════════════════════════════
_anchor_raw = [
    11,15,0,  10,24,0,  11,25,0,    # indices 0,1,2 → stride-8  (76×76)
    23,49,0,  23,55,0,  24,53,0,    # indices 3,4,5 → stride-16 (38×38)
    24,60,0,  27,63,0,  29,74,0,    # indices 6,7,8 → stride-32 (19×19)
]
ALL_ANCHORS = [
    (_anchor_raw[i], _anchor_raw[i+1],
     math.sin(math.radians(_anchor_raw[i+2])),
     math.cos(math.radians(_anchor_raw[i+2])))
    for i in range(0, len(_anchor_raw), 3)
]

HEAD_CONFIG = [
    (8,  [0,1,2]),   # 76×76
    (16, [3,4,5]),   # 38×38
    (32, [6,7,8]),   # 19×19
]
NUM_CLASSES = len(CLASS_NAMES)


# ══════════════════════════════════════════════════════════════════════
# BEV generator fallback (identical to kitti_bev_utils.makeBVFeature)
# ══════════════════════════════════════════════════════════════════════
def make_bev_feature(velo_path):
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    bc  = BOUNDARY
    mask = ((pts[:,0] >= bc['minX']) & (pts[:,0] <= bc['maxX']) &
            (pts[:,1] >= bc['minY']) & (pts[:,1] <= bc['maxY']) &
            (pts[:,2] >= bc['minZ']) & (pts[:,2] <= bc['maxZ']))
    pts = pts[mask]
    pts[:,2] -= bc['minZ']
    H = BEV_HEIGHT + 1; W = BEV_WIDTH + 1; D = DISCRETIZATION
    row = np.clip(np.floor(pts[:,0]/D).astype(np.int32), 0, H-1)
    col = np.clip((np.floor(pts[:,1]/D) + W/2).astype(np.int32), 0, W-1)
    max_z    = float(abs(bc['maxZ'] - bc['minZ']))
    idx_sort = np.lexsort((-pts[:,2], col, row))
    pts_s    = pts[idx_sort]; row_s = row[idx_sort]; col_s = col[idx_sort]
    _, uni   = np.unique(np.stack([row_s,col_s],axis=1), axis=0, return_index=True)
    hmap     = np.zeros((H,W), dtype=np.float32)
    hmap[row_s[uni], col_s[uni]] = pts_s[uni,2] / max_z
    _, uni2, counts = np.unique(np.stack([row,col],axis=1), axis=0,
                                return_index=True, return_counts=True)
    imap = np.zeros((H,W), dtype=np.float32)
    dmap = np.zeros((H,W), dtype=np.float32)
    imap[row[uni2], col[uni2]] = pts[uni2,3]
    dmap[row[uni2], col[uni2]] = np.minimum(1.0, np.log(counts+1)/np.log(64))
    rgb = np.zeros((3, BEV_HEIGHT, BEV_WIDTH), dtype=np.float32)
    rgb[2] = dmap[:BEV_HEIGHT, :BEV_WIDTH]
    rgb[1] = hmap[:BEV_HEIGHT, :BEV_WIDTH]
    rgb[0] = imap[:BEV_HEIGHT, :BEV_WIDTH]
    return rgb


# ══════════════════════════════════════════════════════════════════════
# Decode + NMS
# ══════════════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_head(fm_nchw, stride, anchor_indices):
    """
    Decode one YOLO head from NCHW float32 feature map.
    im, re are RAW — NOT sigmoided (matches yolo_layer.py exactly).
    """
    C, H, W = fm_nchw.shape
    pred = np.ascontiguousarray(fm_nchw).reshape(3, 15, H, W)
    pred = pred.transpose(0, 2, 3, 1)   # [3, H, W, 15]

    gx = np.arange(W, dtype=np.float32).reshape(1, 1, W, 1)
    gy = np.arange(H, dtype=np.float32).reshape(1, H, 1, 1)
    anch = [ALL_ANCHORS[i] for i in anchor_indices]
    aw = np.array([a[0]/stride for a in anch], dtype=np.float32).reshape(3,1,1,1)
    ah = np.array([a[1]/stride for a in anch], dtype=np.float32).reshape(3,1,1,1)

    bx   = (sigmoid(pred[...,0:1]) + gx) * stride
    by   = (sigmoid(pred[...,1:2]) + gy) * stride
    bw   = np.exp(np.clip(pred[...,2:3], -10, 10)) * aw * stride
    bh   = np.exp(np.clip(pred[...,3:4], -10, 10)) * ah * stride
    im   = pred[...,4:5]          # raw
    re   = pred[...,5:6]          # raw
    conf = sigmoid(pred[...,6:7])
    cls  = sigmoid(pred[...,7:])

    out = np.concatenate([bx,by,bw,bh,im,re,conf,cls], axis=-1)
    return out.reshape(-1, 15)


def nms_per_class(dets, nms_thresh):
    if len(dets) == 0:
        return np.zeros((0,15))
    scores  = dets[:,6] * dets[:,7:].max(axis=1)
    cls_ids = dets[:,7:].argmax(axis=1)
    order   = scores.argsort()[::-1]
    dets, scores, cls_ids = dets[order], scores[order], cls_ids[order]
    cx,cy = dets[:,0], dets[:,1]
    w,h   = dets[:,2], dets[:,3]
    x1,y1 = cx-w/2, cy-h/2
    x2,y2 = cx+w/2, cy+h/2
    areas  = (w*h).clip(1e-6)
    used   = np.zeros(len(dets), dtype=bool)
    keep   = []
    for i in range(len(dets)):
        if used[i]: continue
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[i+1:])
        iy1 = np.maximum(y1[i], y1[i+1:])
        ix2 = np.minimum(x2[i], x2[i+1:])
        iy2 = np.minimum(y2[i], y2[i+1:])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou   = inter / (areas[i] + areas[i+1:] - inter + 1e-16)
        used[i+1:] |= (iou > nms_thresh) & (cls_ids[i+1:] == cls_ids[i])
    return dets[keep]


def postprocess(fms, conf_thresh, nms_thresh):
    all_dets = []
    for i, (stride, anch_idx) in enumerate(HEAD_CONFIG):
        fm = fms[i]
        if fm.ndim == 4: fm = fm[0]
        all_dets.append(decode_head(fm, stride, anch_idx))
    all_dets = np.concatenate(all_dets, axis=0)
    return nms_per_class(all_dets[all_dets[:,6] > conf_thresh], nms_thresh)


# ══════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════
def _draw_box(img_bgr, cx, cy, w, h, yaw, color):
    try:
        import cv2
        rect = ((float(cx),float(cy)), (float(w),float(h)), float(np.degrees(yaw)))
        box  = cv2.boxPoints(rect); box = np.intp(box)
        cv2.drawContours(img_bgr, [box], 0, color, 2)
        cv2.line(img_bgr, tuple(box[0]), tuple(box[3]), (255,255,0), 2)
    except Exception:
        pass


def visualize(img_nchw, dets, save_path):
    try:
        import cv2
    except ImportError:
        print("  [WARN] cv2 not available — skipping visualisation.")
        return
    img     = img_nchw[0] if img_nchw.ndim == 4 else img_nchw
    img_bgr = np.ascontiguousarray((img.transpose(1,2,0)*255).astype(np.uint8))
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    for d in dets:
        cx,cy,w,h  = float(d[0]),float(d[1]),float(d[2]),float(d[3])
        yaw        = np.arctan2(float(d[4]), float(d[5]))
        cls_id     = int(d[7:].argmax())
        score      = float(d[6]) * float(d[7+cls_id])
        color      = CLASS_COLORS[cls_id]
        if _have_bev:
            drawRotatedBox(img_bgr, cx, cy, w, h, yaw, color)
        else:
            _draw_box(img_bgr, cx, cy, w, h, yaw, color)
        cv2.putText(img_bgr, f"{CLASS_NAMES[cls_id]} {score:.2f}",
                    (max(int(cx)-30,2), max(int(cy)-int(h/2)-5,12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.imwrite(save_path, img_bgr)
    print(f"  Saved : {save_path}")


# ══════════════════════════════════════════════════════════════════════
# DPU runner class  —  load once, run many times
# ══════════════════════════════════════════════════════════════════════
class DPURunner:
    """
    Loads the xmodel once at construction.
    Call .run(img_nchw) for each image.

    This pattern (single runner, persistent buffers) is what the official
    Xilinx/Vitis-AI examples use and is stable on ZCU104.
    """
    def __init__(self, xmodel_path):
        import xir, vart

        print(f"[DPU] Loading: {xmodel_path}")
        self.graph    = xir.Graph.deserialize(xmodel_path)
        self.root     = self.graph.get_root_subgraph()
        self.children = self.root.toposort_child_subgraph()
        self.dpu_sgs  = [s for s in self.children
                    if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if not self.dpu_sgs:
            raise RuntimeError("No DPU subgraph found in xmodel!")

        self._runner = vart.Runner.create_runner(self.dpu_sgs[0], "run")

        self._in_tensors  = self._runner.get_input_tensors()
        self._out_tensors = self._runner.get_output_tensors()

        # Read fix_points (with fallback)
        try:
            self._in_fp = self._in_tensors[0].get_attr("fix_point")
        except Exception:
            self._in_fp = 7   # safe default for [0,1] input → scale=128
            print(f"[DPU] fix_point not available via get_attr, using default={self._in_fp}")

        self._out_fps = []
        for ot in self._out_tensors:
            try:
                self._out_fps.append(ot.get_attr("fix_point"))
            except Exception:
                self._out_fps.append(2)   # safe default
                print("[DPU] Output fix_point fallback=2")

        # Pre-allocate persistent C-contiguous buffers (official Xilinx pattern)
        # Input is always NHWC on ZCU104
        self._in_dims  = tuple(self._in_tensors[0].dims)
        self._out_dims = [tuple(t.dims) for t in self._out_tensors]

        self._input_buf  = [np.empty(self._in_dims, dtype=np.int8, order='C')]
        self._output_buf = [np.empty(d, dtype=np.int8, order='C')
                            for d in self._out_dims]

        print(f"[DPU] Input  : {self._in_dims}  fix_point={self._in_fp}"
              f"  scale={2**self._in_fp}")
        for i, (d, fp) in enumerate(zip(self._out_dims, self._out_fps)):
            print(f"[DPU] Output[{i}]: {d}  fix_point={fp}"
                  f"  scale={2**-fp:.6f}")

    def run(self, img_nchw):
        """
        img_nchw: numpy float32 [1,3,608,608] in range [0,1]
        Returns: (list of float32 feature maps sorted largest→smallest, dt_ms)
        """
        # Convert NCHW → NHWC, scale to int8
        img = img_nchw[0] if img_nchw.ndim == 4 else img_nchw
        img_nhwc = img.transpose(1, 2, 0)   # [H,W,C]

        scl = float(2 ** self._in_fp)
        
        # Scale, round, clip, and cast to int8
        img_int8 = np.clip(np.round(img_nhwc * scl), -128, 127).astype(np.int8)
        
        # Allocate fresh C-contiguous buffers for this inference (matches dpu_debug.py)
        input_buf  = [np.empty(self._in_dims, dtype=np.int8, order='C')]
        output_buf = [np.empty(d, dtype=np.int8, order='C') for d in self._out_dims]
        
        # Copy data in
        input_buf[0][0, ...] = img_int8

        t0     = time.time()
        job_id = self._runner.execute_async(input_buf, output_buf)
        self._runner.wait(job_id)
        dt_ms  = (time.time() - t0) * 1000.0

        # Dequantize and convert NHWC → NCHW
        fms = []
        for od, fp in zip(output_buf, self._out_fps):
            scl_out = float(2 ** -fp)
            fm      = od.astype(np.float32) * scl_out
            if fm.ndim == 4 and fm.shape[-1] == 45:   # NHWC → NCHW
                fm = fm.transpose(0, 3, 1, 2)
            fms.append(fm)

        # Sort: 76² > 38² > 19² (largest feature map first)
        fms.sort(key=lambda x: x.shape[-1] * x.shape[-2], reverse=True)
        return fms, dt_ms

    def __del__(self):
        if hasattr(self, '_runner'):
            del self._runner


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(description='Complex-YOLOv4 on ZCU104 DPU')
    p.add_argument('--xmodel',       required=True)
    p.add_argument('--image',        default=None, help='BEV .npy file')
    p.add_argument('--image_dir',    default=None, help='Dir of .npy files')
    p.add_argument('--velodyne',     default=None, help='Velodyne .bin file')
    p.add_argument('--velodyne_dir', default=None, help='Dir of .bin files')
    p.add_argument('--conf', type=float, default=0.3)
    p.add_argument('--nms',  type=float, default=0.5)
    p.add_argument('--save_img', action='store_true')
    args = p.parse_args()

    # ── Collect inputs ──────────────────────────────────────────────
    inputs = []

    if args.image:
        img = np.load(args.image).astype(np.float32)
        if img.ndim == 4: img = img[0]
        inputs.append((os.path.basename(args.image), img))

    if args.image_dir:
        for f in sorted(os.listdir(args.image_dir)):
            if f.endswith('.npy'):
                img = np.load(os.path.join(args.image_dir, f)).astype(np.float32)
                if img.ndim == 4: img = img[0]
                inputs.append((f, img))

    if args.velodyne:
        fn  = makeBVFeature if _have_bev else make_bev_feature
        img = fn(args.velodyne, DISCRETIZATION, BOUNDARY) if _have_bev \
              else make_bev_feature(args.velodyne)
        inputs.append((os.path.basename(args.velodyne), img))

    if args.velodyne_dir:
        fn = makeBVFeature if _have_bev else make_bev_feature
        for f in sorted(os.listdir(args.velodyne_dir)):
            if f.endswith('.bin'):
                path = os.path.join(args.velodyne_dir, f)
                img  = fn(path, DISCRETIZATION, BOUNDARY) if _have_bev \
                       else make_bev_feature(path)
                inputs.append((f, img))

    if not inputs:
        sys.exit("ERROR: provide --image / --image_dir / --velodyne / --velodyne_dir")

    print("═" * 56)
    print("Complex-YOLOv4 Inference on ZCU104 (B4096 DPU)")
    print(f"  xmodel  : {args.xmodel}")
    print(f"  images  : {len(inputs)}")
    print(f"  conf_th : {args.conf}   nms_th : {args.nms}")
    print("═" * 56)

    # ── Load DPU once ───────────────────────────────────────────────
    dpu = DPURunner(args.xmodel)

    # ── Run inference ───────────────────────────────────────────────
    total_ms = 0.0
    for label, img in inputs:
        print(f"\n── {label} {'─'*max(0,40-len(label))}")
        img4 = np.expand_dims(img, 0)   # [1,3,608,608]
        print(f"  Input : {img4.shape} | range=[{img.min():.3f},{img.max():.3f}]")

        fms, dt = dpu.run(img4)
        total_ms += dt
        print(f"  DPU   : {dt:.1f} ms")
        for i, fm in enumerate(fms):
            print(f"  Head[{i}] shape={fm.shape}  range=[{fm.min():.3f},{fm.max():.3f}]")

        dets = postprocess(fms, args.conf, args.nms)

        if len(dets) == 0:
            print("  No detections above threshold.")
        else:
            print(f"\n  Detections ({len(dets)} found):")
            print(f"  {'─'*62}")
            print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  "
                  f"{'cx':>6} {'cy':>6} {'w':>5} {'h':>5}  {'Yaw':>8}")
            print(f"  {'─'*62}")
            for j, d in enumerate(dets):
                cls_id = int(d[7:].argmax())
                score  = d[6] * d[7+cls_id]
                yaw    = float(np.degrees(np.arctan2(d[4], d[5])))
                print(f"  {j+1:>3}  {CLASS_NAMES[cls_id]:<16} {score:>6.3f}  "
                      f"{d[0]:>6.1f} {d[1]:>6.1f} "
                      f"{d[2]:>5.1f} {d[3]:>5.1f}  {yaw:>+7.1f}°")

            print(f"\n  Real-world coordinates (metres):")
            for j, d in enumerate(dets):
                cls_id = int(d[7:].argmax())
                rx = (BEV_HEIGHT - d[1]) * DISCRETIZATION
                ry = (d[0] - BEV_WIDTH/2) * DISCRETIZATION
                rw = d[2] * DISCRETIZATION
                rh = d[3] * DISCRETIZATION
                yaw = float(np.degrees(np.arctan2(d[4], d[5])))
                print(f"   [{j+1}] {CLASS_NAMES[cls_id]}: "
                      f"X={rx:+.1f}m  Y={ry:+.1f}m  "
                      f"W={rw:.1f}m  L={rh:.1f}m  Yaw={yaw:+.1f}°")

        if args.save_img:
            stem     = os.path.splitext(label)[0]
            out_path = f"{stem}_det.png"
            visualize(img4, dets, out_path)

    n   = len(inputs)
    avg = total_ms / n if n else 0
    print(f"\n{'═'*56}")
    print(f"Processed {n} image(s)")
    if avg:
        print(f"Avg DPU  : {avg:.1f} ms/frame  →  {1000/avg:.1f} FPS")
    print(f"{'═'*56}")


if __name__ == '__main__':
    main()
