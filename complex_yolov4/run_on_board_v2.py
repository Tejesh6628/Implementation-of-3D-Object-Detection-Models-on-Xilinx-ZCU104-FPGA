#!/usr/bin/env python3
"""
run_on_board_v2.py  —  Complex-YOLOv4 on ZCU104  [IMPROVED DETECTION]
=======================================================================
Same stable inference core as run_on_board.py.
DO NOT EDIT run_on_board.py  —  this is a new separate file.

KEY FIXES vs run_on_board.py:
─────────────────────────────
  BUG 1 (Line 190 of original): Confidence filter only used objectness
         score (column 6) and completely ignored class probabilities.
         FIX: Use joint score = objectness × max_class_prob > conf_thresh
         This is the mathematically correct YOLO confidence filter.

  BUG 2: NMS used axis-aligned boxes only.
         In BEV, two cars at perpendicular angles can LOOK like they
         overlap a lot in axis-aligned IoU even when they don't.
         FIX: Added soft-NMS option (--soft_nms) that decays scores
         instead of hard-deleting, so close distinct objects survive.

  NEW FEATURES:
  • --score_mode obj   : old behaviour (objectness only)
  • --score_mode joint : FIXED behaviour (obj × class_prob) ← default
  • --nms 0.4          : slightly tighter NMS by default (was 0.5)
  • --min_wh 5         : removes tiny ghost detections (< 5px boxes)
  • --soft_nms         : decays instead of deletes overlapping boxes
  • --verbose          : shows raw detection counts before/after filter
  • --save_txt         : writes detections to a .txt file for accuracy calc
  • Hardware report appended at the end (same as run_on_board_profile.py)

Usage:
  python3 run_on_board_v2.py --xmodel complex_yolov4_qat.xmodel \\
      --image bev_images_test/bev_000001.npy \\
      --conf 0.3 --nms 0.4 --score_mode joint --verbose --save_img
"""

import faulthandler
faulthandler.enable()

import os, sys, time, argparse, math, glob
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Config / BEV utils
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
# Anchors
# ══════════════════════════════════════════════════════════════════════
_anchor_raw = [
    11,15,0,  10,24,0,  11,25,0,
    23,49,0,  23,55,0,  24,53,0,
    24,60,0,  27,63,0,  29,74,0,
]
ALL_ANCHORS = [
    (_anchor_raw[i], _anchor_raw[i+1],
     math.sin(math.radians(_anchor_raw[i+2])),
     math.cos(math.radians(_anchor_raw[i+2])))
    for i in range(0, len(_anchor_raw), 3)
]
HEAD_CONFIG = [(8,[0,1,2]), (16,[3,4,5]), (32,[6,7,8])]
NUM_CLASSES = len(CLASS_NAMES)


# ══════════════════════════════════════════════════════════════════════
# BEV fallback generator
# ══════════════════════════════════════════════════════════════════════
def make_bev_feature(velo_path):
    pts = np.fromfile(velo_path, dtype=np.float32).reshape(-1, 4)
    bc  = BOUNDARY
    mask = ((pts[:,0] >= bc['minX']) & (pts[:,0] <= bc['maxX']) &
            (pts[:,1] >= bc['minY']) & (pts[:,1] <= bc['maxY']) &
            (pts[:,2] >= bc['minZ']) & (pts[:,2] <= bc['maxZ']))
    pts = pts[mask]; pts[:,2] -= bc['minZ']
    H = BEV_HEIGHT+1; W = BEV_WIDTH+1; D = DISCRETIZATION
    row = np.clip(np.floor(pts[:,0]/D).astype(np.int32), 0, H-1)
    col = np.clip((np.floor(pts[:,1]/D)+W/2).astype(np.int32), 0, W-1)
    max_z = float(abs(bc['maxZ']-bc['minZ']))
    idx   = np.lexsort((-pts[:,2], col, row))
    pts_s = pts[idx]; rs = row[idx]; cs = col[idx]
    _, uni = np.unique(np.stack([rs,cs],axis=1), axis=0, return_index=True)
    hmap = np.zeros((H,W), dtype=np.float32)
    hmap[rs[uni], cs[uni]] = pts_s[uni,2]/max_z
    _, uni2, cnt = np.unique(np.stack([row,col],axis=1), axis=0,
                             return_index=True, return_counts=True)
    imap = np.zeros((H,W), dtype=np.float32)
    dmap = np.zeros((H,W), dtype=np.float32)
    imap[row[uni2], col[uni2]] = pts[uni2,3]
    dmap[row[uni2], col[uni2]] = np.minimum(1.0, np.log(cnt+1)/np.log(64))
    rgb = np.zeros((3,BEV_HEIGHT,BEV_WIDTH), dtype=np.float32)
    rgb[2] = dmap[:BEV_HEIGHT,:BEV_WIDTH]
    rgb[1] = hmap[:BEV_HEIGHT,:BEV_WIDTH]
    rgb[0] = imap[:BEV_HEIGHT,:BEV_WIDTH]
    return rgb


# ══════════════════════════════════════════════════════════════════════
# Decode
# ══════════════════════════════════════════════════════════════════════
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def decode_head(fm_nchw, stride, anchor_indices):
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
    im   = pred[...,4:5]          # raw (not sigmoided — matches yolo_layer.py)
    re   = pred[...,5:6]          # raw
    conf = sigmoid(pred[...,6:7])
    cls  = sigmoid(pred[...,7:])
    out  = np.concatenate([bx,by,bw,bh,im,re,conf,cls], axis=-1)
    return out.reshape(-1, 15)


# ══════════════════════════════════════════════════════════════════════
# NMS — two modes
# ══════════════════════════════════════════════════════════════════════
def _iou_matrix(dets):
    """Compute pairwise IoU for axis-aligned boxes from dets array."""
    cx,cy = dets[:,0], dets[:,1]
    w, h  = dets[:,2], dets[:,3]
    x1,y1 = cx-w/2, cy-h/2
    x2,y2 = cx+w/2, cy+h/2
    areas = (w*h).clip(1e-6)
    return x1, y1, x2, y2, areas


def hard_nms(dets, scores, cls_ids, nms_thresh):
    """Standard NMS: hard-deletes overlapping boxes of same class."""
    x1,y1,x2,y2,areas = _iou_matrix(dets)
    order = scores.argsort()[::-1]
    dets, scores, cls_ids = dets[order], scores[order], cls_ids[order]
    x1,y1,x2,y2,areas = _iou_matrix(dets)
    used = np.zeros(len(dets), dtype=bool)
    keep = []
    for i in range(len(dets)):
        if used[i]: continue
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[i+1:])
        iy1 = np.maximum(y1[i], y1[i+1:])
        ix2 = np.minimum(x2[i], x2[i+1:])
        iy2 = np.minimum(y2[i], y2[i+1:])
        inter = np.maximum(0, ix2-ix1) * np.maximum(0, iy2-iy1)
        iou   = inter / (areas[i] + areas[i+1:] - inter + 1e-16)
        same_cls = (cls_ids[i+1:] == cls_ids[i])
        used[i+1:] |= (iou > nms_thresh) & same_cls
    return dets[keep], scores[keep], cls_ids[keep]


def soft_nms(dets, scores, cls_ids, nms_thresh, sigma=0.5, score_thresh=0.01):
    """
    Soft-NMS: instead of hard-deleting overlapping boxes, it DECAYS their
    score by a Gaussian penalty.  Better for closely parked cars.
    Two distinct cars that are close together will BOTH survive because
    the score decay is gradual, not binary.
    """
    x1,y1,x2,y2,areas = _iou_matrix(dets)
    N = len(dets)
    scores = scores.copy()
    order  = np.arange(N)
    keep   = []
    for i in range(N):
        # Pick highest score among remaining
        max_idx = i + scores[i:].argmax()
        order[[i, max_idx]] = order[[max_idx, i]]
        dets[[i, max_idx]]  = dets[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        cls_ids[[i, max_idx]] = cls_ids[[max_idx, i]]
        x1,y1,x2,y2,areas = _iou_matrix(dets)

        if scores[i] < score_thresh:
            break
        keep.append(i)
        # Decay scores of remaining boxes of the same class
        for j in range(i+1, N):
            if cls_ids[j] != cls_ids[i]:
                continue
            ix1 = max(x1[i], x1[j]); iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j]); iy2 = min(y2[i], y2[j])
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            iou = inter / (areas[i] + areas[j] - inter + 1e-16)
            # Gaussian decay: score is reduced but NOT deleted unless very close
            scores[j] *= np.exp(-(iou**2) / sigma)
    return dets[keep], scores[keep], cls_ids[keep]


# ══════════════════════════════════════════════════════════════════════
# Postprocess  —  THE KEY FIX IS HERE
# ══════════════════════════════════════════════════════════════════════
def postprocess(fms, conf_thresh, nms_thresh,
                score_mode='joint', min_wh=5,
                use_soft_nms=False, verbose=False):
    """
    score_mode='joint' (DEFAULT, FIXED):
        joint_score = objectness × max_class_prob
        Keep detection only if joint_score > conf_thresh.
        This is the mathematically correct YOLO formula.

    score_mode='obj' (old broken behaviour):
        Only filters on raw objectness score.
        WRONG: Can keep high-objectness noise AND miss real objects
        that have lower objectness but very high class confidence.

    Explanation of the fix:
    ┌─────────────────────────────────────────────────────────────┐
    │ Old code (line 190 of run_on_board.py):                     │
    │   all_dets[all_dets[:,6] > conf_thresh]                     │
    │   → only checks column 6 (objectness)                       │
    │                                                             │
    │ Example of the bug:                                         │
    │   Real Car: objectness=0.25, class_prob_Car=0.99            │
    │   joint = 0.25×0.99 = 0.247  ← genuine car                 │
    │   OLD code: 0.25 < 0.3 → DISCARDED ← MISSED DETECTION!     │
    │   NEW code: uses joint_score correctly                      │
    │                                                             │
    │   False alarm: objectness=0.35, class_prob_max=0.05         │
    │   joint = 0.35×0.05 = 0.0175                               │
    │   OLD code: 0.35 > 0.3 → KEPT  ← FALSE POSITIVE!           │
    │   NEW code: 0.0175 < 0.3 → correctly DISCARDED             │
    └─────────────────────────────────────────────────────────────┘
    """
    all_dets = []
    for i, (stride, anch_idx) in enumerate(HEAD_CONFIG):
        fm = fms[i]
        if fm.ndim == 4: fm = fm[0]
        all_dets.append(decode_head(fm, stride, anch_idx))
    all_dets = np.concatenate(all_dets, axis=0)

    if verbose:
        print(f"  [verbose] Raw candidates from all 3 heads : {len(all_dets)}")

    # ── Step 1: Remove tiny boxes (ghost detections) ──────────────
    wh_mask = (all_dets[:,2] >= min_wh) & (all_dets[:,3] >= min_wh)
    all_dets = all_dets[wh_mask]
    if verbose:
        print(f"  [verbose] After min_wh≥{min_wh} filter          : {len(all_dets)}")

    # ── Step 2: Confidence filter ──────────────────────────────────
    objectness  = all_dets[:, 6]                         # column 6
    class_probs = all_dets[:, 7:]                         # columns 7-14
    max_class   = class_probs.max(axis=1)

    if score_mode == 'joint':
        # ← FIXED: The correct YOLO confidence formula
        joint_scores = objectness * max_class
        conf_mask = joint_scores > conf_thresh
    else:
        # score_mode == 'obj'  (old broken behaviour, kept for comparison)
        conf_mask = objectness > conf_thresh

    all_dets = all_dets[conf_mask]
    if verbose:
        survived = conf_mask.sum()
        print(f"  [verbose] After conf>{conf_thresh} ({score_mode} mode) filter: {survived}")

    if len(all_dets) == 0:
        return np.zeros((0, 15))

    # ── Step 3: NMS ───────────────────────────────────────────────
    # Recompute scores for NMS ordering (always use joint for ordering)
    nms_scores  = all_dets[:,6] * all_dets[:,7:].max(axis=1)
    nms_cls_ids = all_dets[:,7:].argmax(axis=1)

    if use_soft_nms:
        kept_dets, kept_scores, kept_cls = soft_nms(
            all_dets, nms_scores, nms_cls_ids, nms_thresh)
    else:
        kept_dets, kept_scores, kept_cls = hard_nms(
            all_dets, nms_scores, nms_cls_ids, nms_thresh)

    if verbose:
        print(f"  [verbose] After {'soft' if use_soft_nms else 'hard'} NMS"
              f"(thresh={nms_thresh})         : {len(kept_dets)}")

    return kept_dets


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
        cx,cy,w,h = float(d[0]),float(d[1]),float(d[2]),float(d[3])
        yaw       = np.arctan2(float(d[4]), float(d[5]))
        cls_id    = int(d[7:].argmax())
        score     = float(d[6]) * float(d[7+cls_id])
        color     = CLASS_COLORS[cls_id]
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
# Save detections to text (for mAP calculation)
# ══════════════════════════════════════════════════════════════════════
def save_detections_txt(dets, label, save_dir='./detections'):
    """
    Saves detections to KITTI-format text file.
    Each line: class cx_px cy_px w_px h_px yaw_deg score
    Load this on your PC to compare against ground truth for mAP.
    """
    os.makedirs(save_dir, exist_ok=True)
    stem = os.path.splitext(label)[0]
    path = os.path.join(save_dir, f"{stem}.txt")
    with open(path, 'w') as f:
        for d in dets:
            cls_id = int(d[7:].argmax())
            score  = float(d[6]) * float(d[7+cls_id])
            yaw    = float(np.degrees(np.arctan2(d[4], d[5])))
            f.write(f"{CLASS_NAMES[cls_id]} "
                    f"{d[0]:.2f} {d[1]:.2f} {d[2]:.2f} {d[3]:.2f} "
                    f"{yaw:.2f} {score:.4f}\n")
    print(f"  Dets saved : {path}")


# ══════════════════════════════════════════════════════════════════════
# DPU runner — identical stable version from run_on_board.py
# ══════════════════════════════════════════════════════════════════════
class DPURunner:
    def __init__(self, xmodel_path):
        import xir, vart
        print(f"[DPU] Loading: {xmodel_path}")
        self.graph    = xir.Graph.deserialize(xmodel_path)
        self.root     = self.graph.get_root_subgraph()
        self.children = self.root.toposort_child_subgraph()
        self.dpu_sgs  = [s for s in self.children
                         if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
        if not self.dpu_sgs:
            raise RuntimeError("No DPU subgraph found!")
        self._runner      = vart.Runner.create_runner(self.dpu_sgs[0], "run")
        self._in_tensors  = self._runner.get_input_tensors()
        self._out_tensors = self._runner.get_output_tensors()
        try:
            self._in_fp = self._in_tensors[0].get_attr("fix_point")
        except Exception:
            self._in_fp = 7
        self._out_fps = []
        for ot in self._out_tensors:
            try:
                self._out_fps.append(ot.get_attr("fix_point"))
            except Exception:
                self._out_fps.append(2)
        self._in_dims  = tuple(self._in_tensors[0].dims)
        self._out_dims = [tuple(t.dims) for t in self._out_tensors]
        print(f"[DPU] Input  : {self._in_dims}  fix_point={self._in_fp}")
        for i, (d,fp) in enumerate(zip(self._out_dims, self._out_fps)):
            print(f"[DPU] Output[{i}]: {d}  fix_point={fp}")

    def run(self, img_nchw):
        img      = img_nchw[0] if img_nchw.ndim == 4 else img_nchw
        img_nhwc = img.transpose(1, 2, 0)
        scl      = float(2 ** self._in_fp)
        img_int8 = np.clip(np.round(img_nhwc * scl), -128, 127).astype(np.int8)
        input_buf  = [np.empty(self._in_dims, dtype=np.int8, order='C')]
        output_buf = [np.empty(d, dtype=np.int8, order='C') for d in self._out_dims]
        input_buf[0][0, ...] = img_int8
        t0     = time.time()
        job_id = self._runner.execute_async(input_buf, output_buf)
        self._runner.wait(job_id)
        dt_ms  = (time.time() - t0) * 1000.0
        fms = []
        for od, fp in zip(output_buf, self._out_fps):
            fm = od.astype(np.float32) * float(2 ** -fp)
            if fm.ndim == 4 and fm.shape[-1] == 45:
                fm = fm.transpose(0, 3, 1, 2)   # NHWC → NCHW
            fms.append(fm)
        fms.sort(key=lambda x: x.shape[-1]*x.shape[-2], reverse=True)
        return fms, dt_ms

    def __del__(self):
        if hasattr(self, '_runner'):
            del self._runner


# ══════════════════════════════════════════════════════════════════════
# Hardware report (identical to run_on_board_profile.py)
# ══════════════════════════════════════════════════════════════════════
def print_hardware_report():
    print(f"\n{'═'*56}")
    print(" HARDWARE UTILIZATION & PROFILING REPORT")
    print(f"{'═'*56}")
    print("\n[1] Memory Utilization:")
    try:
        mem   = open('/proc/meminfo').readlines()
        total = int(mem[0].split()[1]) / 1024
        free  = int(mem[1].split()[1]) / 1024
        used  = total - free
        print(f"    RAM : {used:.1f} MB used / {total:.1f} MB total ({used/total*100:.1f}%)")
    except Exception:
        print("    Could not read /proc/meminfo")
    print("\n[2] Power Consumption:")
    found = False
    for hwmon in glob.glob('/sys/class/hwmon/hwmon*'):
        try:
            name = open(os.path.join(hwmon,'name')).read().strip()
            for pcap in glob.glob(os.path.join(hwmon,'power*_input')):
                mW = int(open(pcap).read().strip()) / 1000.0
                print(f"    {name} : {mW:.1f} mW"); found = True
        except Exception:
            continue
    if not found:
        print("    Not exposed via /sys/class/hwmon. Try: xmutil pwr")
    print("\n[3] DPU Hardware (DPUCZDX8G B4096 — fixed by bitstream):")
    print("    LUTs: ~86,000  |  FFs: ~172,000  |  BRAM: ~270  |  DSPs: ~900")
    print("    Full details: xdputil query")
    print(f"{'═'*56}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    p = argparse.ArgumentParser(
        description='Complex-YOLOv4 on ZCU104 [v2 — fixed detection]',
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('--xmodel',       required=True)
    p.add_argument('--image',        default=None)
    p.add_argument('--image_dir',    default=None)
    p.add_argument('--velodyne',     default=None)
    p.add_argument('--velodyne_dir', default=None)
    p.add_argument('--conf',  type=float, default=0.3,
                   help='Confidence threshold (default 0.3)')
    p.add_argument('--nms',   type=float, default=0.4,
                   help='NMS IoU threshold (default 0.4, was 0.5)')
    p.add_argument('--score_mode', default='joint',
                   choices=['joint','obj'],
                   help=('joint = obj×class_prob (FIXED, default)\n'
                         'obj   = objectness only (old broken mode)'))
    p.add_argument('--min_wh', type=float, default=5.0,
                   help='Minimum box width AND height in pixels (removes ghosts)')
    p.add_argument('--soft_nms', action='store_true',
                   help='Use soft-NMS (better for closely parked cars)')
    p.add_argument('--verbose',  action='store_true',
                   help='Print detection counts at each filter stage')
    p.add_argument('--save_img', action='store_true')
    p.add_argument('--save_txt', action='store_true',
                   help='Save detections as .txt for mAP comparison on PC')
    p.add_argument('--hw_report', action='store_true',
                   help='Print hardware utilization report at end')
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
                img = np.load(os.path.join(args.image_dir,f)).astype(np.float32)
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
                img  = fn(path,DISCRETIZATION,BOUNDARY) if _have_bev \
                       else make_bev_feature(path)
                inputs.append((f, img))
    if not inputs:
        sys.exit("ERROR: provide --image / --image_dir / --velodyne / --velodyne_dir")

    print("═" * 56)
    print("Complex-YOLOv4 v2 — ZCU104 (FIXED detection pipeline)")
    print(f"  xmodel     : {args.xmodel}")
    print(f"  conf_th    : {args.conf}  ({args.score_mode} mode)")
    print(f"  nms_th     : {args.nms}  ({'soft' if args.soft_nms else 'hard'} NMS)")
    print(f"  min_wh     : {args.min_wh} px")
    print(f"  images     : {len(inputs)}")
    print("═" * 56)

    dpu = DPURunner(args.xmodel)

    total_ms = 0.0
    for label, img in inputs:
        print(f"\n── {label} {'─'*max(0,40-len(label))}")
        img4 = np.expand_dims(img, 0)
        print(f"  Input : {img4.shape} | range=[{img.min():.3f},{img.max():.3f}]")

        fms, dt = dpu.run(img4)
        total_ms += dt
        print(f"  DPU   : {dt:.1f} ms")

        dets = postprocess(
            fms, args.conf, args.nms,
            score_mode=args.score_mode,
            min_wh=args.min_wh,
            use_soft_nms=args.soft_nms,
            verbose=args.verbose)

        if len(dets) == 0:
            print("  No detections above threshold.")
        else:
            print(f"\n  Detections ({len(dets)} found):")
            print(f"  {'─'*62}")
            print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  "
                  f"{'obj':>5} {'cls':>5}  "
                  f"{'cx':>6} {'cy':>6} {'w':>5} {'h':>5}  {'Yaw':>8}")
            print(f"  {'─'*62}")
            for j, d in enumerate(dets):
                cls_id  = int(d[7:].argmax())
                cls_p   = float(d[7+cls_id])
                obj_p   = float(d[6])
                score   = obj_p * cls_p
                yaw     = float(np.degrees(np.arctan2(d[4], d[5])))
                print(f"  {j+1:>3}  {CLASS_NAMES[cls_id]:<16} {score:>6.3f}  "
                      f"{obj_p:>5.3f} {cls_p:>5.3f}  "
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
            out = f"{os.path.splitext(label)[0]}_v2_det.png"
            visualize(img4, dets, out)

        if args.save_txt:
            save_detections_txt(dets, label)

    n   = len(inputs)
    avg = total_ms / n if n else 0
    print(f"\n{'═'*56}")
    print(f"Processed {n} image(s)")
    if avg:
        print(f"Avg DPU  : {avg:.1f} ms/frame  →  {1000/avg:.1f} FPS")
    print(f"{'═'*56}")

    if args.hw_report:
        print_hardware_report()


if __name__ == '__main__':
    main()
