#!/usr/bin/env python3
"""
verify_ptq_v2.py
================
Verifies the PTQ model produced by YOUR existing quantize_for_dpu.py
(the one that used Darknet from darknet2pytorch.py + leakyrelu_epoch_30).

Works with the existing /workspace/quantize_result/ directory.
Does NOT edit any other files.

Usage:
    conda activate vitis-ai-pytorch
    python /workspace/verify_ptq_v2.py
    python /workspace/verify_ptq_v2.py --image /workspace/bev_images_test/bev_000001.npy

RESULT:
  [PASS] Both detect Cars  -> PTQ is accurate, board decode should work
  [FAIL] Float32=Car, PTQ=Pedestrian -> PTQ accuracy lost, fast_finetune needed
"""

import sys, os, glob, argparse, math, types
import numpy as np
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════
#  STEP 0: Stub broken imports BEFORE importing anything from /workspace/src
#  This allows 'from models.darknet2pytorch import Darknet' to work.
# ══════════════════════════════════════════════════════════════════════

# Stub utils.iou_rotated_boxes_utils (compiled .so that's missing some symbols)
_iou = types.ModuleType('utils.iou_rotated_boxes_utils')
for _s in ['iou_pred_vs_target_boxes', 'iou_rotated_boxes_targets_vs_anchors',
           'get_polygons_areas_fix_xy']:
    setattr(_iou, _s, lambda *a, **k: None)
sys.modules['utils.iou_rotated_boxes_utils'] = _iou

# Stub shapely (not installed in Docker by default)
_shp     = types.ModuleType('shapely')
_shp_geo = types.ModuleType('shapely.geometry')
class _FakePoly:
    def __init__(self, *a, **k): pass
    def buffer(self, *a): return self
    def intersection(self, *a): return self
    @property
    def area(self): return 0.0
_shp_geo.Polygon = _FakePoly
sys.modules['shapely']          = _shp
sys.modules['shapely.geometry'] = _shp_geo

# Now add workspace to path and import the original Darknet
sys.path.insert(0, '/workspace/src')
from models.darknet2pytorch import Darknet   # the real one from your codebase

# ══════════════════════════════════════════════════════════════════════
#  Paths — same as YOUR OLD quantize_for_dpu.py
# ══════════════════════════════════════════════════════════════════════
CFG_FILE   = '/workspace/src/config/cfg/complex_yolov4.cfg'
CHECKPOINT = '/workspace/checkpoints/Model_complexer_yolo_leakyrelu_epoch_30.pth'
OUTPUT_DIR = '/workspace/quantize_result'   # your existing calibration result


# ══════════════════════════════════════════════════════════════════════
#  Old-style DarknetBackbone — IDENTICAL to your old quantize_for_dpu.py
#  (node names MUST match for quant_info.json to load correctly)
# ══════════════════════════════════════════════════════════════════════
import torch.nn as nn

class DarknetBackbone(nn.Module):
    """Wraps the original Darknet and captures the 3 detection feature maps."""
    def __init__(self, cfg):
        super().__init__()
        self.net = Darknet(cfgfile=cfg, use_giou_loss=False)

    def forward(self, x):
        ind          = -2
        outputs      = {}
        yolo_outputs = []

        for block in self.net.blocks:
            ind += 1
            if block['type'] == 'net':
                continue
            elif block['type'] in ['convolutional', 'maxpool', 'reorg',
                                   'upsample', 'avgpool', 'softmax', 'connected']:
                x            = self.net.models[ind](x)
                outputs[ind] = x
            elif block['type'] == 'route':
                layers = block['layers'].split(',')
                layers = [int(i) if int(i) > 0 else int(i) + ind for i in layers]
                if len(layers) == 1:
                    if 'groups' not in block or int(block.get('groups', 1)) == 1:
                        x = outputs[layers[0]]
                    else:
                        g   = int(block['groups'])
                        gid = int(block['group_id'])
                        _, b, _, _ = outputs[layers[0]].shape
                        x = outputs[layers[0]][:, b // g * gid : b // g * (gid + 1)]
                elif len(layers) == 2:
                    x = torch.cat((outputs[layers[0]], outputs[layers[1]]), 1)
                elif len(layers) == 4:
                    x = torch.cat([outputs[layers[i]] for i in range(4)], 1)
                outputs[ind] = x
            elif block['type'] == 'shortcut':
                fl  = int(block['from'])
                fl  = fl if fl > 0 else fl + ind
                x   = outputs[fl] + outputs[ind - 1]
                act = block['activation']
                if act == 'leaky':
                    x = torch.nn.functional.leaky_relu(x, 0.1015625, inplace=True)
                elif act == 'relu':
                    x = torch.relu(x)
                outputs[ind] = x
            elif block['type'] == 'yolo':
                # Capture the conv output BEFORE the YOLO decode
                yolo_outputs.append(outputs[ind - 1])
                outputs[ind] = outputs[ind - 1]   # placeholder
            else:
                pass   # cost, etc.

        # Returns a tuple of 3 feature maps: (76x76, 38x38, 19x19) — order from cfg
        return tuple(yolo_outputs)


def load_weights(model, checkpoint_path):
    """Load checkpoint into model.net (handles DataParallel prefix)."""
    state = torch.load(checkpoint_path, map_location='cpu')
    # Handle DataParallel / training wrappers
    if isinstance(state, dict):
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.net.load_state_dict(state, strict=False)
    print(f"  Weights loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    return model


# ══════════════════════════════════════════════════════════════════════
#  Anchors (from complex_yolov4.cfg, parsed as darknet2pytorch.py does)
#  angle=0 for all anchors → sin(0)=0, cos(0)=1
# ══════════════════════════════════════════════════════════════════════
NUM_CLASSES = 8
CLASS_NAMES = ["Car", "Van", "Truck", "Pedestrian",
               "Person_sitting", "Cyclist", "Tram", "Misc"]

_anchor_raw = [11,15,0, 10,24,0, 11,25,0, 23,49,0, 23,55,0,
               24,53,0, 24,60,0, 27,63,0, 29,74,0]
ALL_ANCHORS = [
    (_anchor_raw[i], _anchor_raw[i+1],
     math.sin(math.radians(_anchor_raw[i+2])),
     math.cos(math.radians(_anchor_raw[i+2])))
    for i in range(0, len(_anchor_raw), 3)
]
HEAD_MASKS   = [[0,1,2], [3,4,5], [6,7,8]]
HEAD_STRIDES = [8, 16, 32]


# ══════════════════════════════════════════════════════════════════════
#  Decode — mirrors yolo_layer.py forward() exactly
#  im and re are RAW (not sigmoided)
# ══════════════════════════════════════════════════════════════════════
def decode_fmaps(fmaps):
    """
    fmaps: list/tuple of 3 tensors, sorted largest spatial first:
           [(1,45,76,76), (1,45,38,38), (1,45,19,19)]
    Returns: tensor [1, 22743, 15]
    """
    all_out = []
    for i, fm in enumerate(fmaps):
        stride = HEAD_STRIDES[i]
        anch   = [ALL_ANCHORS[j] for j in HEAD_MASKS[i]]
        B, C, H, W = fm.shape

        pred = fm.view(B, 3, 15, H, W).permute(0, 1, 3, 4, 2).contiguous()

        gx = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W, 1)
        gy = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1, 1)
        aw = torch.tensor([a[0] / stride for a in anch]).float().view(1, 3, 1, 1, 1)
        ah = torch.tensor([a[1] / stride for a in anch]).float().view(1, 3, 1, 1, 1)

        bx   = (torch.sigmoid(pred[..., 0:1]) + gx) * stride
        by   = (torch.sigmoid(pred[..., 1:2]) + gy) * stride
        bw   = torch.exp(pred[..., 2:3].clamp(-10, 10)) * aw * stride
        bh   = torch.exp(pred[..., 3:4].clamp(-10, 10)) * ah * stride
        im   = pred[..., 4:5]               # RAW — NOT sigmoided (per yolo_layer.py)
        re   = pred[..., 5:6]               # RAW — NOT sigmoided
        conf = torch.sigmoid(pred[..., 6:7])
        cls  = torch.sigmoid(pred[..., 7:])

        out = torch.cat([bx, by, bw, bh, im, re, conf, cls], dim=-1)
        all_out.append(out.view(B, -1, 15))

    return torch.cat(all_out, dim=1)   # [1, 22743, 15]


def simple_nms(preds, conf_thresh, nms_thresh):
    """Per-class axis-aligned NMS (no shapely needed on CPU)."""
    preds   = preds[preds[:, 6] >= conf_thresh]
    if len(preds) == 0:
        return []
    scores  = preds[:, 6] * preds[:, 7:].max(dim=1)[0]
    cls_ids = preds[:, 7:].argmax(dim=1)
    order   = scores.argsort(descending=True)
    preds, scores, cls_ids = preds[order], scores[order], cls_ids[order]

    used = torch.zeros(len(preds), dtype=torch.bool)
    keep = []
    for i in range(len(preds)):
        if used[i]: continue
        keep.append(i)
        cx, cy, w, h = preds[i, 0], preds[i, 1], preds[i, 2], preds[i, 3]
        for j in range(i + 1, len(preds)):
            if used[j] or cls_ids[j] != cls_ids[i]: continue
            cx2, cy2, w2, h2 = preds[j,0], preds[j,1], preds[j,2], preds[j,3]
            ix1 = max(cx - w/2, cx2 - w2/2); iy1 = max(cy - h/2, cy2 - h2/2)
            ix2 = min(cx + w/2, cx2 + w2/2); iy2 = min(cy + h/2, cy2 + h2/2)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter / (w*h + w2*h2 - inter + 1e-6) > nms_thresh:
                used[j] = True

    result = []
    for i in keep:
        d   = preds[i].numpy()
        yaw = float(np.degrees(np.arctan2(d[4], d[5])))
        result.append({
            'cls':   CLASS_NAMES[int(cls_ids[i])],
            'score': float(scores[i]),
            'cx': d[0], 'cy': d[1], 'w': d[2], 'h': d[3], 'yaw': yaw
        })
    return result


# ══════════════════════════════════════════════════════════════════════
#  Run one model and report detections
# ══════════════════════════════════════════════════════════════════════
def run_and_report(model, img_t, label, conf, nms):
    model.eval()
    with torch.no_grad():
        raw = model(img_t)   # tuple/list of 3 feature maps

    # Sort by spatial area (largest first) to align with HEAD_MASKS
    fmaps = sorted(list(raw), key=lambda x: x.shape[-1] * x.shape[-2], reverse=True)

    decoded = decode_fmaps(fmaps)
    dets    = simple_nms(decoded[0], conf, nms)

    print(f"\n  {'─'*56}")
    print(f"  {label}")
    print(f"  {'─'*56}")
    for i, fm in enumerate(fmaps):
        print(f"  Head[{i}]  shape={tuple(fm.shape)}  "
              f"range=[{fm.min().item():.3f}, {fm.max().item():.3f}]")

    if not dets:
        print(f"\n  [WARN] No detections above conf={conf}")
        print( "         Try --conf 0.3 to see more candidates.")
    else:
        print(f"\n  Detections ({len(dets)}):")
        print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  "
              f"{'cx':>6} {'cy':>6} {'w':>5} {'h':>5}  {'Yaw':>8}")
        print(f"  {'─'*58}")
        for j, d in enumerate(dets):
            print(f"  {j+1:>3}  {d['cls']:<16} {d['score']:>6.3f}  "
                  f"{d['cx']:>6.1f} {d['cy']:>6.1f} "
                  f"{d['w']:>5.1f} {d['h']:>5.1f}  {d['yaw']:>+7.1f} deg")
    return dets


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description='Verify old PTQ model accuracy')
    ap.add_argument('--image',    default=None, help='BEV .npy test image')
    ap.add_argument('--conf',     type=float, default=0.5)
    ap.add_argument('--nms',      type=float, default=0.5)
    ap.add_argument('--skip_fp32', action='store_true',
                    help='Skip float32 check, only run PTQ')
    args = ap.parse_args()

    # ── Find test image ─────────────────────────────────────────────
    img_path = args.image
    if not img_path:
        for pat in ['/workspace/bev_images_test/*.npy',
                    '/workspace/calibration_data/calib_0000.npy']:
            hits = sorted(glob.glob(pat))
            if hits: img_path = hits[0]; break
    if not img_path or not os.path.exists(img_path):
        sys.exit("ERROR: no test image. Use --image /path/to/bev.npy")

    img = np.load(img_path).astype(np.float32)
    if img.ndim == 4: img = img[0]
    img_t = torch.from_numpy(img).unsqueeze(0)   # [1,3,608,608]

    print("═" * 58)
    print("  Old PTQ Backbone Accuracy Verification")
    print(f"  Backbone   : Darknet (from darknet2pytorch.py)")
    print(f"  Checkpoint : {os.path.basename(CHECKPOINT)}")
    print(f"  Quant dir  : {OUTPUT_DIR}")
    print(f"  Image      : {os.path.basename(img_path)}")
    print(f"  Conf={args.conf}   NMS={args.nms}")
    print("═" * 58)

    # Check quant_info.json exists
    qi = os.path.join(OUTPUT_DIR, 'quant_info.json')
    if not os.path.exists(qi):
        print(f"\nERROR: {qi} not found.")
        print("The fast_finetune/calibration is still running.")
        print("Wait for quantize_for_dpu.py to finish, then re-run this script.")
        sys.exit(1)
    else:
        mtime = os.path.getmtime(qi)
        import datetime
        ts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"\n[OK] Found quant_info.json  (last modified: {ts})")

    # ── Load checkpoint ─────────────────────────────────────────────
    print(f"\n[INFO] Loading {CHECKPOINT} ...")

    # ── 1. Float32 model ────────────────────────────────────────────
    dets_fp32 = None
    if not args.skip_fp32:
        print("\n[1/2] Float32 model (the ground truth) ...")
        m_fp32 = DarknetBackbone(CFG_FILE)
        load_weights(m_fp32, CHECKPOINT)
        dets_fp32 = run_and_report(
            m_fp32, img_t,
            f"FLOAT32  [{os.path.basename(CHECKPOINT)}]",
            args.conf, args.nms)
    else:
        print("[1/2] Float32 skipped (--skip_fp32).")

    # ── 2. PTQ quantized model ───────────────────────────────────────
    print("\n[2/2] PTQ INT8 model (CPU simulation from quant_info.json) ...")
    dets_ptq = None
    try:
        from pytorch_nndct.apis import torch_quantizer

        m_ptq = DarknetBackbone(CFG_FILE)
        dummy = torch.randn(1, 3, 608, 608)

        # Reload the original weights BEFORE creating the quantized model
        load_weights(m_ptq, CHECKPOINT)

        # IMPORTANT: input_args must be a tuple (matches the old calibration call)
        quant = torch_quantizer('test', m_ptq, (dummy,), output_dir=OUTPUT_DIR)
        qm    = quant.quant_model

        dets_ptq = run_and_report(
            qm, img_t,
            "PTQ INT8  [from /workspace/quantize_result/quant_info.json]",
            args.conf, args.nms)

    except Exception as e:
        err = str(e)
        print(f"\n  ERROR: {err}")
        if 'mismatch' in err.lower():
            print()
            print("  NODE NAME MISMATCH — quant_info.json was built with a")
            print("  different backbone structure. Possible causes:")
            print("    1. The calibration used a different quantize_for_dpu.py.")
            print("    2. Wait for fast_finetune to finish and re-run.")
        elif 'not found' in err.lower() or 'FileNotFound' in err:
            print("  quant_info.json not found. Run quantize_for_dpu.py first.")

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"\n{'═' * 58}")
    print("  VERDICT")
    print(f"{'═' * 58}")

    if dets_fp32 is None and dets_ptq is None:
        print("  Both models failed — check errors above.")
    elif dets_ptq is None:
        print("  PTQ model failed to load — see error above.")
        print("  Float32 result above shows what the model SHOULD detect.")
    elif dets_fp32 is None:
        print("  Float32 skipped. PTQ result shown above.")
        if dets_ptq:
            print(f"  PTQ found: {sorted(set(d['cls'] for d in dets_ptq))}")
            print("  Run without --skip_fp32 to compare against float32.")
    else:
        fp32_cls = sorted(set(d['cls'] for d in dets_fp32))
        ptq_cls  = sorted(set(d['cls'] for d in dets_ptq))

        print(f"  Float32 detects : {fp32_cls}")
        print(f"  PTQ detects     : {ptq_cls}")
        print()

        if fp32_cls == ptq_cls:
            print("  [PASS] Classes MATCH!")
            print("  PTQ quantization is accurate.")
            print()
            print("  NEXT STEP: Deploy the compiled xmodel to the board.")
            print("  Use run_on_board.py (already saved alongside this file).")
        else:
            print("  [FAIL] CLASS MISMATCH — PTQ accuracy degraded.")
            print()
            if dets_ptq:
                # Check if PTQ head ranges are clipped (sign of bad fix_point)
                print("  Diagnosis: Check Head range values above.")
                print("    If any Head range shows -32.000 or +31.75,")
                print("    int8 values are clipping → fix_point too coarse.")
            print()
            print("  FIXES:")
            print("    1. Wait for fast_finetune to finish (already running!).")
            print("       After it completes, re-quantize will give better accuracy.")
            print("    2. After fast_finetune completes and quant_info.json updates,")
            print("       re-run this script to verify again.")
    print()


if __name__ == '__main__':
    main()
