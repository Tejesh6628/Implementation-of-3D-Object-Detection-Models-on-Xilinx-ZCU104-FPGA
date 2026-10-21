#!/usr/bin/env python3
"""
verify_quantized.py  —  Fully self-contained PTQ accuracy check
================================================================
Run inside Vitis AI Docker (vitis-ai-pytorch) to verify whether
the PTQ-quantized model gives the same detections as float32.

Usage:
    conda activate vitis-ai-pytorch
    python /workspace/verify_quantized.py
    python /workspace/verify_quantized.py --image /workspace/bev_images_test/bev_000001.npy --conf 0.5

RESULT INTERPRETATION:
  ✅  Float32 and PTQ detect SAME classes
      → Quantization is fine. Board decode or preprocessing differs.

  ❌  Float32=Car, PTQ=Pedestrian  (or any class mismatch)
      → PTQ calibration degraded accuracy.
      → Fix: run fast_finetune (already added to quantize_for_dpu.py).
"""

import sys, os, glob, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════
# Paths — adjust if your Docker mount differs
# ══════════════════════════════════════════════════════════════════════
CFG_FILE   = '/workspace/src/config/cfg/complex_yolov4.cfg'
CHECKPOINT = '/workspace/checkpoints/Model_complexer_yolo_dpu_slope_epoch_40.pth'
OUTPUT_DIR = '/workspace/quantize_result'   # where quant_info.json lives

# ══════════════════════════════════════════════════════════════════════
# Embedded CFG parser  (no external deps)
# ══════════════════════════════════════════════════════════════════════
def parse_cfg(cfgfile):
    blocks = []
    with open(cfgfile) as fp:
        block = None
        for raw in fp:
            line = raw.rstrip()
            if not line or line[0] == '#':
                continue
            if line[0] == '[':
                if block:
                    blocks.append(block)
                block = {'type': line.strip('[]')}
                if block['type'] == 'convolutional':
                    block['batch_normalize'] = 0
            else:
                k, v = line.split('=', 1)
                k = k.strip()
                if k == 'type':
                    k = '_type'
                block[k] = v.strip()
        if block:
            blocks.append(block)
    return blocks


# ══════════════════════════════════════════════════════════════════════
# DPU-friendly backbone  (identical to quantize_for_dpu.py)
# Mish→LeakyReLU, upsample_expand→nn.Upsample, YOLO→Identity
# ══════════════════════════════════════════════════════════════════════
class MaxPoolDark(nn.Module):
    def __init__(self, size=2, stride=1):
        super().__init__()
        self.size = size
        self.stride = stride

    def forward(self, x):
        p = self.size // 2
        if ((x.shape[2]-1)//self.stride) != ((x.shape[2]+2*p-self.size)//self.stride):
            p1, p2 = (self.size-1)//2, (self.size-1)//2+1
        else:
            p1 = p2 = (self.size-1)//2
        if ((x.shape[3]-1)//self.stride) != ((x.shape[3]+2*p-self.size)//self.stride):
            p3, p4 = (self.size-1)//2, (self.size-1)//2+1
        else:
            p3 = p4 = (self.size-1)//2
        return F.max_pool2d(F.pad(x, (p3,p4,p1,p2), mode='replicate'),
                            self.size, stride=self.stride)


class EmptyModule(nn.Module):
    def forward(self, x): return x


class DarknetBackbone(nn.Module):
    def __init__(self, cfgfile):
        super().__init__()
        self.blocks = parse_cfg(cfgfile)
        self.models  = self._build()

    def _build(self):
        models = nn.ModuleList()
        pf, of = 3, []
        ps, os_ = 1, []
        conv_id = 0
        for b in self.blocks:
            t = b['type']
            if t == 'net':
                pf = int(b['channels']); continue
            if t == 'convolutional':
                conv_id += 1
                bn   = int(b['batch_normalize'])
                filt = int(b['filters'])
                k    = int(b['size'])
                s    = int(b['stride'])
                pad  = (k-1)//2 if int(b['pad']) else 0
                act  = b['activation']
                m = nn.Sequential()
                if bn:
                    m.add_module(f'conv{conv_id}',
                        nn.Conv2d(pf, filt, k, s, pad, bias=False))
                    m.add_module(f'bn{conv_id}', nn.BatchNorm2d(filt))
                else:
                    m.add_module(f'conv{conv_id}',
                        nn.Conv2d(pf, filt, k, s, pad))
                if act in ('leaky', 'mish'):
                    m.add_module(f'lrelu{conv_id}', nn.LeakyReLU(0.1, inplace=True))
                elif act == 'relu':
                    m.add_module(f'relu{conv_id}', nn.ReLU(inplace=True))
                pf = filt; of.append(pf); ps *= s; os_.append(ps)
                models.append(m)
            elif t == 'maxpool':
                sz = int(b['size']); s = int(b['stride'])
                if s == 1 and sz % 2:
                    m = nn.MaxPool2d(sz, s, padding=sz//2)
                elif s == sz:
                    m = nn.MaxPool2d(sz, s, padding=0)
                else:
                    m = MaxPoolDark(sz, s)
                of.append(pf); ps *= s; os_.append(ps); models.append(m)
            elif t == 'upsample':
                s = int(b['stride'])
                m = nn.Upsample(scale_factor=s, mode='nearest')
                of.append(pf); ps //= s; os_.append(ps); models.append(m)
            elif t == 'route':
                lyrs = b['layers'].split(',')
                idx  = len(models)
                li   = [int(i) if int(i)>0 else int(i)+idx for i in lyrs]
                if len(li) == 1:
                    if 'groups' not in b or int(b.get('groups',1)) == 1:
                        pf = of[li[0]]; ps = os_[li[0]]
                    else:
                        pf = of[li[0]] // int(b['groups']); ps = os_[li[0]]
                elif len(li) == 2:
                    pf = of[li[0]] + of[li[1]]; ps = os_[li[0]]
                elif len(li) == 4:
                    pf = sum(of[l] for l in li); ps = os_[li[0]]
                of.append(pf); os_.append(ps); models.append(EmptyModule())
            elif t == 'shortcut':
                idx = len(models)
                pf = of[idx-1]; ps = os_[idx-1]
                of.append(pf); os_.append(ps); models.append(EmptyModule())
            elif t == 'yolo':
                of.append(pf); os_.append(ps); models.append(nn.Identity())
            else:
                pass   # unknown block — skip
        return models

    def forward(self, x):
        ind = -2
        outputs = {}
        fmaps = []
        for b in self.blocks:
            ind += 1
            t = b['type']
            if t == 'net':
                continue
            elif t in ('convolutional','maxpool','upsample','avgpool','softmax','connected'):
                x = self.models[ind](x); outputs[ind] = x
            elif t == 'route':
                lyrs = b['layers'].split(',')
                li   = [int(i) if int(i)>0 else int(i)+ind for i in lyrs]
                if len(li) == 1:
                    if 'groups' not in b or int(b.get('groups',1)) == 1:
                        x = outputs[li[0]]
                    else:
                        g = int(b['groups']); gid = int(b['group_id'])
                        _, ch, _, _ = outputs[li[0]].shape
                        x = outputs[li[0]][:, ch//g*gid : ch//g*(gid+1)]
                elif len(li) == 2:
                    x = torch.cat((outputs[li[0]], outputs[li[1]]), 1)
                elif len(li) == 4:
                    x = torch.cat([outputs[l] for l in li], 1)
                outputs[ind] = x
            elif t == 'shortcut':
                fl = int(b['from'])
                fl = fl if fl > 0 else fl + ind
                x  = outputs[fl] + outputs[ind-1]
                a  = b['activation']
                if a == 'leaky':   x = F.leaky_relu(x, 0.1, inplace=True)
                elif a == 'relu':  x = F.relu(x, inplace=True)
                outputs[ind] = x
            elif t == 'yolo':
                fmaps.append(x)
            else:
                continue
        # Return 3 separate feature maps (NOT flattened) for easy decode
        return fmaps   # list of [1,45,76,76], [1,45,38,38], [1,45,19,19]


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════
NUM_CLASSES = 8
CLASS_NAMES = ["Car","Van","Truck","Pedestrian",
               "Person_sitting","Cyclist","Tram","Misc"]

# Anchors from cfg: (w, h, sin(angle), cos(angle))  — all angles=0
_r = [11,15,0, 10,24,0, 11,25,0, 23,49,0, 23,55,0,
      24,53,0, 24,60,0, 27,63,0, 29,74,0]
ALL_ANCHORS = [(_r[i], _r[i+1],
                math.sin(math.radians(_r[i+2])),
                math.cos(math.radians(_r[i+2])))
               for i in range(0, len(_r), 3)]

HEAD_MASKS   = [[0,1,2], [3,4,5], [6,7,8]]
HEAD_STRIDES = [8, 16, 32]


# ══════════════════════════════════════════════════════════════════════
# Decode  (mirrors yolo_layer.py forward exactly)
# im, re are RAW (not sigmoided) — same as original training code
# ══════════════════════════════════════════════════════════════════════
def decode_fmaps(fmaps):
    """fmaps: list of 3 tensors [1,45,H,W]. Returns [1,22743,15]."""
    all_out = []
    for i, fm in enumerate(fmaps):
        stride = HEAD_STRIDES[i]
        anch   = [ALL_ANCHORS[j] for j in HEAD_MASKS[i]]
        B, C, H, W = fm.shape

        pred = fm.view(B, 3, 15, H, W).permute(0,1,3,4,2).contiguous()

        gx = torch.arange(W, dtype=torch.float32).view(1,1,1,W,1)
        gy = torch.arange(H, dtype=torch.float32).view(1,1,H,1,1)
        aw = torch.tensor([a[0]/stride for a in anch]).float().view(1,3,1,1,1)
        ah = torch.tensor([a[1]/stride for a in anch]).float().view(1,3,1,1,1)

        bx   = (torch.sigmoid(pred[...,0:1]) + gx) * stride
        by   = (torch.sigmoid(pred[...,1:2]) + gy) * stride
        bw   = torch.exp(pred[...,2:3].clamp(-10,10)) * aw * stride
        bh   = torch.exp(pred[...,3:4].clamp(-10,10)) * ah * stride
        im   = pred[...,4:5]                 # raw — NOT sigmoided
        re   = pred[...,5:6]                 # raw — NOT sigmoided
        conf = torch.sigmoid(pred[...,6:7])
        cls  = torch.sigmoid(pred[...,7:])

        out = torch.cat([bx,by,bw,bh,im,re,conf,cls], dim=-1)
        all_out.append(out.view(B,-1,15))

    return torch.cat(all_out, dim=1)   # [1, 22743, 15]


def simple_nms(preds, conf_thresh, nms_thresh):
    """Quick per-class axis-aligned NMS (no shapely needed)."""
    preds = preds[preds[:,6] >= conf_thresh]
    if len(preds) == 0:
        return []
    scores  = preds[:,6] * preds[:,7:].max(dim=1)[0]
    cls_ids = preds[:,7:].argmax(dim=1)
    order   = scores.argsort(descending=True)
    preds, scores, cls_ids = preds[order], scores[order], cls_ids[order]

    used = torch.zeros(len(preds), dtype=torch.bool)
    keep = []
    for i in range(len(preds)):
        if used[i]: continue
        keep.append(i)
        cx,cy,w,h = preds[i,0],preds[i,1],preds[i,2],preds[i,3]
        for j in range(i+1, len(preds)):
            if used[j] or cls_ids[j] != cls_ids[i]: continue
            cx2,cy2,w2,h2 = preds[j,0],preds[j,1],preds[j,2],preds[j,3]
            ix1 = max(cx-w/2, cx2-w2/2); iy1 = max(cy-h/2, cy2-h2/2)
            ix2 = min(cx+w/2, cx2+w2/2); iy2 = min(cy+h/2, cy2+h2/2)
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            union = w*h + w2*h2 - inter + 1e-6
            if inter/union > nms_thresh:
                used[j] = True

    result = []
    for i in keep:
        d   = preds[i].numpy()
        s   = float(scores[i])
        c   = int(cls_ids[i])
        yaw = float(np.degrees(np.arctan2(d[4], d[5])))
        result.append({'cls':CLASS_NAMES[c],'score':s,
                       'cx':d[0],'cy':d[1],'w':d[2],'h':d[3],'yaw':yaw})
    return result


# ══════════════════════════════════════════════════════════════════════
# Report helper
# ══════════════════════════════════════════════════════════════════════
def run_and_report(model, img_t, label, conf, nms):
    model.eval()
    with torch.no_grad():
        fmaps = model(img_t)

    # fmaps is a list of 3 tensors from DarknetBackbone.forward()
    # Sort largest spatial first to match HEAD_MASKS order
    fmaps.sort(key=lambda x: x.shape[-1]*x.shape[-2], reverse=True)

    decoded = decode_fmaps(fmaps)
    dets    = simple_nms(decoded[0], conf, nms)

    print(f"\n  {'─'*54}")
    print(f"  {label}")
    print(f"  {'─'*54}")
    for i, fm in enumerate(fmaps):
        print(f"  Head[{i}] shape={tuple(fm.shape)}  "
              f"range=[{fm.min().item():.3f}, {fm.max().item():.3f}]")

    if not dets:
        print(f"  ⚠  No detections above conf={conf}")
    else:
        print(f"  Detections ({len(dets)}):")
        print(f"  {'#':>3}  {'Class':<16} {'Score':>6}  "
              f"{'cx':>6} {'cy':>6} {'w':>5} {'h':>5}  {'Yaw':>8}")
        print(f"  {'─'*58}")
        for j, d in enumerate(dets):
            print(f"  {j+1:>3}  {d['cls']:<16} {d['score']:>6.3f}  "
                  f"{d['cx']:>6.1f} {d['cy']:>6.1f} "
                  f"{d['w']:>5.1f} {d['h']:>5.1f}  {d['yaw']:>+7.1f}°")
    return dets


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', default=None, help='BEV .npy test image')
    ap.add_argument('--conf',  type=float, default=0.5)
    ap.add_argument('--nms',   type=float, default=0.5)
    ap.add_argument('--skip_fp32', action='store_true',
                    help='Skip float32 (saves ~30s, just run PTQ check)')
    args = ap.parse_args()

    # ── Find test image ─────────────────────────────────────────────
    img_path = args.image
    if not img_path:
        for pat in ['/workspace/bev_images_test/*.npy',
                    '/workspace/calibration_data/calib_0000.npy']:
            hits = sorted(glob.glob(pat))
            if hits: img_path = hits[0]; break
    if not img_path or not os.path.exists(img_path):
        sys.exit("ERROR: no test image found. Use --image /path/to/bev.npy")

    img = np.load(img_path).astype(np.float32)
    if img.ndim == 4: img = img[0]       # [1,3,H,W] → [3,H,W]
    img_t = torch.from_numpy(img).unsqueeze(0)   # [1,3,608,608]

    print("═"*56)
    print("  Complex-YOLOv4 PTQ Accuracy Verification")
    print(f"  Image : {os.path.basename(img_path)}")
    print(f"  Shape : {list(img_t.shape)}  Range: [{img.min():.3f},{img.max():.3f}]")
    print(f"  Conf  : {args.conf}   NMS : {args.nms}")
    print("═"*56)

    # ── Load checkpoint ─────────────────────────────────────────────
    print(f"\n[INFO] Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location='cpu')
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        ckpt = ckpt['model_state_dict']

    # ── 1. Float32 model ────────────────────────────────────────────
    dets_fp32 = None
    if not args.skip_fp32:
        print("\n[1/2] Float32 model ...")
        m_fp32 = DarknetBackbone(CFG_FILE)
        m_fp32.load_state_dict(ckpt, strict=False)
        dets_fp32 = run_and_report(
            m_fp32, img_t,
            f"FLOAT32  [{os.path.basename(CHECKPOINT)}]",
            args.conf, args.nms)
    else:
        print("[1/2] Float32 skipped (--skip_fp32).")

    # ── 2. PTQ quantized model (test mode reads quant_info.json) ────
    dets_ptq = None
    print("\n[2/2] PTQ int8 model (simulated on CPU) ...")
    try:
        from pytorch_nndct.apis import torch_quantizer

        m_ptq  = DarknetBackbone(CFG_FILE)
        dummy  = torch.randn(1, 3, 608, 608)
        quant  = torch_quantizer('test', m_ptq, dummy, output_dir=OUTPUT_DIR)
        qm     = quant.quant_model
        qm.load_state_dict(ckpt, strict=False)
        dets_ptq = run_and_report(
            qm, img_t,
            "PTQ INT8  [quantize_result/quant_info.json]",
            args.conf, args.nms)

    except FileNotFoundError:
        print("  ERROR: quantize_result/quant_info.json not found.")
        print("         Run 'python quantize_for_dpu.py' first to generate it.")
    except Exception as e:
        print(f"  ERROR: {e}")

    # ── Verdict ─────────────────────────────────────────────────────
    print(f"\n{'═'*56}")
    print("  VERDICT")
    print(f"{'═'*56}")
    if dets_fp32 is None or dets_ptq is None:
        print("  One model failed — check errors above.")
    else:
        fp32_cls = sorted(set(d['cls'] for d in dets_fp32))
        ptq_cls  = sorted(set(d['cls'] for d in dets_ptq))
        if fp32_cls == ptq_cls:
            print(f"  ✅  MATCH — both detect: {fp32_cls}")
            print("      Quantization is accurate.")
            print("      If board gives wrong results → board preprocessing differs.")
        else:
            print(f"  ❌  MISMATCH!")
            print(f"      Float32 detects : {fp32_cls}")
            print(f"      PTQ detects     : {ptq_cls}")
            print()
            print("  Root cause: PTQ calibration degraded accuracy (common with")
            print("  complex YOLO + only 100-200 calibration images).")
            print()
            print("  NEXT STEPS (in order):")
            print("    1. Ensure quantize_for_dpu.py has FAST_FINETUNE = True")
            print("       and re-run:  python quantize_for_dpu.py")
            print("    2. Use GPU Docker image for 10x faster fast_finetune.")
            print("    3. Or increase calibration to 1000+ images.")
    print()


if __name__ == '__main__':
    main()
