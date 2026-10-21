"""
STEP 2b (CPU / PTQ version): Post-Training Quantization for SECOND - Vitis AI 3.0
===================================================================================
Run INSIDE the Vitis AI 3.0 Docker container (CPU image).

This script replaces 2b_qat_quantize.py when no GPU is available.
It does NOT import OpenPCDet (no CUDA extensions needed in Vitis AI env).
It reads the 200 bev_*.npy calibration files you already generated.

Environment: activate Vitis AI conda env (NOT your pcdet env)
    conda activate vitis-ai-pytorch    (or your Vitis env name)
    python -c "from pytorch_nndct.apis import torch_quantizer; print('OK')"

Two-pass usage:
    # Pass 1 - calibration (sets quantization scale factors)
    python 2b_ptq_cpu_quantize.py \
        --ckpt       ../SECOND_model/checkpoints/checkpoint_epoch_80.pth \
        --calib_dir  ./calib_data \
        --output_dir ./quantized \
        --mode calib \
        --device cpu

    # Pass 2 - test / export xmodel
    python 2b_ptq_cpu_quantize.py \
        --ckpt       ../SECOND_model/checkpoints/checkpoint_epoch_80.pth \
        --calib_dir  ./calib_data \
        --output_dir ./quantized \
        --mode test \
        --device cpu

After both passes you will have:
    ./quantized/SECONDBEVWrapper_int.xmodel  <- feed to 3_compile_xmodel.py
"""

import sys
import os
import io
import re as _re
import shutil
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

# ---- pytorch_nndct (must be inside Vitis AI conda env) --------------------
try:
    from pytorch_nndct.apis import torch_quantizer, Inspector
    NNDCT_AVAILABLE = True
except ImportError:
    print("[WARNING] pytorch_nndct not found.")
    print("  You must be inside the Vitis AI conda environment.")
    print("  Run: conda activate vitis-ai-pytorch")
    NNDCT_AVAILABLE = False


# ===========================================================================
# BEV spatial constants (fixed by dataset, not by anchor config)
# ===========================================================================
C_BEV         = 256    # NUM_BEV_FEATURES from second.yaml
H_BEV, W_BEV = 200, 176  # BEV grid after 8x sparse backbone downsampling
# NOTE: NUM_CLS_OUT / NUM_BOX_OUT / NUM_DIR_OUT are NOT hardcoded here.
# They are read dynamically from the checkpoint in read_model_shapes().
# This avoids size mismatch errors when the anchor config differs.


# ===========================================================================
# Pure-PyTorch reconstruction of backbone_2d + dense_head
# (NO OpenPCDet imports, NO CUDA extensions)
# ===========================================================================

def _make_block(in_ch, out_ch, num_convs, stride=1):
    """
    Down-path conv block used in BaseBEVBackbone.

    Uses Conv2d(padding=1) directly instead of ZeroPad2d + Conv2d(no padding).
    Both are mathematically equivalent, but ZeroPad2d uses aten::pad which is
    NOT defined in XIR — it gets assigned to CPU and fragments the DPU subgraph.
    Using padding=1 inside Conv2d keeps the entire block on DPU.
    """
    layers = [
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    ]
    for _ in range(num_convs - 1):
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ]
    return nn.Sequential(*layers)


def _make_deblock(in_ch, out_ch, up_stride):
    """
    Up-path deconv block used in BaseBEVBackbone.

    OpenPCDet ALWAYS uses ConvTranspose2d here, even when up_stride=1
    (it becomes a 1x1 transposed conv, equivalent to a 1x1 conv but with
    weight shape [in_ch, out_ch, 1, 1] matching the checkpoint).
    Using nn.Conv2d for stride=1 gives a DIFFERENT weight shape and causes
    the 'size mismatch for backbone_2d.deblocks.0.0.weight' error.
    """
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, up_stride, stride=up_stride, bias=False),
        nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


class BaseBEVBackbone(nn.Module):
    """
    Mirrors OpenPCDet BaseBEVBackbone.
    Channel counts and number of convolutions are inferred from the
    checkpoint at runtime via read_model_shapes().
    """
    def __init__(self, cfg):
        """
        cfg: dict with keys:
          c_bev, block0_out, block1_out,
          num_convs_block0, num_convs_block1,
          deblock0_out, deblock0_stride,
          deblock1_out, deblock1_stride
        """
        super().__init__()
        self.blocks = nn.ModuleList([
            _make_block(cfg['c_bev'],      cfg['block0_out'], cfg['num_convs_block0'], stride=1),
            _make_block(cfg['block0_out'], cfg['block1_out'], cfg['num_convs_block1'], stride=2),
        ])
        self.deblocks = nn.ModuleList([
            _make_deblock(cfg['block0_out'], cfg['deblock0_out'], cfg['deblock0_stride']),
            _make_deblock(cfg['block1_out'], cfg['deblock1_out'], cfg['deblock1_stride']),
        ])

    def forward(self, data_dict):
        x = data_dict['spatial_features']
        ups = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            ups.append(self.deblocks[i](x))
        data_dict['spatial_features_2d'] = torch.cat(ups, dim=1)
        return data_dict


class AnchorHeadSingle(nn.Module):
    """Mirrors OpenPCDet AnchorHeadSingle. Channel counts set from checkpoint."""
    def __init__(self, cfg):
        """
        cfg: dict with keys:
          head_in_ch, num_cls_out, num_box_out, num_dir_out
        """
        super().__init__()
        in_ch = cfg['head_in_ch']
        self.num_cls_out = cfg['num_cls_out']
        self.num_box_out = cfg['num_box_out']
        self.num_dir_out = cfg['num_dir_out']
        self.conv_cls = nn.Conv2d(in_ch, self.num_cls_out, 1)
        self.conv_box = nn.Conv2d(in_ch, self.num_box_out, 1)
        self.conv_dir = nn.Conv2d(in_ch, self.num_dir_out, 1)

    def forward(self, data_dict):
        feat = data_dict['spatial_features_2d']   # [B, head_in_ch, H, W]
        #
        # IMPORTANT: Do NOT add permute/view/reshape here.
        # Vitis AI PTQ (export_quant_config) fails with KeyError when the
        # final conv outputs flow into non-DPU ops (permute, reshape) before
        # being returned, because NNDCT cannot collect output quant stats.
        # Returning raw [B, C, H, W] tensors keeps everything DPU-quantizable.
        # The reshape (-> [B, N_anchors, cls/7/2]) is done on the board in
        # 4_run_on_board.py after DPU inference.
        #
        cls  = self.conv_cls(feat)   # [B, num_cls_out, H, W]
        box  = self.conv_box(feat)   # [B, num_box_out, H, W]
        dirp = self.conv_dir(feat)   # [B, num_dir_out, H, W]
        data_dict['batch_cls_preds']     = cls
        data_dict['batch_box_preds']     = box
        data_dict['batch_dir_cls_preds'] = dirp
        return data_dict


class SECONDBEVWrapper(nn.Module):
    """
    BEV-only wrapper: spatial_features -> backbone_2d -> dense_head
    -> (cls_preds, box_preds, dir_preds)

    Outputs are raw spatial tensors (no reshape):
      cls_preds : [B, num_cls_out, H, W]
      box_preds : [B, num_box_out, H, W]
      dir_preds : [B, num_dir_out, H, W]

    The reshape to [B, N_anchors, C] is intentionally omitted here so that
    NNDCT can collect quantization stats on the conv outputs without being
    blocked by non-DPU permute/reshape operations.
    """
    def __init__(self, backbone_cfg, head_cfg):
        super().__init__()
        self.backbone_2d = BaseBEVBackbone(backbone_cfg)
        self.dense_head  = AnchorHeadSingle(head_cfg)

    def forward(self, spatial_features):
        dd = {'spatial_features': spatial_features}
        dd = self.backbone_2d(dd)
        dd = self.dense_head(dd)
        return dd['batch_cls_preds'], dd['batch_box_preds'], dd['batch_dir_cls_preds']


# ===========================================================================
# Weight loading from OpenPCDet checkpoint
# ===========================================================================

class _NumpyCompatUnpickler(pickle.Unpickler):
    """
    Custom Unpickler that redirects 'numpy._core.*' -> 'numpy.core.*'.

    The checkpoint was saved on a machine with NumPy >= 1.25 which moved
    internal modules to 'numpy._core'. The Vitis AI Docker has NumPy 1.x
    where those modules still live under 'numpy.core'. This intercepts every
    find_class call during unpickling and silently remaps the module path.
    """
    def find_class(self, module, name):
        # Remap numpy._core -> numpy.core
        if module.startswith('numpy._core'):
            module = module.replace('numpy._core', 'numpy.core', 1)
        return super().find_class(module, name)


def _torch_load_compat(path):
    """
    Drop-in replacement for torch.load that handles the numpy._core
    ModuleNotFoundError on old-NumPy environments (Vitis AI Docker).

    Strategy:
      1. Try the normal torch.load first (works if NumPy versions match).
      2. On ModuleNotFoundError for numpy._core, fall back to reading the
         raw bytes and using our patched Unpickler via torch's own
         _load helper, or manually via pickle.
    """
    try:
        return torch.load(path, map_location='cpu')
    except ModuleNotFoundError as e:
        if 'numpy._core' not in str(e) and 'numpy' not in str(e):
            raise  # unrelated error, re-raise

    print(f'  [INFO] numpy._core not found in this env (NumPy < 1.25).')
    print(f'  [INFO] Applying numpy._core -> numpy.core compatibility fix ...')

    # Patch sys.modules so any subsequent import of numpy._core.* works
    import types, importlib
    try:
        import numpy._core  # noqa: F401
    except ImportError:
        _shim = types.ModuleType('numpy._core')
        _shim.__dict__.update(np.core.__dict__)
        sys.modules['numpy._core'] = _shim
        for sub in ['multiarray', 'umath', 'numeric', 'fromnumeric',
                    'function_base', 'shape_base', 'arrayprint',
                    'overrides', 'numerictypes']:
            key = f'numpy._core.{sub}'
            if key not in sys.modules:
                try:
                    sys.modules[key] = importlib.import_module(f'numpy.core.{sub}')
                except ImportError:
                    pass

    # Re-try with patched sys.modules
    try:
        return torch.load(path, map_location='cpu')
    except Exception:
        pass

    # Last resort: use our custom Unpickler on the raw file bytes.
    # torch checkpoints are zip files; we use torch's own zip reader.
    import zipfile
    with zipfile.ZipFile(path) as zf:
        # The main pickle archive is always named 'archive/data.pkl'
        names = zf.namelist()
        pkl_name = next((n for n in names if n.endswith('data.pkl')), None)
        if pkl_name is None:
            raise RuntimeError(
                f'Cannot find data.pkl inside {path}. '
                'The checkpoint format may be unsupported.')
        with zf.open(pkl_name) as pkl_file:
            data = _NumpyCompatUnpickler(pkl_file).load()
    return data


def read_model_shapes(ckpt_path):
    """
    Load the checkpoint ONCE, read actual weight shapes, and return:
      backbone_cfg  - dict for BaseBEVBackbone
      head_cfg      - dict for AnchorHeadSingle
      state_dict    - full model_state dict (cached to avoid double load)

    This function is the single source of truth for model architecture.
    It reads shapes directly from the checkpoint so it never mis-matches.
    """
    print(f'Reading checkpoint shapes: {ckpt_path}')
    ckpt = _torch_load_compat(ckpt_path)
    sd   = ckpt.get('model_state', ckpt)

    # ── Backbone block0 first conv: shape [out0, c_bev, k, k] ──────────────
    # With ZeroPad2d removed: Conv2d is now at index 0 in the Sequential.
    # Fallback to index 1 for old checkpoints that had ZeroPad2d at index 0.
    b0 = (sd.get('backbone_2d.blocks.0.0.weight') or   # new: no ZeroPad2d
          sd.get('backbone_2d.blocks.0.1.weight'))      # old: ZeroPad2d at [0]
    c_bev      = int(b0.shape[1]) if b0 is not None else C_BEV
    block0_out = int(b0.shape[0]) if b0 is not None else 128

    b1 = (sd.get('backbone_2d.blocks.1.0.weight') or   # new: no ZeroPad2d
          sd.get('backbone_2d.blocks.1.1.weight'))      # old: ZeroPad2d at [0]
    block1_out = int(b1.shape[0]) if b1 is not None else 256

    # ── Deblocks: ConvTranspose2d shape is [in_ch, out_ch, kH, kW] ─────────
    db0 = sd.get('backbone_2d.deblocks.0.0.weight')
    deblock0_out    = int(db0.shape[1]) if db0 is not None else 256
    deblock0_stride = int(db0.shape[2]) if db0 is not None else 1   # kernel=stride

    db1 = sd.get('backbone_2d.deblocks.1.0.weight')
    deblock1_out    = int(db1.shape[1]) if db1 is not None else 256
    deblock1_stride = int(db1.shape[2]) if db1 is not None else 2

    # ── Count Conv2d layers per block from checkpoint ────────────────────────
    # OpenPCDet uses 1 initial conv + LAYER_NUMS inner convs per block.
    # With LAYER_NUMS=[5,5] that is 6 Conv2d per block (not 5!).
    # We count 4D weight tensors (Conv2d) to get the exact number.
    num_convs_block0 = sum(1 for k in sd
                          if k.startswith('backbone_2d.blocks.0.')
                          and k.endswith('.weight') and sd[k].dim() == 4)
    num_convs_block1 = sum(1 for k in sd
                          if k.startswith('backbone_2d.blocks.1.')
                          and k.endswith('.weight') and sd[k].dim() == 4)

    backbone_cfg = dict(
        c_bev=c_bev,
        block0_out=block0_out,
        block1_out=block1_out,
        num_convs_block0=num_convs_block0,
        num_convs_block1=num_convs_block1,
        deblock0_out=deblock0_out,
        deblock0_stride=deblock0_stride,
        deblock1_out=deblock1_out,
        deblock1_stride=deblock1_stride,
    )

    # ── Dense head output channels ──────────────────────────────────────────
    cls_w = sd.get('dense_head.conv_cls.weight')
    box_w = sd.get('dense_head.conv_box.weight')
    # OpenPCDet stores dir head as 'conv_dir_cls' but our wrapper uses 'conv_dir'
    dir_w = sd.get('dense_head.conv_dir_cls.weight',
                   sd.get('dense_head.conv_dir.weight'))

    num_cls_out = int(cls_w.shape[0]) if cls_w is not None else 6
    num_box_out = int(box_w.shape[0]) if box_w is not None else 42
    num_dir_out = int(dir_w.shape[0]) if dir_w is not None else 12
    head_in_ch  = int(cls_w.shape[1]) if cls_w is not None else 512

    head_cfg = dict(
        head_in_ch=head_in_ch,
        num_cls_out=num_cls_out,
        num_box_out=num_box_out,
        num_dir_out=num_dir_out,
    )

    print(f'  Backbone : c_bev={c_bev}, blocks=[{block0_out},{block1_out}]')
    print(f'             num_convs=[{num_convs_block0},{num_convs_block1}]')
    print(f'             deblocks=[{deblock0_out}(s={deblock0_stride}), {deblock1_out}(s={deblock1_stride})]')
    print(f'  Head     : in={head_in_ch}, cls={num_cls_out}, box={num_box_out}, dir={num_dir_out}')
    print(f'             → {num_box_out//7} anchors/loc, {num_dir_out//2} dir bins/anchor')

    return backbone_cfg, head_cfg, sd


def _remap_block_key(k):
    """
    Shift backbone block Sequential indices by -1.

    The checkpoint was saved with ZeroPad2d at position [0] inside each block
    Sequential, so Conv2d was at [1], BN at [2], etc.:
      blocks.0.1.weight  -> Conv2d   (checkpoint)
      blocks.0.2.weight  -> BN
      blocks.0.2.bias    -> BN
      blocks.0.4.weight  -> Conv2d   (next conv)
      ...

    Our model no longer has ZeroPad2d, so Conv2d is at [0], BN at [1]:
      blocks.0.0.weight  -> Conv2d   (our model)
      blocks.0.1.weight  -> BN
      blocks.0.1.bias    -> BN
      blocks.0.3.weight  -> Conv2d   (next conv)
      ...

    We subtract 1 from the layer index inside the block Sequential so
    the checkpoint key matches the current model key.
    Only applied to backbone_2d.blocks.X.Y.* keys (not deblocks or head).
    """
    m = _re.match(r'^(backbone_2d\.blocks\.\d+\.)(\d+)(\..+)$', k)
    if m:
        prefix, idx_str, suffix = m.groups()
        new_idx = int(idx_str) - 1   # ZeroPad2d removed → shift all down by 1
        if new_idx >= 0:
            return f'{prefix}{new_idx}{suffix}'
    return k


def load_bev_weights_from_sd(model, state_dict):
    """
    Load backbone_2d and dense_head weights from an already-loaded state dict.
    Key remappings applied:
      dense_head.conv_dir_cls.* -> dense_head.conv_dir.*
      backbone_2d.blocks.X.Y.*  -> backbone_2d.blocks.X.(Y-1).*
        (because ZeroPad2d was removed from _make_block, shifting all indices)
    """
    bev_state = {}
    has_zeropad = ('backbone_2d.blocks.0.0.weight' not in state_dict)
    
    for k, v in state_dict.items():
        if k.startswith('backbone_2d.') or k.startswith('dense_head.'):
            new_k = k.replace('dense_head.conv_dir_cls', 'dense_head.conv_dir')
            if has_zeropad:
                new_k = _remap_block_key(new_k)   # fix ZeroPad2d index shift
            bev_state[new_k] = v

    missing, unexpected = model.load_state_dict(bev_state, strict=False)
    if missing:
        print(f'  [WARN] {len(missing)} missing keys  -> first few: {missing[:3]}')
    if unexpected:
        print(f'  [WARN] {len(unexpected)} unexpected keys -> first few: {unexpected[:3]}')
    print(f'  Loaded {len(bev_state)} weight tensors into model.')
    return model


# ===========================================================================
# Calibration dataset
# ===========================================================================

class BEVCalibDataset(Dataset):
    """
    Reads BEV numpy arrays from calib_dir.
    Auto-detects naming pattern:
      bev_*.npy   -> output of 2a_prepare_calibration.py
      calib_*.npy -> user-supplied (e.g. calib_0000.npy ... calib_0199.npy)
      *.npy       -> fallback: any .npy file in the directory
    Each file must contain an array of shape [1, C, H, W] or [C, H, W].
    """
    def __init__(self, calib_dir):
        d = Path(calib_dir)
        self.files = []
        for pattern in ['bev_*.npy', 'calib_*.npy', '*.npy']:
            self.files = sorted(d.glob(pattern))
            if self.files:
                print(f'Calibration dataset: {len(self.files)} files '
                      f'(pattern "{pattern}" in {calib_dir})')
                break
        if not self.files:
            raise FileNotFoundError(
                f'No .npy calibration files found in {calib_dir}.\n'
                f'Expected bev_*.npy or calib_*.npy files.'
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(str(self.files[idx]))   # [1, C, H, W] or [C, H, W]
        if arr.ndim == 4:
            arr = arr[0]                       # squeeze batch dim -> [C, H, W]
        return torch.from_numpy(arr.astype(np.float32))


# ===========================================================================
# Argument parser
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='CPU-only PTQ quantization of SECOND BEV model for ZCU104'
    )
    p.add_argument('--ckpt',       required=True,
                   help='Path to checkpoint_epoch_80.pth')
    p.add_argument('--calib_dir',  required=True,
                   help='Dir containing bev_*.npy from 2a_prepare_calibration.py')
    p.add_argument('--output_dir', default='./quantized')
    p.add_argument('--batch_size', type=int, default=4,
                   help='Keep small (4-8) on CPU to avoid OOM')
    p.add_argument('--num_workers',type=int, default=0,
                   help='DataLoader workers (0 = main process, safe on CPU)')
    p.add_argument('--device',     default='cpu',
                   help='cpu  (GPU not present in CPU Vitis AI image)')
    p.add_argument('--target',     default='DPUCZDX8G_ISA1_B4096',
                   help='DPU target for ZCU104 B4096')
    p.add_argument('--mode',       choices=['inspect', 'calib', 'test'],
                   required=True,
                   help='inspect=check DPU ops | calib=PTQ pass1 | test=export xmodel')
    return p.parse_args()


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    if not NNDCT_AVAILABLE:
        print('\n[ERROR] pytorch_nndct is not installed.')
        print('  Activate Vitis AI env first:')
        print('    conda activate vitis-ai-pytorch')
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device(args.device)

    print(f'Device  : {device}')
    print(f'Mode    : {args.mode}')
    print(f'Target  : {args.target}')
    print(f'Out dir : {out_dir}')

    # ---- Read checkpoint shapes & build model ------------------------------
    print('\nReading checkpoint to infer model architecture ...')
    backbone_cfg, head_cfg, cached_sd = read_model_shapes(args.ckpt)

    print('\nBuilding BEV model (pure PyTorch, no OpenPCDet) ...')
    bev_model = SECONDBEVWrapper(backbone_cfg, head_cfg)
    load_bev_weights_from_sd(bev_model, cached_sd)
    del cached_sd   # free memory
    bev_model = bev_model.to(device).eval()

    # Dummy input for quantizer graph tracing (batch=1 is enough)
    dummy_input = torch.zeros(1, C_BEV, H_BEV, W_BEV, device=device)

    # ---- INSPECT -----------------------------------------------------------
    if args.mode == 'inspect':
        print('\n=== INSPECT MODE ===')
        print('Checking which layers are DPU-compatible ...')
        inspector = Inspector(args.target)
        inspector.inspect(bev_model, (dummy_input,),
                          output_dir=str(out_dir / 'inspect'),
                          image_format='png', verbose=True)
        print(f'\nInspection report saved to: {out_dir / "inspect"}')
        print('Review the report, then run with --mode calib')
        return

    # ---- CALIB (PTQ calibration) -------------------------------------------
    if args.mode == 'calib':
        print('\n=== CALIBRATION (PTQ) MODE ===')
        print('Feeds all 200 calibration samples through the model')
        print('to determine per-layer INT8 quantization scale factors.\n')

        # Wipe the output directory before initialising the quantizer.
        # NNDCT checks whether <output_dir>/<ModelName>.py already exists; if
        # it does, it silently increments the name (quantized -> quantized_1)
        # but does NOT create that new directory, causing FileNotFoundError
        # when it tries to write bias_corr.pth.  Deleting the folder first
        # guarantees NNDCT always uses exactly the directory we specified.
        if out_dir.exists():
            print(f'  [INFO] Removing stale output dir: {out_dir}')
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        quantizer = torch_quantizer(
            'calib', bev_model, (dummy_input,),
            output_dir=str(out_dir), device=device, target=args.target,
        )
        qm = quantizer.quant_model.eval()

        dataset = BEVCalibDataset(args.calib_dir)
        loader  = DataLoader(dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)
        print(f'Running {len(dataset)} samples '
              f'({len(loader)} batches, batch_size={args.batch_size}) ...')
        with torch.no_grad():
            for i, batch in enumerate(loader):
                qm(batch.to(device))
                if (i + 1) % 10 == 0 or (i + 1) == len(loader):
                    print(f'  Batch {i+1}/{len(loader)} done')

        quantizer.export_quant_config()
        print(f'\n[DONE] Calibration complete.')
        print(f'  Quant config saved to: {out_dir}')
        print('\nNext step:  python 2b_ptq_cpu_quantize.py ... --mode test')
        return

    # ---- TEST / EXPORT xmodel ----------------------------------------------
    if args.mode == 'test':
        print('\n=== TEST / EXPORT MODE ===')
        print('Loading quantized model with calibrated scale factors ...\n')

        # ── Prevent NNDCT from auto-incrementing the output dir ──────────────
        # During calib, NNDCT wrote <out_dir>/SECONDBEVWrapper.py.
        # When test mode starts, it sees that file, assumes a collision, and
        # silently switches to <out_dir>_1/ — which it never creates — causing
        # FileNotFoundError later.  We delete only that generated .py file;
        # quant_info.json and bias_corr.pth from calib must be kept.
        stale_py = out_dir / 'SECONDBEVWrapper.py'
        if stale_py.exists():
            stale_py.unlink()
            print(f'  [INFO] Removed stale {stale_py.name} to prevent dir increment')

        quantizer = torch_quantizer(
            'test', bev_model, (dummy_input,),
            output_dir=str(out_dir), device=device, target=args.target,
        )
        qm = quantizer.quant_model.eval()

        # ── CRITICAL: ONLY use batch_size=1 during test mode ─────────────────
        # If any forward pass uses batch > 1, NNDCT bakes that batch size into
        # the activation shapes in the xmodel.  The DPU compiler then assigns
        # every conv-fix node to CPU (batch must be 1 on DPU), resulting in
        # 0 DPU subgraphs.  Use batch=1 throughout — for both verification and
        # the export pass.
        dataset     = BEVCalibDataset(args.calib_dir)
        loader_bs1  = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=args.num_workers)

        print('Running batch_size=1 quantized inference (required for DPU export) ...')
        with torch.no_grad():
            for i, batch in enumerate(loader_bs1):
                cls_p, box_p, dir_p = qm(batch.to(device))
                if i == 0:
                    print(f'  cls_preds : {cls_p.shape}  [B=1, C, H, W] raw spatial')
                    print(f'  box_preds : {box_p.shape}')
                    print(f'  dir_preds : {dir_p.shape}')

        print('\nExporting xmodel (batch=1 baked in) ...')
        quantizer.export_xmodel(output_dir=str(out_dir), deploy_check=True)
        xmodel_path = out_dir / 'SECONDBEVWrapper_int.xmodel'
        print(f'\n[DONE] xmodel exported to: {xmodel_path}')
        print(f'\nCompile command:')
        print(f'  vai_c_xir \\')
        print(f'      -x {xmodel_path} \\')
        print(f'      -a ./arch.json \\')
        print(f'      -o ./compiled \\')
        print(f'      -n SECOND_model')
        return


if __name__ == '__main__':
    main()
