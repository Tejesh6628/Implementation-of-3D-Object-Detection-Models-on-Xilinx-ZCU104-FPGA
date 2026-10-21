"""
STEP 3: Compile xmodel for DPUCZDX8G (ZCU104 B4096)
=====================================================
Run INSIDE Vitis AI 3.0 Docker container on HOST.

This script wraps the `vai_c_xir` compiler (Vitis AI 3.0 compiler) to
produce the final .xmodel for the ZCU104 DPU.

Usage:
    python 3_compile_xmodel.py \
        --input_model ./quantized/SECONDBEVWrapper_int.xmodel \
        --output_dir  ./compiled \
        --arch_json   /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json

Or you can run vai_c_xir directly (shown as shell commands inside):
    vai_c_xir \
        -x ./quantized/SECONDBEVWrapper_int.xmodel \
        -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
        -o ./compiled \
        -n second_bev

The script also prints the DPU subgraph fingerprint for verification.
"""

import argparse, subprocess, sys
from pathlib import Path

# ── Vitis AI 3.0 DPU arch path on the Docker image ───────────────────────────
ARCH_JSON_DEFAULT = '/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input_model', required=True,
                   help='Path to quantized int.xmodel from step 2b')
    p.add_argument('--output_dir',  default='./compiled')
    p.add_argument('--arch_json',   default=ARCH_JSON_DEFAULT)
    p.add_argument('--net_name',    default='second_bev',
                   help='Name prefix for compiled output files')
    return p.parse_args()


def compile(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        'vai_c_xir',
        '-x', args.input_model,
        '-a', args.arch_json,
        '-o', str(out),
        '-n', args.net_name,
    ]

    print('Running:', ' '.join(cmd))
    ret = subprocess.run(cmd, capture_output=False)
    if ret.returncode != 0:
        print('\n[ERROR] vai_c_xir failed. Check the output above.')
        sys.exit(1)

    xmodel_out = out / f'{args.net_name}.xmodel'
    print(f'\nCompilation successful: {xmodel_out}')

    # ── Print subgraph summary using xir Python API ───────────────────────────
    try:
        import xir
        g = xir.Graph.deserialize(str(xmodel_out))
        sg = g.get_subgraph()
        children = sg.toposort_child_subgraph()
        print(f'\nSubgraph summary for {xmodel_out.name}:')
        print(f'  Total subgraphs : {len(children)}')
        for i, s in enumerate(children):
            attrs = s.get_attr('device') if s.has_attr('device') else 'unknown'
            in_t  = [t.name for t in s.get_input_tensors()]
            out_t = [t.name for t in s.get_output_tensors()]
            print(f'  [{i}] device={attrs}')
            print(f'       inputs : {in_t}')
            print(f'       outputs: {out_t}')
    except ImportError:
        print('(xir module not available; skipping subgraph inspection)')

    return str(xmodel_out)


def main():
    args = parse_args()

    if not Path(args.input_model).exists():
        print(f'[ERROR] Input model not found: {args.input_model}')
        print('  Run step 2b first (2b_qat_quantize.py --mode QAT then --mode test)')
        sys.exit(1)

    if not Path(args.arch_json).exists():
        print(f'[ERROR] arch.json not found: {args.arch_json}')
        print('  Make sure you are inside the Vitis AI 3.0 Docker container.')
        print('  Default path: /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json')
        sys.exit(1)

    out_xmodel = compile(args)
    print(f'\n[DONE] Copy {out_xmodel} to your ZCU104 SD card or via scp.')
    print('Example:')
    print(f'  scp {out_xmodel} root@<board_ip>:/home/root/second_deploy/')


if __name__ == '__main__':
    main()
