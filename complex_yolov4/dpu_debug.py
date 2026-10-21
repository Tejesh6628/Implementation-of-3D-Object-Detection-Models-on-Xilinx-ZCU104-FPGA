#!/usr/bin/env python3
"""
dpu_debug.py  —  Step-by-step DPU runner diagnostic for ZCU104
===============================================================
Run this on the board to find exactly which VART call segfaults.

Usage:
    python3 dpu_debug.py
    python3 dpu_debug.py --xmodel complex_yolov4.xmodel
"""
import faulthandler, sys, os, argparse
faulthandler.enable()   # prints C-level traceback on SIGSEGV

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--xmodel', default='complex_yolov4.xmodel')
args = ap.parse_args()

XMODEL = args.xmodel
if not os.path.exists(XMODEL):
    sys.exit(f"ERROR: {XMODEL} not found")

print(f"\nDPU Diagnostic for: {XMODEL}")
print("=" * 50)

# ── Step 1: import xir ────────────────────────────────
print("\n[1] import xir ... ", end='', flush=True)
import xir
print("OK")

# ── Step 2: import vart ───────────────────────────────
print("[2] import vart ... ", end='', flush=True)
import vart
print("OK")

# ── Step 3: deserialize xmodel ───────────────────────
print("[3] xir.Graph.deserialize() ... ", end='', flush=True)
graph = xir.Graph.deserialize(XMODEL)
print("OK")

# ── Step 4: get root subgraph ─────────────────────────
print("[4] get_root_subgraph() ... ", end='', flush=True)
root = graph.get_root_subgraph()
print("OK")

# ── Step 5: list child subgraphs ──────────────────────
print("[5] toposort_child_subgraph() ... ", end='', flush=True)
children = root.toposort_child_subgraph()
print(f"OK  ({len(children)} children)")
for i, sg in enumerate(children):
    has_dev = sg.has_attr("device")
    dev     = sg.get_attr("device") if has_dev else "NO_DEVICE"
    name    = sg.get_name() if hasattr(sg, 'get_name') else '?'
    print(f"   [{i}] device={dev}  name={name}")

dpu_sgs = [s for s in children
           if s.has_attr("device") and s.get_attr("device").upper() == "DPU"]
if not dpu_sgs:
    sys.exit("ERROR: No DPU subgraph found in xmodel!")

# ── Step 6: create runner ─────────────────────────────
print("[6] vart.Runner.create_runner() ... ", end='', flush=True)
runner = vart.Runner.create_runner(dpu_sgs[0], "run")
print("OK")

# ── Step 7: read tensor metadata ─────────────────────
print("[7] get_input/output_tensors() ... ", end='', flush=True)
in_tensors  = runner.get_input_tensors()
out_tensors = runner.get_output_tensors()
print("OK")

print(f"\n  Input tensors  : {len(in_tensors)}")
for t in in_tensors:
    print(f"    name  = {t.name}")
    print(f"    dims  = {t.dims}")
    print(f"    dtype = {t.dtype}")

print(f"\n  Output tensors : {len(out_tensors)}")
for t in out_tensors:
    print(f"    name  = {t.name}")
    print(f"    dims  = {t.dims}")
    print(f"    dtype = {t.dtype}")

# ── Step 8: read fix_point ────────────────────────────
print("\n[8] get_attr('fix_point') ...")
try:
    in_fp  = in_tensors[0].get_attr("fix_point")
    print(f"  Input  fix_point = {in_fp}  (scale = {2**in_fp:.1f})")
except Exception as e:
    print(f"  [WARN] fix_point not available via get_attr: {e}")
    in_fp = 7   # safe default for 0-1 normalised BEV input

for i, ot in enumerate(out_tensors):
    try:
        fp = ot.get_attr("fix_point")
        print(f"  Output[{i}] fix_point = {fp}  (scale = {2**-fp:.6f})")
    except Exception as e:
        print(f"  [WARN] Output[{i}] fix_point not available: {e}")

# ── Step 9: allocate C-contiguous buffers ────────────
print("\n[9] Allocating C-contiguous int8 buffers ...")
in_dims  = tuple(in_tensors[0].dims)
out_dims = [tuple(t.dims) for t in out_tensors]
print(f"  Input  buffer : {in_dims}")
for i, d in enumerate(out_dims):
    print(f"  Output[{i}] buffer : {d}")

input_buf  = [np.zeros(in_dims, dtype=np.int8, order='C')]
output_buf = [np.zeros(d,       dtype=np.int8, order='C') for d in out_dims]

# Verify buffers are genuinely C-contiguous
assert input_buf[0].flags['C_CONTIGUOUS'],  "Input buffer not C-contiguous!"
for i, b in enumerate(output_buf):
    assert b.flags['C_CONTIGUOUS'], f"Output[{i}] not C-contiguous!"
print("  Buffers OK (all C-contiguous)")

# ── Step 10: execute_async ────────────────────────────
print("\n[10] runner.execute_async() ... ", end='', flush=True)
job_id = runner.execute_async(input_buf, output_buf)
print(f"OK  (job_id={job_id})")

# ── Step 11: wait ─────────────────────────────────────
print("[11] runner.wait() ... ", end='', flush=True)
runner.wait(job_id)
print("OK")

# ── Step 12: inspect output ───────────────────────────
print("\n[12] Output value ranges (zeros in → should be near zero):")
for i, (buf, ot) in enumerate(zip(output_buf, out_tensors)):
    print(f"  Output[{i}] shape={buf.shape}  min={buf.min()}  max={buf.max()}")

del runner
print("\n" + "=" * 50)
print("ALL STEPS PASSED — DPU runner works correctly!")
print("\nNext: run run_on_board.py")
