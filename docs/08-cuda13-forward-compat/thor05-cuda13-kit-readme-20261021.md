# Thor05 CUDA13 Comfy deployment kit (2026-10-21)

## Scope

This kit is for Thor05 only. It keeps the original kernel,
nvgpu driver, system libnvrm libraries, system CUDA alternatives, and boot
configuration unchanged. It does not contain credentials or model weights.

## What is included

- `compat-orin-13.2.tar.gz`: official `cuda-compat-orin-13-2` user-space driver
  compatibility libraries, package version `13.2.44290101-1`.
- `bench-runtime-wheels.tar`: offline AArch64 Python/CUDA stack, including
  Torch `2.13.0+cu130`, Triton `3.7.1`, and Comfy Kitchen `0.2.31`.
- `comfy-source.tar.gz`: working Comfy `comfy/` source snapshot, commit
  `8f04fd6ecf4a548937971e840af2a985529fab76`.
- `bench-fixtures.tar.gz` and `bench-results.tar.gz`: reproducible fixtures and
  all 24 timing/numerical records.
- `prepare-bench-env.py`, `launch-thor05-cu13.py`, and benchmark scripts.

## Why CUDA13 wins here

ComfyUI's `comfy/quant_ops.py` explicitly disables the Comfy Kitchen CUDA
backend when `torch.version.cuda < 13`. The CUDA12 control therefore uses the
eager dequantization path. The CUDA13 run uses the native ConvRot W4A4/W4A8
quantized tensor-core GEMM path.

### Music3 complete 36-layer DiT forward

| Latent length | CUDA12 eager | CUDA13 native | Speedup | NRMSE | Cosine |
|---:|---:|---:|---:|---:|---:|
| 256 | 242.667 ms | 57.773 ms | 4.20x | 3.76% | 0.999292 |
| 512 | 315.542 ms | 98.999 ms | 3.19x | 4.10% | 0.999159 |

This loads the existing 2.50GB INT8 ConvRot DiT. It is one complete forward,
not text encoding, sampling loops, VAE, audio generation, or an end-to-end
latency claim.

### H3 real weight layers

| M | CUDA12 eager | CUDA13 native | Speedup | NRMSE |
|---:|---:|---:|---:|---:|
| 256 | 2.988 ms | 0.454 ms | 6.58x | 0.441% |
| 1024 | 4.373 ms | 1.539 ms | 2.84x | 0.450% |

### Qwen-Image 2.1 real weight layers

| M | CUDA12 eager | CUDA13 native | Speedup | NRMSE |
|---:|---:|---:|---:|---:|
| 256 | 1.353 ms | 0.247 ms | 5.47x | 0.466% |
| 1024 | 2.117 ms | 0.840 ms | 2.52x | 0.462% |

Layer measurements use fixed random BF16 activations, not activations collected
from full workflows. Numerical gates passed. Perceptual quality still requires
real image/video/audio generation qualification before production promotion.

## Deploy procedure

1. Copy this directory to persistent Thor05 storage, for example
   `/zeekr_data/ai_workspace/cuda13-thor05-20261021`.
2. Verify `MANIFEST.sha256` with `sha256sum -c MANIFEST.sha256`.
3. Extract the three archives in-place:

   ```bash
   tar -xzf compat-orin-13.2.tar.gz
   tar -xf bench-runtime-wheels.tar
   tar -xzf comfy-source.tar.gz
   python3 prepare-bench-env.py
   ```

4. Back up the target ComfyUI's existing `comfy/` directory, then replace it
   with `bench/comfy-source/comfy/`. This snapshot carries the model and quant
   routing code needed by the benchmark; replacing only `quant_ops.py` is not a
   qualified deployment.
5. Run an isolated import check:

   ```bash
   python3 launch-thor05-cu13.py --import-only
   ```

6. Launch the existing ComfyUI entry point only while Thor05 GPU work is idle:

   ```bash
   python3 launch-thor05-cu13.py \
     --comfy-root /path/to/ComfyUI \
     -- /usr/bin/python3 main.py --listen 127.0.0.1 --port 8188
   ```

The launcher prepends the isolated CUDA13 Python and library paths. It does not
install a service or change system defaults.

## Safety and rollback

- Before launching, confirm no active GPU workload, inspect dmesg for Xid/NVRM
  errors, and stop the current GPU service only by its qualified procedure.
- Keep the original kernel and system driver packages. Do not install the deb.
- Roll back by stopping the isolated process, restoring the backed-up `comfy/`
  directory, and removing the copied kit directory.
- If Xid, GPU fault, timeout, unkillable D state, or hugepage leakage occurs,
  stop, preserve dmesg/process/memory evidence, and escalate; do not continue.

## Evidence root

The board's original working directory was
`/zeekr_data/ai_workspace/h3-cu13-compat-20261021`. This model-library kit is
the clean redistribution copy; private LLM environment snapshots were excluded.
