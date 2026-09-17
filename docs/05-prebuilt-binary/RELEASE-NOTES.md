# Prebuilt: stable-diffusion.cpp for NVIDIA DRIVE Thor (sm_101 / CUDA 12.8)

预编译的 `stable-diffusion.cpp` aarch64 二进制，面向 **NVIDIA DRIVE Thor**（Tegra264 / sm_101a / DriveOS 7.0.3 / CUDA 12.8）。
下载后校验 → 上板即可运行，**不需要交叉编译**，也不需要 PyTorch / ComfyUI。

Prebuilt aarch64 binaries of `stable-diffusion.cpp` for NVIDIA DRIVE Thor boards — download, verify, run. No cross-compilation, no PyTorch/ComfyUI.

---

## Files / 文件

| File | Size | sha256 |
|---|---|---|
| `sd-cli` | 100,813,608 B (96.1 MiB) | `02cee738e66d4a0d798a7d778d6bbaeea064ce37f06c3f0bf5b536f7addf1f10` |
| `sd-server` | 101,320,080 B (96.6 MiB) | `f8cd7e0d1f9bfa3b78d144df0a4d1d830ce4be84315f8012a692c3b95863e567` |
| `sd-cpp-thor-sm101-20260917.tar.gz`（两个二进制 + `SHA256SUMS`） | 115,564,708 B (110.2 MiB) | `647e388bf640991b147ebe4825c40ecd180f442200e950c3363a6164b64c39d7` |

> 独立 sha256 校验（推荐）：
> ```bash
> sha256sum -c <<'EOF'
> 02cee738e66d4a0d798a7d778d6bbaeea064ce37f06c3f0bf5b536f7addf1f10  sd-cli
> f8cd7e0d1f9bfa3b78d144df0a4d1d830ce4be84315f8012a692c3b95863e567  sd-server
> EOF
> # 两行都应输出 "OK"
> ```
> 用 tar.gz 的话，包内自带 `SHA256SUMS`：`tar -xzf <包> && sha256sum -c SHA256SUMS`

## Build provenance / 构建来源（可复现）

| 项 | 值 |
|---|---|
| Upstream | [leejet/stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) @ `59c23bc` (2026-09-15) |
| Target | aarch64 (GNU/Linux), CUDA **12.8** (sbsa), `CMAKE_CUDA_ARCHITECTURES=101` |
| Flags | `-DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SHARED_LIBS=OFF` |
| Toolchain | aarch64 GNU 14 + CUDA 12.8 sbsa (`targets/sbsa-linux`) + `qemu-aarch64-static` for codegen |
| Source paths | compiled with `-ffile-prefix-map` ⇒ **no build-host absolute paths embedded**（已逐项扫描确认 0 处） |
| SASS | `cuobjdump -lelf` 显示 `sm_101` cubin |

## Verified on real hardware / 真机实测

在 DRIVE Thor（DriveOS 7.0.3，CUDA 12.8 运行库）上实跑通过：

| 用例 | 结果 |
|---|---|
| `sd-cli --help` / `ldd` | 可执行，**缺失依赖 0** |
| **Z-Image-Turbo 512² / 8 步** | `generate_image completed in 11.94 s`（墙钟 13 s），产物 421,535 B |
| **Z-Image-Turbo 1024² / 8 步** | `generate_image completed in 51.90 s`（墙钟 53 s），产物 1,677,499 B |
| 显存归还 | 两次运行后大页池空闲页回到满值（无泄漏） |

（MiniMax-H3 视频 + 音频生成另在同类板卡上实测跑通，见文档。）

## What it supports / 支持范围

- **Z-Image-Turbo** 文生图（512²/1024²，8 步）
- **MiniMax-H3** 视频 + 音频联合生成（`-M vid_gen`；权重 43 GB，需独占大页池）
- 以及上游 `sd.cpp` 支持的其它扩散模型

## Quick start / 快速开始

```bash
# 1) 上板（板上没有 rsync，用 scp）
scp sd-cli sd-server user@<board>:<DATA_DIR>/sd-cpp/bin/
ssh user@<board> 'chmod 755 <DATA_DIR>/sd-cpp/bin/sd-*; <DATA_DIR>/sd-cpp/bin/sd-cli --help | head'

# 2) 放模型（二进制不含权重，获取方式与校验值见 docs/05-prebuilt-binary/prebuilt-binary-quickstart.md §3）

# 3) 出图（Z-Image，512² 约 13 秒）
cd <DATA_DIR>/sd-cpp && ./bin/sd-cli \
  --diffusion-model models/z_image_turbo-Q4_K_S.gguf \
  --vae models/ae.safetensors \
  --llm models/Qwen3-4B-Q4_K_M.gguf \
  --cfg-scale 1.0 --steps 8 --diffusion-fa -W 512 -H 512 \
  -p "a red apple on a wooden table, studio light" -o out/test.png
```

## Requirements / 运行要求

- NVIDIA DRIVE Thor（**sm_101**）。Jetson Orin (sm_87) / Xavier / 桌面 GPU / x86 **不适用**，需自行编译。
- CUDA **12.8** 运行库（`libcudart` / `libcublas` / `libcublasLt`）——DriveOS 7.0.3 自带；`ldd` 无缺失即可运行。
- 无需 python / torch / docker / 板上编译器。
- ⚠️ 本平台**大页池 = GPU 显存**：**不要**为了腾显存去写 `nr_hugepages=0`（会造成"假 OOM"）。

详细步骤、模型获取、池预算与排障：
**docs/05-prebuilt-binary/prebuilt-binary-quickstart.md**

---

## Two generation lines, one binary / 两条生成线，同一个二进制

This binary serves both lines — no separate build is needed:

| 线 | 模式 | 产出 |
|---|---|---|
| **图片（Z-Image-Turbo）** | 默认（文生图） | PNG，512² ≈13.5 s / 1024² ≈57 s |
| **视频 + 音频（MiniMax-H3）** | `-M vid_gen`（**必须带 `--audio-vae`**） | **带音轨的视频**（h264 + aac 32 kHz 立体声；音轨与画面同一次生成） |

- 图片线快速上手：`docs/05-prebuilt-binary/prebuilt-binary-quickstart.md`
- **视频线快速上手**：`docs/02-h3-video-audio/h3-video-audio.md` + `docs/04-resolution-vram/resolution-vram-measured.md`
  （四个权重文件与校验值、**独占池**流程、完整命令、产出验声、排障）
- **示例视频**（模型生成的音轨，可直接播放）：`samples/h3_864x480_person.mp4`、`samples/h3_1024x576_person.mp4`、`samples/h3_768p_person.mp4`
  （仓库 `samples/` 目录，三档分辨率实测成片，带音轨）

> ⚠️ H3 权重要 **43 GB**，须**独占**本平台的大页池（≈46 GiB）：跑前停掉占 GPU 的服务、
> 跑完启回；**不要**为了腾显存去写 `nr_hugepages=0`。
