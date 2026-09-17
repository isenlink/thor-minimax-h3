# Z-Image-Turbo 文生图：下载即用（DRIVE Thor）

> One-liner: Text-to-image with **Z-Image-Turbo** on an NVIDIA DRIVE Thor board, using a single
> cross-compiled `stable-diffusion.cpp` binary — **no PyTorch, no ComfyUI, no docker, no
> on-board compiler**. This page covers prerequisites, the three model files with checksums,
> the exact commands, a full parameter reference, measured numbers (including running
> **alongside** a resident LLM service), batching, and troubleshooting.
>
> 适用平台：DRIVE Thor（Tegra264 / sm_101a / DriveOS 7.0.3 / CUDA 12.8）
> 本仓库版：脱敏整理

---

> **单位口径（本文档统一）**
> - **GiB** 用于池／显存口径（本平台大页池 46 GiB 级；`sd.cpp` 打印的 "MB" 实际是 **MiB**）
> - **GB** 用于磁盘文件口径（1 GB = 10⁹ B）
> - 例：H3 权重磁盘占 **45.0 GB**（= 41.9 GiB），运行时占 VRAM **42.0 GiB**（框架报 `43 017 MB`）

## 0. 它产出什么

一张（或一批）**PNG 图片**，命令行直出。以下是**本仓库发布版二进制的实测产物**，就在
[`samples/`](../../samples/) 下：

| 样例 | 分辨率 / 步数 | 实测耗时（池空闲） |
|---|---|---|
| [`samples/z_image_512x512.png`](../../samples/) | 512×512 / 8 步 | **11.94 s**（墙钟 13 s） |
| [`samples/z_image_1024x1024.png`](../../samples/) | 1024×1024 / 8 步 | **51.90 s**（墙钟 53 s） |

Z-Image-Turbo 是**蒸馏版（Turbo）**模型：**8 步 + `--cfg-scale 1.0`** 就是它的正常用法
（不要按 SD1.5 的习惯用 20 步 / cfg 7，那只会更慢且画质更差）。

---

## 1. 与视频线的关键区别：**不需要独占池**

| | 图片线（Z-Image） | 视频线（MiniMax-H3） |
|---|---|---|
| 权重占用（CUDA 侧，VRAM） | **6.82 GiB**（sd.cpp 报 `6 980 MB`，实为 MiB） | **42.0 GiB**（报 `43 017 MB`） |
| 能否与同机常驻 LLM 服务**共存** | ✅ **可以**（实测：LLM 占 36.7 GiB 时仍有余量） | ❌ 必须独占池 |
| 跑之前要不要停服务 | **不用** | 必须（跑完启回） |

**实操含义**：图片线可以"随时跑"，不需要动同机的任何服务。只有当你开
**hires 放大（1280² 及以上）**时才会碰到池余量不足（见 §11）。

---

## 2. 硬前提（逐项确认）

| 项 | 要求 | 检查 |
|---|---|---|
| 平台 | **sm_101**（DRIVE Thor） | 跑起来若报 `no kernel image is available...` 即不匹配，需自行编译 |
| 二进制 | 本仓库 Release 的 `sd-cli`（**与视频线共用同一个二进制**） | `sd-cli --help` 有输出 |
| 磁盘 | 三件套合计 **≈ 7.3 GB**（Q4 档） | 放板上**数据分区**（可写目录通常只有一两个） |
| 运行库 | CUDA 12.8（DriveOS 7.0.3 自带） | `ldd sd-cli \| grep -c "not found"` = 0 |
| Python | **不需要**（命令行用；想服务化才用，且只用标准库） | — |

---

## 3. 三个模型文件（含实测通过的校验值）

| 角色 | 命令行参数 | 我们实测用的文件 | 字节数 | sha256 |
|---|---|---|---|---|
| DiT 主权重 | `--diffusion-model` | `z_image_turbo-Q4_K_S.gguf`（低显存档） | 4,658,972,736 | `c193f86867daf4e57007ec2fdb7bee8e7273b0f6ea20153ef2cb4037c4930257` |
| VAE | `--vae` | `ae.safetensors` | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |
| 文本编码器 | `--llm` | `Qwen3-4B-Q4_K_M.gguf` | 2,497,280,256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| **合计** | — | — | **7,491,557,380**（≈6.98 GiB） | — |

**上游来源**（以 `stable-diffusion.cpp` 官方 `docs/z_image.md` 为准）：

| 件 | 来源 |
|---|---|
| DiT（GGUF） | `huggingface.co/leejet/Z-Image-Turbo-GGUF` |
| DiT（safetensors） | `Comfy-Org/z_image_turbo` → `split_files/diffusion_models` |
| VAE | `black-forest-labs/FLUX.1-schnell`（Z-Image 复用该 VAE） |
| 文本编码器 | GGUF：`unsloth/Qwen3-4B-Instruct-2507` 的 GGUF 版；safetensors：`Comfy-Org/z_image_turbo` → `split_files/text_encoders` |

可选档位：

| 档 | 体积 | 说明 |
|---|---|---|
| `z_image_turbo-Q4_K_S.gguf` | 4.4 GB | **推荐起步**（实测与 Q8 速度几乎相同，见 §7.2） |
| `z_image_turbo-Q8_0.gguf` | 7.2 GB | 画质档；池余量更紧张 |
| 更低档（Q3_K / Q2_K 等） | 更小 | 上游支持，画质自测 |

> ⚠️ 这里的 `Qwen3-4B` 是**扩散流水线自己的文本编码器**（4B 级小模型），
> 与同机上可能常驻的**文本推理 LLM 服务不是同一个模型**，两者互不依赖。

---

## 4. 上板

```bash
# 目录（数据分区顶层通常需 sudo 建目录并 chown）
sudo mkdir -p <DATA_DIR>/sd-cpp/{bin,models,out,logs}
sudo chown -R "$USER" <DATA_DIR>/sd-cpp

# 二进制（Release 下载；与视频线共用同一个 sd-cli）
scp -o Compression=no sd-cli <board>:<DATA_DIR>/sd-cpp/bin/
ssh <board> 'chmod 755 <DATA_DIR>/sd-cpp/bin/sd-cli'

# 三个模型（7.49 GB；板上无 rsync，用 scp -o Compression=no；传完核对 sha256）
scp -o Compression=no z_image_turbo-Q4_K_S.gguf ae.safetensors Qwen3-4B-Q4_K_M.gguf \
    <board>:<DATA_DIR>/sd-cpp/models/
```

**板侧自检**：

```bash
ssh <board> 'cd <DATA_DIR>/sd-cpp
  ldd bin/sd-cli | grep -c "not found"     # 应为 0
  ls -l models/                             # 三个文件都在
  bin/sd-cli --help | head -3               # 有输出即可运行'
```

---

## 5. 跑一张（完整命令）

```bash
cd <DATA_DIR>/sd-cpp
./bin/sd-cli \
  --diffusion-model models/z_image_turbo-Q4_K_S.gguf \
  --vae             models/ae.safetensors \
  --llm             models/Qwen3-4B-Q4_K_M.gguf \
  --cfg-scale 1.0 --steps 8 --diffusion-fa \
  -W 1024 -H 1024 \
  -p "a cozy reading corner by the window, soft afternoon light, plants, warm tones" \
  -o out/first.png
```

**期望日志**（尾部）：

```
[INFO ] image.cpp:1037 - generate_image completed in 51.90s
[INFO ] main.cpp:504  - save result image 0 to 'out/first.png' (success)
```

`out/first.png` 出现（1024² 约 1.7 MB）即成功。512² 只要十几秒，适合先冒烟。

**长任务/批量**记得脱离终端（断线不会被 SIGHUP 杀掉）：

```bash
{ cd <DATA_DIR>/sd-cpp && setsid nohup ./bin/sd-cli ... > logs/run.log 2>&1 < /dev/null & }
```

---

## 6. 参数速查（取自 `sd-cli --help`，按用途分组）

### 6.1 必需

| 参数 | 说明 |
|---|---|
| `--diffusion-model <file>` | DiT 主权重（Z-Image 的 GGUF 或 safetensors） |
| `--vae <file>` | VAE |
| `--llm <file>` | 文本编码器（Z-Image = Qwen3-4B） |
| `-p, --prompt <str>` | 提示词 |
| `-o, --output <path>` | 输出路径；**支持 printf 风格 `%d`**（如 `out/img_%03d.png`） |

### 6.2 尺寸与采样

| 参数 | 默认 | 说明 |
|---|---|---|
| `-W, --width <int>` | 512 | 宽度（像素） |
| `-H, --height <int>` | 512 | 高度 |
| `--steps <int>` | 20 | 采样步数——**Z-Image-Turbo 用 8** |
| `--cfg-scale <float>` | 7.0 | 无分类器引导——**Z-Image-Turbo 用 1.0** |
| `--sampling-method` | auto（本模型实测为 `euler` + `discrete` 调度） | `euler / euler_a / heun / dpm2 / dpm++2s_a / dpm++2m / …` |
| `--flow-shift <float>` | auto | Flow 模型的 shift（SD3.x / WAN 系） |
| `-s, --seed` | 42 | 随机种子；**< 0 = 每次随机** |
| `-n, --negative-prompt <str>` | 空 | 负面提示词 |
| `--rng` | cuda(sd-webui) | `std_default / cuda / cpu`（ComfyUI 口径是 `cpu`） |

### 6.3 输出与批量

| 参数 | 说明 |
|---|---|
| `-b, --batch-count <int>` | 一批出几张 |
| `--output-begin-idx <int>` | 序列输出起始编号（配合 `-o` 里的 `%d`） |
| `--compression-quality <int>` | JPEG/WebP/视频的质量（默认 90） |

### 6.4 显存/性能

| 参数 | 说明 |
|---|---|
| `--diffusion-fa` | **扩散模型走 FlashAttention**（推荐开） |
| `--vae-tiling` | VAE 分块处理，降显存峰值（池紧时开） |
| `--offload-to-cpu` | 权重放普通内存、按需加载进显存（池很紧时的最后手段，会变慢） |
| `--backend <assign>` | 指定模块设备，如 `vae=cpu`、`clip=cpu,vae=cuda0,diffusion=cuda0` |
| `--hires-upscaler <name>` | 放大算法：`Lanczos` / `Nearest` / `Latent` / …（**首字母区分大小写**） |
| `--hires-upscalers-dir <dir>` | 模型类放大器目录 |

### 6.5 运行模式与调试

| 参数 | 说明 |
|---|---|
| `-M, --mode` | `img_gen`（默认，文生图）/ `adetailer` / `vid_gen`（视频）/ `upscale` / `convert` / `metadata` |
| `-v, --verbose` / `--log-level` | 日志详细度（`debug/verbose/info/warn/error`） |
| `--preview <method>` | 采样过程预览：`none/proj/tae/vae`；`--preview-path` 指定落盘路径 |

---

## 7. 实测性能（不同口径，**报数必须带口径**）

### 7.1 池空闲（独占） vs 与常驻 LLM 服务共存

| 场景 | 512² / 8 步 | 1024² / 8 步 |
|---|---|---|
| 池空闲（无其它 GPU 负载） | **11.9 s** | **51.9 s** |
| LLM 服务常驻、空闲 | ≈ 13.5 s | ≈ 57 s |
| LLM 服务**轻载**（并发短请求） | — | ≈ 59.8 s（+4%） |
| LLM 服务**重载**（3 万+ token 长上下文） | ≈ 28 s | ≈ **108–110 s（约 2×）** |

⇒ 两者可共存，但**共享算力/带宽**：同机 LLM 重载时出图慢一倍是必然，不是故障。
对外提供服务时建议把这条写进接口文档，避免调用方误判超时。

### 7.2 量化档与尺寸的交叉对比

| 对比 | 结果 |
|---|---|
| Q4_K_S vs Q8_0（同 1024²） | 57.53 s vs 57.10 s ⇒ **量化档几乎不影响速度**（瓶颈在算力/带宽，不在权重读取） |
| 每步耗时 512² → 1024² | 1.46 s → ≈ 6.6 s/step（面积 4×，耗时略超 4× ⇒ 注意力项有平方级成分） |
| 可直接套用的估算式 | `1024²/8步 ≈ 1.2 + 8×6.7 + 1.5 ≈ 57 s`；`512²/8步 ≈ 0.7 + 8×1.46 + 1.2 ≈ 13.5 s` |

### 7.3 加载与内存

| 项 | 数值 |
|---|---|
| 装载（权重） | `total params memory size = 6980 MB`（text_encoders 2375 + diffusion 4444 + vae 160，**全在 VRAM**）；热载仅 0.4 s |
| 磁盘读取 | ≈ 1.0 GB/s（首次加载 DiT 4.4 GB 约数秒） |
| 池占用峰值 | ≈ **6.82 GiB**（sd.cpp 报 `6 980 MB`；Q8 档更高） |

---

## 8. 批量出图（可照抄）

### 8.1 一次多张（同一提示词不同 seed）

```bash
./bin/sd-cli --diffusion-model models/z_image_turbo-Q4_K_S.gguf \
  --vae models/ae.safetensors --llm models/Qwen3-4B-Q4_K_M.gguf \
  --cfg-scale 1.0 --steps 8 --diffusion-fa -W 1024 -H 1024 \
  -p "your prompt" -b 4 -s -1 \
  -o out/batch_%03d.png
```
（`-b 4` 一批 4 张，`-s -1` 用随机种子，`%03d` 自动编号）

### 8.2 多提示词批量（脚本模板）

```bash
#!/bin/bash
# batch_gen.sh — 逐行读提示词文件，串行出图（单 GPU 建议串行）
set -u
cd <DATA_DIR>/sd-cpp
SD=./bin/sd-cli
COMMON=(--diffusion-model models/z_image_turbo-Q4_K_S.gguf
        --vae models/ae.safetensors --llm models/Qwen3-4B-Q4_K_M.gguf
        --cfg-scale 1.0 --steps 8 --diffusion-fa -W 1024 -H 1024)
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i+1))
  T0=$SECONDS
  $SD "${COMMON[@]}" -p "$line" -s -1 -o "out/gen_$(printf %03d $i).png" \
      >> logs/batch.log 2>&1
  echo "[$i] rc=$? wall=$((SECONDS-T0))s  $line"
done < prompts.txt
```

要点：**串行**（一块 GPU，并行只会互相抢）；每次记录耗时便于回归；产物用 `%03d` 编号。

---

## 9. 常见需求配方

| 需求 | 做法 |
|---|---|
| 快速冒烟（十几秒） | `-W 512 -H 512 --steps 8 --cfg-scale 1.0 --diffusion-fa` |
| 常规出图（1 分钟） | `-W 1024 -H 1024`，其余同上 |
| 复现同一张 | 固定 `-s <seed>`（默认 42） |
| 每次不同 | `-s -1` |
| 排除不想要的东西 | `-n "blurry, watermark, text"` |
| 池余量偏紧 | 加 `--vae-tiling`；或用 Q4 档、降分辨率 |
| 想更大（1280²/1536²） | 需要先**腾池**（见 §11），或 `--offload-to-cpu`（更慢） |

---

## 10. 排障

| 症状 | 根因 | 处理 |
|---|---|---|
| `cudaMalloc failed: out of memory` | 池余量不足（同机服务占用大） | 先用 Q4 档/512²；或加 `--vae-tiling`；实在不行腾池（见 §11）；**不要清池** |
| `vae decode compute failed` + 随后 `retrying with spatial tiling` 并成功 | 触发了自动分块解码（已贴近上限） | **正常回退**，无需处理；但说明池余量偏紧 |
| hires 放大失败（`vae encode compute failed`） | 放大阶段要额外显存 | 不开 hires；或先腾池（§12） |
| `--hires-upscaler lanczos` 秒退 | **取值区分大小写** | 写 `Lanczos` |
| `no kernel image is available for execution on the device` | 平台不是 `sm_101` | 需自行交叉编译 |
| `ldd` 报 `not found` | CUDA 运行库路径问题 | 正常情况 DriveOS 自带；必要时设置 `LD_LIBRARY_PATH` 指向 CUDA 的 aarch64 运行库目标路径 |
| 建目录 `Permission denied` | 数据分区顶层属主不是你 | `sudo mkdir -p … && sudo chown -R $USER …` |
| `rsync: command not found` | 板上无 rsync | 用 `scp -o Compression=no` |
| 用 `/usr/bin/time` 计时命令秒退 | 板上**没有**该工具 | 用 `$SECONDS` |
| SSH 起任务后命令挂住 | 没脱离终端 | `setsid nohup … < /dev/null &` |
| 出的图是花屏/纯色 | 权重或 VAE 文件损坏/拿错 | 用 §3 的校验值核对文件 |

---

## 11. 调优与限制

| 方向 | 做法 | 注意 |
|---|---|---|
| 提速 | 降分辨率（512² ≈ 1/4 时间）、`--diffusion-fa`、避免与重载 LLM 抢 | 量化档换档**不提速**（实测持平） |
| 省显存 | `--vae-tiling`、换 Q4 档、`--offload-to-cpu`（慢） | 池是 CUDA 的硬边界，**不能靠清池解决** |
| 提画质 | 换 Q8 档（速度几乎不变）、更多步数（Turbo 模型收益有限） | 1024² 已贴近池上限 |
| 更大分辨率 | 需先腾池 / `--offload-to-cpu` | 实测 1280²+ 在"LLM 常驻"下必然失败 |

**关于"腾池"**：本平台的大页池就是 GPU 显存。
**不要**为了腾显存去写 `nr_hugepages=0`（会把 CUDA 可用的池清掉，表现为"假 OOM"）。
正确做法是**停掉占用 GPU 的同机服务**，跑完启回——本仓库脚本
[`../../scripts/sd-pool-gate.sh`](../../scripts/sd-pool-gate.sh)
的 `zimage` 分支已内置池状态打印。

---

## 12. 服务化（可选）

想让局域网里的程序/AI 直接调用（不必登录板子），把出图包成 HTTP 服务：
见 **[http-api-deployment.md](../06-http-api/http-api-deployment.md)**
（完整的单文件实现、systemd 部署、自检与排障；末节讨论"常驻 vs 按需"——**仅作建议**）。

---

## 附：这条线用到的脚本

| 脚本 | 用途 |
|---|---|
| [`../../scripts/sd-pool-gate.sh`](../../scripts/sd-pool-gate.sh) | 池门禁统一入口（`status` / `zimage` / `h3`） |
| [`../../scripts/zimage-api.py`](../../scripts/zimage-api.py) | 按需 HTTP 服务（图片线服务化） |

视频生成（MiniMax-H3，带音轨）见**并列的视频线分支**中的 `minimax-h3-quickstart.md`
（本文档所在分支只聚焦图片线）。
---

[整理者注] 本文由真实部署过程整理；内部主机名、账号、凭据与业务标识已按公开分享规范移除或代称化
（如板端数据分区统一写作 `/brand_data`）。文中的命令与数据均来自实际运行记录。
