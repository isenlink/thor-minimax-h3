# MiniMax-H3 视频 + 音频：下载即用（DRIVE Thor）

> One-liner: Running **MiniMax-H3** on an NVIDIA DRIVE Thor board — it generates a **video
> file with a synchronized, model-generated soundtrack** (not a post-production dub).
> This page is the practical path: what it produces, how much unified memory it needs,
> the **four** model files with checksums, the pool-exclusivity procedure, the exact command,
> the measured numbers, and how to verify the audio track is really there.
>
> 适用平台：DRIVE Thor（Tegra264 / sm_101a / DriveOS 7.0.3 / CUDA 12.8）
> 本仓库版：脱敏整理

---

> **单位口径（本文档统一）**
> - **GiB** 用于池／显存口径（本平台大页池 46 GiB 级；`sd.cpp` 打印的 "MB" 实际是 **MiB**）
> - **GB** 用于磁盘文件口径（1 GB = 10⁹ B）
> - 例：H3 权重磁盘占 **45.0 GB**（= 41.9 GiB），运行时占 VRAM **42.0 GiB**（框架报 `43 017 MB`）

## 0. 先看它产出什么

**一条带声音的视频**：画面与音轨由**同一次联合去噪**产出（音轨不是后期配音、也不是 BGM 叠加）。

实测产物的流信息（`ffprobe` 原样）：

| 档位 | 视频流 | 音频流 |
|---|---|---|
| 864×480 × **56 帧**（2.33 s @24fps） | h264，2.333 s | **aac，32 000 Hz，2 声道**，2.325 s |
| 864×480 × **124 帧**（5.17 s @24fps） | h264，5.167 s | **aac，32 000 Hz，2 声道**，5.175 s |
| （`sd-cli` 直出格式 `.avi`） | mjpeg | **pcm_s16le，32 000 Hz，2 声道** |

音画时长严格对齐 ⇒ 同一次生成。**两条示例视频就在本目录 [`samples/`](../../samples/) 下**，可直接看/听：

- [`samples/h3_864x480_56f_2s33.mp4`](../../samples/)（1.9 MB）
- [`samples/h3_864x480_124f_5s17.mp4`](../../samples/)（3.7 MB）

> ⚠️ **关键参数**：必须传 `--audio-vae <音频VAE>`，否则只有画面没有声音。
> 上游文档原文：*"Omitting `--audio-vae` still runs the joint diffusion model but produces
> video without a decoded audio track."*

---

## 1. 硬前提（不满足就别开始）

| 项 | 要求 | 说明 |
|---|---|---|
| 平台 | **sm_101**（DRIVE Thor） | 其它算力需自行交叉编译 |
| GPU 内存 | 本机"大页池"≈ **46 GiB**，H3 权重运行时占 **42.0 GiB** | ⇒ **必须独占池**：跑前停掉同机占 GPU 的服务，跑完启回 |
| 磁盘 | 权重四件套共 **45.0 GB**（41.9 GiB） | 放到板上的**数据分区**（可写目录通常只有一两个，见 §3） |
| 二进制 | 本仓库 Release 的 `sd-cli`（**与图片生成共用同一个二进制**） | 无需额外编译 |

**池 = GPU 显存**（本平台统一内存架构）：查池

```bash
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages     # 总页数（每页 2 MiB）
cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages  # 空闲页数
sudo cat /sys/kernel/debug/nvmap/iovmm/clients                 # 谁在占 GPU 内存、占多少
```

⚠️ **绝对不要为了"腾显存"去写 `nr_hugepages=0`** —— 那会把 CUDA 能用的池清掉，表现为"假 OOM"。
让池的正确方式是**停掉占用方**（见 §4）。

---

## 2. 四个模型文件（缺一不可）

| 角色 | 命令行参数 | 我们实测用的文件 | 体积 | 上游指定来源 |
|---|---|---|---|---|
| DiT（视频+音频联合去噪） | `--diffusion-model` | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 B | safetensors：`Comfy-Org/MiniMax-H3` → `diffusion_models`；GGUF：`leejet/MiniMax-H3-GGUF` |
| 视频 VAE | `--vae` | `minimax_h3_video_vae_fp16.safetensors` | `5,207,808,496` | `Comfy-Org/MiniMax-H3` → `vae` |
| **音频 VAE** | `--audio-vae` | `minimax_h3_audio_vae_fp32.safetensors` | `605,254,808` | `Comfy-Org/MiniMax-H3` → `vae` |
| 文本编码器 | `--llm` | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | `18,218,065,024` | GGUF：`leejet/MiniMax-H3-GGUF`；safetensors：`Comfy-Org/MiniMax-H3` → `text_encoders` |

**实测通过的一组校验值**（与上面这组文件逐字节对应，供你核对下载件）：

| 文件 | 字节数 | sha256 |
|---|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | 20,970,379,616 | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | `18,218,065,024` | `11e6efe70a57ce7f4838c47bdbd1a1c4b8ce10e2b7747f1b065990b70f4b05fc` |
| `minimax_h3_video_vae_fp16.safetensors` | `5,207,808,496` | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| `minimax_h3_audio_vae_fp32.safetensors` | `605,254,808` | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| **合计** | **45,001,507,944**（= 45.0 GB = 41.9 GiB） | — |

> ⚠️ **文本编码器必须是 MiniMax-H3 专用变体**（上游明确要求）：Qwen3-VL-32B 截断到
> **50 层语言层**、**不带最终 LM norm**，视觉塔（含 3 个 DeepStack merger）必须齐全。
> 实测张量结构为 **902 个张量 = model 551 + visual 351**（层 0–49）。
> 拿错版本会加载失败——**先按"GGUF 头部比对"确认再传输**（方法见
> [prebuilt-binary-quickstart.md](../05-prebuilt-binary/prebuilt-binary-quickstart.md) §3.4）。

> 说明：DiT 是 **fl2va（首尾帧）变体**的 pruned int8（convrot）档，实测 932 个张量、
> 含 `adaln_t_table`（AdaLN curve-table 变体）。若你从上游取，选 `fl2va` 的
> safetensors 或 GGUF 均可；**认文件名 + 校验值**最省事。

---

## 3. 上板

```bash
# 目录（数据分区顶层通常需 sudo 建目录并 chown；板上可写的大分区一般只有一个）
sudo mkdir -p <DATA_DIR>/sd-cpp/{bin,models,out,logs}
sudo chown -R "$USER" <DATA_DIR>/sd-cpp

# 二进制（从本仓库 Release 取；与图片生成共用同一个 sd-cli）
scp -o Compression=no sd-cli <board>:<DATA_DIR>/sd-cpp/bin/
ssh <board> 'chmod 755 <DATA_DIR>/sd-cpp/bin/sd-cli && <DATA_DIR>/sd-cpp/bin/sd-cli --help | head -3'

# 四个权重（45.0 GB；板上无 rsync，用 scp；传完逐个核对 sha256）
scp -o Compression=no <四个文件> <board>:<DATA_DIR>/sd-cpp/models/
```

**板侧自检**（跑之前先确认三件事）：

```bash
ssh <board> 'cd <DATA_DIR>/sd-cpp
  ldd bin/sd-cli | grep -c "not found"        # 缺失依赖应为 0
  ls -l models/                                # 四个文件都在
  echo "池余: $(cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages)"'
```

---

## 4. 独占池：跑 H3 的标准动作

H3 运行时占 VRAM **42.0 GiB**，加上计算缓冲会顶到 46 GiB 池的天花板 ⇒ **必须把池整块让出来**。

```bash
# ① 谁在占？
sudo cat /sys/kernel/debug/nvmap/iovmm/clients       # CLIENT/PROCESS/PID/SIZE
systemctl list-units --type=service --state=running | grep -iE 'llama|inference'

# ② 停掉占用方（例：常驻 LLM 服务）
sudo systemctl stop <llm-service>
#    等 nvmap 占用归零才算真释放（实测 20 秒内），别只看 systemctl 返回

# ③ 跑 H3（见 §5）

# ④ 跑完启回
sudo systemctl start <llm-service>   # 实测 20 秒内 READY
```

**一劳永逸的做法**：用门禁脚本包起来——检测 GPU 被占 → 自动停服务 → 等池释放 → 跑 → 自动启回。
本仓库脚本：[`../../scripts/sd-pool-gate.sh`](../../scripts/sd-pool-gate.sh)

```bash
# 用法
sd-pool-gate.sh status                                    # 看池/GPU 占用/温度
PROMPT="你的提示词" sd-pool-gate.sh h3 <DiT> <TE> <W> <H> <帧数> <步数> <输出名>
# 注：提示词走 PROMPT 环境变量（脚本内部读该变量）；第 8 个及之后的参数会原样追加给 sd-cli
```

---

## 5. 跑一次（完整命令）

```bash
cd <DATA_DIR>/sd-cpp
# 长任务务必脱离终端（断线不会杀掉任务）
{ setsid nohup ./bin/sd-cli -M vid_gen \
    --diffusion-model models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
    --vae            models/minimax_h3_video_vae_fp16.safetensors \
    --audio-vae      models/minimax_h3_audio_vae_fp32.safetensors \
    --llm            models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
    -p "A cute cat surfing on a tropical ocean wave, riding a white surfboard, cinematic tracking shot, realistic water, bright sunlight" \
    --cfg-scale 1.0 -W 864 -H 480 --video-frames 56 --steps 8 \   # 8 步为实测的速度/质量平衡点（上游示例未指定，默认 20）
    --diffusion-fa --fps 24 \
    -o out/h3_my_first \
  > logs/h3_run.log 2>&1 < /dev/null & }
tail -f logs/h3_run.log
```

**参数要点**：

| 参数 | 说明 |
|---|---|
| `-M vid_gen` | 必需：视频生成模式（**不是**文生图模式） |
| `--vae` / `--audio-vae` | 视频 VAE 与**音频 VAE**（漏掉 `--audio-vae` = 无声视频）。注：`--help` 里该参数的文案写的是 "LTX audio vae"，但源码同时支持 H3 的音频 VAE（`src/model/vae/minimax_h3_audio_vae.hpp`），属上游 help 文案滞后 |
| `--video-frames` | 帧数（24 fps；56 帧≈2.33 s、124 帧≈5.17 s）。⚠️ 帧数按 **`17k+5` 向上对齐**——传 60 会静默变成 73；可用值：**5/22/39/56/73/90/107/124** |
| `--fps 24` | 输出帧率 |
| `--cfg-scale 1.0` | 本模型实测用 1.0 |
| `--diffusion-fa` | 走 FlashAttention 路径 |

---

## 6. 验证"真的有声音"（收工前必做）

```bash
ffprobe -v error -show_entries stream=index,codec_type,codec_name,channels,sample_rate,duration \
        -of default=noprint_wrappers=1 out/h3_my_first.avi    # sd-cli 直出为 avi(+pcm)
```

**期望**：输出里同时有 `codec_type=video` 与 `codec_type=audio`。
若只有 video ⇒ 检查命令行是否漏了 `--audio-vae`。

转成便于分享的 mp4（浏览器/手机直接播）：

```bash
ffmpeg -i out/h3_my_first.avi -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k \
       -movflags +faststart out/h3_my_first.mp4
```

---

## 7. 实测性能（2026-09-16，864×480，8 步，`--cfg-scale 1.0 --diffusion-fa`）

| 档位 | 总耗时 | 文本条件 | 采样 | 视频 VAE | 音频 VAE | 峰值池占用 |
|---|---|---|---|---|---|---|
| 256×256×5 帧（2 步，最小 POC） | **29 s** | 10.3 s | 13.3 s | 4.4 s | 0.7 s | — |
| 864×480×**56 帧**（2.33 s 片） | **314 s**（≈5.2 min） | 10.0 s | 240.8 s | 56.7 s | 4.1 s | 41.8 / 46 GiB |
| 864×480×**124 帧**（5.17 s 片） | **953 s**（≈15.9 min） | 10.1 s | 802.1 s | 129.6 s | 8.4 s | **42.8 / 46 GiB** |

- 权重内存（框架实报）：**43 017 MB** = 文本编码器 17 376 + DiT 20 083 + VAE 5 558（**全部在 VRAM**）
- 载入：DiT 冷载 ≈11 s、TE ≈11 s（热载 <1 s）；磁盘读 ≈1.0 GB/s
- 有效算力反推：int8 GEMM ≈ **7.6 TFLOPS**（同平台 bf16 峰值实测 ≈60 TFLOPS）——瓶颈在这
- **性价比档位：56 帧**（5.2 分钟换 2.33 秒成片）；124 帧耗时是 3 倍、只换到 2.2 倍时长
- 温度：全程 48–54 °C

---

## 8. 排障

| 症状 | 根因 | 处理 |
|---|---|---|
| 输出文件**没有声音** | 漏了 `--audio-vae` | 加上音频 VAE 重跑（§5） |
| `cudaMalloc failed: out of memory` | 池没让出来（或有别的进程在占） | 按 §4 停占用方并等 `nvmap` 归零；**不要清池** |
| 加载报架构/张量不符 | 文本编码器不是 H3 变体，或 DiT 拿错变体 | 按 §2 的 902 张量/校验值核对 |
| `no kernel image is available...` | 平台不是 `sm_101` | 需自行交叉编译 |
| 想开更长时长/更高分辨率 | 池余量不足（峰值已 42.8/46 GiB） | 用更小的 DiT 量化档（如 Q4 GGUF ≈10.6 GB）腾出余量；或缩短帧数 |
| SSH 起任务后命令挂住不返回 | 没脱离终端 | `setsid nohup ... < /dev/null &` |
| 用 `/usr/bin/time` 计时命令秒退 | 板上没有该工具 | 用 `$SECONDS` |

---

## 9. 还能调什么

| 方向 | 做法 | 注意 |
|---|---|---|
| 更省内存 | DiT 换 GGUF 量化档（Q4_K_M ≈10.6 GB） | 总量可压到 ≈33 GB，为长时长留余量；画质/速度需自行 A/B |
| 更长时长 | 提高 `--video-frames`（24 fps） | 耗时大致线性增长；峰值池占用也会涨 |
| 首帧/尾帧条件（I2VA / FL2VA） | 加 `--init-img`（和 `--end-img`） | 本仓库用的 DiT 即 fl2va 变体 |
| 音轨单独处理 | `sd-cli` 直出 avi 里是 PCM 音轨，可按 §6 转码 | — |

---

## 附：这条线用到的脚本

| 脚本 | 用途 |
|---|---|
| [`../../scripts/sd-pool-gate.sh`](../../scripts/sd-pool-gate.sh) | 池门禁统一入口：`status` / `zimage` / `h3`（h3 自动停-等池-跑-恢复） |

图片生成（Z-Image）另见 [prebuilt-binary-quickstart.md](../05-prebuilt-binary/prebuilt-binary-quickstart.md)；
把出图做成 HTTP 服务见 [http-api-deployment.md](../06-http-api/http-api-deployment.md)。
---

[整理者注] 本文由真实部署过程整理；内部主机名、账号、凭据与业务标识已按公开分享规范移除或代称化
（如板端数据分区统一写作 `/brand_data`）。文中的命令与数据均来自实际运行记录。
