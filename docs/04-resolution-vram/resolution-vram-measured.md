# MiniMax-H3 视频+音频生成：分辨率上限与显存腾挪实测

> One-liner: Measured resolution / timing / VRAM envelope for **MiniMax-H3 video+audio generation**
> with `stable-diffusion.cpp` on an NVIDIA DRIVE Thor board (aarch64) — including the one flag
> combination that frees **15.9 GiB** of the unified-memory pool by keeping the text encoder on disk.

---

## 一、结论速览（全部为实测，非推算）

| 分辨率 | 帧数 / 时长 | 耗时 | 显存池峰值 | 结果 |
|---|---|---|---|---|
| 864×480 | 124 帧 / 5.17 s | **931 s** | 42.8 GiB | ✅ |
| 1024×576 | 124 帧 / 5.17 s | **1652 s** | 44.10 GiB | ✅ |
| **1344×768（模型原生档）** | 124 帧 / 5.17 s | **4228 s** | **34.44 GiB** | ✅（需下方 §二 的参数） |
| 1280×704 | 124 帧 / 5.17 s | — | — | ❌ 显存分配失败 |
| 1344×768（**不加** §二 参数） | 124 帧 / 5.17 s | — | — | ❌ 显存分配失败（差 63.5 MB）|

- 参考平台：单板统一内存池可用显存 **46 GiB**（可扩到 50 GiB）；权重四件套实占 **42.0 GiB**
  ⇒ 默认配置下"权重常驻"几乎吃光显存，这是上面三处失败的共同原因。
- 帧数受模型约束：按 `17k+5` **向上取整**（传 60 会被静默改成 73）；fps 强制 24。

## 二、关键发现：文本编码器不必常驻显存

H3 的文本编码器（`qwen3vl_32b`，Q4_K_M ≈ **17.4 GB**）**只在开头用一次**（实测编码耗时 **9.57 s**），
之后整段采样再也用不到。默认它却常驻显存，把本该留给采样的空间挤掉。

```bash
--auto-fit off --params-backend te=disk
```

| 项 | 数据 |
|---|---|
| 手段 | `--params-backend te=disk`（文本编码器走磁盘驻留，用时读取），配 `--auto-fit off` 关闭自动规划覆盖 |
| **腾出的显存** | **15.88 GiB**（同配置对照：显存峰值 **41.28 → 25.40 GiB**） |
| 代价 | 文本编码耗时 9.57 s → **9.76 s**（**基本无感**） |
| 收益 | 1344×768 由"必然失败"变为**可跑通**（4228 s） |

> ⚠️ 验证这一类参数**必须看显存池的实际占用**，**不能只看日志里 `total params memory size` 那一行**：
> 该行在 `te=disk` 生效时**仍会照报该模块 `(VRAM)`**，容易误判为"无效"。
> 稳妥做法：同一配置跑两遍（带 / 不带参数），各采样空闲大页数取峰值，差值≈该模块体积即为生效。

**另外两条路（本平台实测不可行，供参考）**：
- `--backend te=cpu`（把编码器计算放到 CPU）：机制有效，但需要 **17.4 GB 系统内存**，
  而池已占用大部分物理内存 ⇒ 进程被 OOM killer 终止。
- `--backend vae=cpu`：机制有效（VAE ≈5.5 GB 可放进系统内存），但 **VAE 解码由 2 分钟劣化到约 50 分钟**，不实用。
- 本平台**没有 swap**；且即便有，swap 面向系统内存、与显存池分配无关，不解决显存不足。

## 三、耗时口径（含各阶段拆解，1344×768 × 124 帧 × 8 步）

| 阶段 | 耗时 |
|---|---|
| 权重加载（分段，逐段 8~10 s） | ≈ 分多段累计 |
| 文本编码（TE 走磁盘） | 9.76 s |
| 采样 8 步 | **≈ 3840 s（每步 479.7 s）** |
| VAE 解码（1344×768） | **357.9 s**（864×480 仅 129.7 s） |
| **合计** | **4227.9 s（≈70.5 min）** |

- 每步耗时随分辨率**超线性**增长（864×480 为 96~100 s/步），主因是图按显存被切成更多段计算。
- 分辨率按像素数线性外推耗时是不准的，**必须实测**。

## 四、提示词配方（"人物出镜 + 说话 + 轻 BGM"，实测有效）

```
A young woman with long dark hair stands on a sunny beach, facing the camera and smiling.
She speaks clearly and warmly: "The water here feels so soft on my skin."
Soft acoustic guitar music plays quietly in the background, with gentle ocean waves.
```

三个要点（缺一项则不达标）：
1. **人物全程出镜**：`stands …, facing the camera`（正面 + 静态机位；帧数取 124 = 5.17 s）
2. **让人说话**：对白用**双引号直接写进提示词** ⇒ 模型按字面生成对应语音
3. **轻背景乐**：必须写 `music plays **quietly** in the background`

实测：音轨与画面**同一次联合生成**（h264 + aac 32 kHz 立体声，音画时长严格对齐）；
对白经 ASR 转写与提示词**逐字一致**。

## 五、复现命令

```bash
./bin/sd-cli -M vid_gen \
  --diffusion-model models/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --vae           models/minimax_h3_video_vae_fp16.safetensors \
  --audio-vae     models/minimax_h3_audio_vae_fp32.safetensors \
  --llm           models/qwen3vl_32b_minimax_h3-Q4_K_M.gguf \
  --auto-fit off --params-backend te=disk \
  -p '<见 §四提示词>' \
  --cfg-scale 1.0 -W 1344 -H 768 --video-frames 124 --steps 8 \
  --diffusion-fa --fps 24 -o out/h3_768p
```

- `--audio-vae` **必传**，否则不出音轨。
- `-o` 不带扩展名时自动补 `.avi`（音轨在 `.avi`/`.webm` 中封装）。

## 六、交付前自检三项（缺一不可）

1. **容器检查**：`ffprobe` 确认视频 ≥5.0 s 且含音频流（本例 h264 5.167 s + aac 32 kHz 立体声 5.175 s）
2. **人声实证**：提取音轨（`ffmpeg -vn -ac 1 -ar 16000`）后用 ASR 转写 —— 有语音段且内容对得上提示词，
   "带人声"才算证据，不能按提示词推断
3. **人物出镜实证**：抽 0.4 / 2.6 / 4.8 s 三帧拼图目视 —— 确认**同一人物全程**在画面、无崩坏/水印

---

*本文数据来自一次连续实测（同一块板、同一组权重、同一提示词），所有耗时均为该次运行实测值。*
