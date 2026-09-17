# 实测样例（两条生成线，同一块 DRIVE Thor 板）

> 本目录是**同一项目两条并列生成线**的真实机实测产物，全部由本仓库 Release 的 `sd-cli`
> 二进制跑出（sm_101，无需 PyTorch / ComfyUI / docker）。
>
> - **视频线（MiniMax-H3）**：带模型生成音轨的视频（h264 + aac 立体声，音轨与画面同一次联合去噪产出，非后期配音）。
> - **图片线（Z-Image-Turbo）**：文生图 PNG（命令行直出，无后期处理）。
>
> 两条线共用同一个二进制，差别在资源画像：图片线权重 ≈7 GB（可与常驻 LLM 共存），
> 视频线权重 45.0 GB（运行时占 VRAM 42.0 GiB，**必须独占** 46 GiB 级大页池）。

## 视频线（MiniMax-H3）

### 分辨率 × 显存腾挪实测（三档，同一提示词连续跑出）

| 文件 | 分辨率 | 帧数 / 时长 | 实测耗时 | 说明 |
|---|---|---|---|---|
| `h3_864x480_person.mp4` | 864×480 | 124 帧 / 5.17 s | **931 s** | 最低档，性价比参考 |
| `h3_1024x576_person.mp4` | 1024×576 | 124 帧 / 5.17 s | **1652 s** | 中档 |
| `h3_768p_person.mp4` | **1344×768**（模型原生档） | 124 帧 / 5.17 s | **4228 s** | 最高档，需 `te=disk` 参数才跑通 |
| `frames-864x480.jpg` / `frames-1024x576.jpg` / `frames-768p.jpg` | — | — | — | 各档 0.4 / 2.6 / 4.8 s 三帧拼图（人物出镜实证） |

数据与耗时拆解见 [docs/04-resolution-vram/resolution-vram-measured.md](../docs/04-resolution-vram/resolution-vram-measured.md)。

**提示词（三档共用）**：

```
A young woman with long dark hair stands on a sunny beach, facing the camera and smiling.
She speaks clearly and warmly: "The water here feels so soft on my skin."
Soft acoustic guitar music plays quietly in the background, with gentle ocean waves.
```

三个要点（缺一项则不达标）：
1. **人物全程出镜**：`stands …, facing the camera`（正面 + 静态机位）；
2. **让人说话**：对白用**双引号直接写进提示词** ⇒ 模型按字面生成对应语音；
3. **轻背景乐**：必须写 `music plays **quietly** in the background`。

### 帧数档实测（两档，同一提示词）

| 文件 | 时长 | 规格 | 实测耗时 | 实际画面 |
|---|---|---|---|---|
| `h3_864x480_56f_2s33.mp4` | 2.33 s | 864×480 @ 24 fps，56 帧 | **314 s** | 一只猫在冲浪板上冲浪 |
| `h3_864x480_124f_5s17.mp4` | 5.17 s | 864×480 @ 24 fps，124 帧 | **953 s** | 一名人类冲浪者在冲浪 |

提示词（两段相同）：`A cute cat surfing on a tropical ocean wave, riding a white surfboard, cinematic tracking shot, realistic water, bright sunlight`

> ⚠️ **同一提示词、不同时长档，产出的主体不同**（56 帧档是猫、124 帧档是人）——这是长时长档下的
> **模型漂移**，不是剪辑或换提示词造成的。看效果/做预览时请以实际画面为准。

**音轨信息（ffprobe 原样）**：视频流 h264；音频流 **aac，32 000 Hz，2 声道**，与画面时长严格对齐
（2.325 vs 2.333 s、5.175 vs 5.167 s）。音轨为模型生成的**器乐/音效**，已用 ASR 转写确认**无人声（0 段）**。

## 图片线（Z-Image-Turbo）

| 文件 | 规格 | 生成参数 | 实测耗时（池空闲） |
|---|---|---|---|
| `z_image_512x512.png`（421 KB） | 512×512 | 8 步 / `--cfg-scale 1.0` / `--diffusion-fa` / Q4_K_S 档 | **11.94 s** |
| `z_image_1024x1024.png`（1.7 MB） | 1024×1024 | 同上 | **51.90 s** |

**提示词**：
- 512×512：`a red apple on a wooden table, studio light`
- 1024×1024：`a cozy reading corner by the window, soft afternoon light, plants, warm tones`

权重：`z_image_turbo-Q4_K_S.gguf` + `ae.safetensors` + `Qwen3-4B-Q4_K_M.gguf`（校验值见 [docs/07-z-image/z-image-quickstart.md](../docs/07-z-image/z-image-quickstart.md) §3）。

## 容器口径（ffprobe 实测）

- 视频：h264；音频：aac 32 kHz 立体声 —— **音画时长严格对齐**；
- 对白经 ASR 转写与提示词**逐字一致**（"带人声"是实证，不是按提示词推断）；
- 容器元数据只含提示词、采样参数（8 步 / cfg 1.0 / seed 42 / 尺寸）、模型名与上游 commit，
  **不含**绝对路径、主机名、账号或内网地址。

## 说明

- 成片为 AI 生成（虚构人物/画面），无真人肖像；画面已逐帧抽样确认无文字 / 水印 / 品牌标识。
- 最高档（1344×768）单条 4228 s（≈70 min），是"独占池 + 分钟级"负载的典型，适合后台批量出片。
- 同机若常驻 LLM 推理服务，出图/出片耗时会随其负载变化（图片线空闲 ≈13.5 s / 57 s，重载可达 ≈28 s / 108–110 s）——报数带口径。
