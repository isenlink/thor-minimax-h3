# 实测样例（MiniMax-H3 视频 + 音频）

> 三档分辨率的**真实机实测成片**，同一块 DRIVE Thor 板、同一组权重、同一提示词连续跑出。
> 每档含：交付版 `.mp4`（h264 + aac 立体声，**带音轨**）+ 三帧抽检拼图 `.jpg`。
> 数据与耗时拆解见 [docs/04-resolution-vram/resolution-vram-measured.md](../docs/04-resolution-vram/resolution-vram-measured.md)。

## 文件清单

| 文件 | 分辨率 | 帧数 / 时长 | 实测耗时 | 说明 |
|---|---|---|---|---|
| `h3_864x480_person.mp4` | 864×480 | 124 帧 / 5.17 s | **931 s** | 最低档，性价比参考 |
| `h3_1024x576_person.mp4` | 1024×576 | 124 帧 / 5.17 s | **1652 s** | 中档 |
| `h3_768p_person.mp4` | **1344×768**（模型原生档） | 124 帧 / 5.17 s | **4228 s** | 最高档，需 `te=disk` 参数才跑通 |
| `frames-864x480.jpg` / `frames-1024x576.jpg` / `frames-768p.jpg` | — | — | — | 各档 0.4 / 2.6 / 4.8 s 三帧拼图（人物出镜实证） |

## 提示词（三档共用）

```
A young woman with long dark hair stands on a sunny beach, facing the camera and smiling.
She speaks clearly and warmly: "The water here feels so soft on my skin."
Soft acoustic guitar music plays quietly in the background, with gentle ocean waves.
```

三个要点（缺一项则不达标）：
1. **人物全程出镜**：`stands …, facing the camera`（正面 + 静态机位）；
2. **让人说话**：对白用**双引号直接写进提示词** ⇒ 模型按字面生成对应语音；
3. **轻背景乐**：必须写 `music plays **quietly** in the background`。

## 容器口径（ffprobe 实测）

- 视频：h264，5.167 s；音频：aac 32 kHz 立体声，5.175 s —— **音画时长严格对齐**；
- 对白经 ASR 转写与提示词**逐字一致**（"带人声"是实证，不是按提示词推断）；
- 容器元数据只含提示词、采样参数（8 步 / cfg 1.0 / seed 42 / 尺寸）、模型名与上游 commit，
  **不含**绝对路径、主机名、账号或内网地址。

## 说明

- 成片为 AI 生成（虚构人物），无真人肖像；画面已逐帧抽样确认无文字 / 水印 / 品牌标识。
- 最高档（1344×768）单条 4228 s（≈70 min），是"独占池 + 分钟级"负载的典型，适合后台批量出片。
