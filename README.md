# Stable Diffusion on NVIDIA DRIVE Thor / 车规域控上跑扩散模型（视频 + 音频双轨生成）

> 民间实测记录：在 NVIDIA DRIVE Thor 域控板（p3960-0010 / Tegra264 / sm_101a, DriveOS 7.0.3, CUDA 12.8）上，
> 用 `stable-diffusion.cpp` 跑通 MiniMax-H3 联合视频+音频生成 —— 一个交叉编译好的 aarch64 二进制，
> **无需 PyTorch / ComfyUI**。43 GB 权重塞进 46 GiB 统一内存池，56/124 帧实测性能全部真机测量。
>
> Grassroots field notes: running MiniMax-H3 (joint video+audio generation) on an NVIDIA DRIVE Thor
> automotive domain controller via `stable-diffusion.cpp` — how a 43 GB weight set fits into a
> 46 GiB unified-memory pool, why the pool must be exclusively owned, the model-format traps
> that cost us a wasted download, and full measured performance on real hardware.

**语言 / Language**：中文（英文版规划中）

## TL;DR

| 项 | 结果 |
|---|---|
| 框架 | `stable-diffusion.cpp`（`-M vid_gen`），无需 PyTorch / ComfyUI |
| 权重总量 | **43 017 MB** = 文本编码器 17 376 + DiT 20 083 + VAE 5 558（全在池内） |
| 864×480×56 帧（2.33 s 片，8 步） | **314 s**（≈5.2 分钟） |
| 864×480×124 帧（5.17 s 片，8 步） | **953 s**（≈15.9 分钟） |
| 峰值池占用 | **42.8 / 46 GiB** —— 必须独占池 |
| 有效算力（反推） | int8 GEMM ≈ **7.6 TFLOPS**（同平台 bf16 峰值 ≈60） |
| 温度 | 全程 48–54 °C（被动散热，无压力） |
| 音轨 | 与视频同步产出，无需二次配音 |

**一句话**：能跑，但它是"**独占池 + 分钟级**"的负载 —— 适合后台批量出片，不适合交互式秒回。

## 文档导航

| 文档 | 内容 |
|---|---|
| [docs/02-h3-video-audio/h3-video-audio.md](docs/02-h3-video-audio/h3-video-audio.md) | 主文档：内存预算、权重选型与两个格式陷阱、池门禁脚本、实测性能、踩坑清单、复现清单 |
| [docs/03-troubleshooting/troubleshooting.md](docs/03-troubleshooting/troubleshooting.md) | 按症状索引的踩坑速查表（与主文档 §五 同步） |
| [scripts/sd-pool-gate.sh](scripts/sd-pool-gate.sh) | 池门禁入口脚本：检测 GPU 占用 → 自动让池 → 跑完自动恢复 |

## 核心要点（三分钟版）

1. **统一内存 = 大页池**。CUDA 侧一切占用必须落在 `nr_hugepages × 2 MiB` 的池内（本机 23552 页 = 46 GiB）。
   **永远不要写 `nr_hugepages=0` 去"腾显存"** —— 那会把池清掉，表现为"假 OOM"。
2. **独占池是硬约束**。H3 权重 43 GB 贴着 46 GiB 上限，与同机常驻 LLM 服务（36.7 GiB）互斥。
   门禁脚本自动完成"停服务 → 等 nvmap 归零 → 跑 → 恢复服务"。
3. **权重格式两个陷阱**：
   - ComfyUI 打包的 DiT GGUF 头部 `general.architecture = wan`，`sd.cpp` 无 comfy 兼容分支，加载必败；
   - 同目录的文本编码器 GGUF **可以**直接复用（逐张量比对同构），省一次 17 GiB 下载。
   - **通用手法**：GGUF 头部就是张量清单，HTTP Range 拉前几百 KB 即可完整比对，比下载快 100 倍（见主文档 §2.2）。
4. **瓶颈在算力不在带宽**。int8 GEMM 有效算力 ≈7.6 TFLOPS（bf16 峰值 ≈60），换量化档位不改变量级；
   56 帧档性价比最优（5.2 分钟换 2.33 秒成片）。

## 环境与复现

DRIVE Thor（p3960-0010 / Tegra264 / sm_101a）· DriveOS 7.0.3 · CUDA 12.8 · aarch64 交叉编译 `stable-diffusion.cpp`。
完整复现步骤见主文档 [§六 复现清单](docs/02-h3-video-audio/h3-video-audio.md#六复现清单)。

## License

MIT —— 见 [LICENSE](LICENSE)。
