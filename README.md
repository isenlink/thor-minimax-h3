# Stable Diffusion on NVIDIA DRIVE Thor / 车规域控上跑扩散模型（视频+音频 / 图片 两条线）

> 民间实测记录：在 NVIDIA DRIVE Thor 域控板（p3960-0010 / Tegra264 / sm_101a, DriveOS 7.0.3, CUDA 12.8）上，
> 用 `stable-diffusion.cpp` 跑通**两条并列生成线** —— 一个交叉编译好的 aarch64 二进制（`sd-cli`），
> **无需 PyTorch / ComfyUI / docker**：
>
> - **视频线（MiniMax-H3）**：联合视频+音频生成，43 GB 权重塞进 46 GiB 统一内存池，**必须独占池**，
>   56/124 帧与三档分辨率实测性能全部真机测量；
> - **图片线（Z-Image-Turbo）**：文生图，权重 ≈7 GB，**可与同机常驻 LLM 服务共存**，512²/1024² 实测。
>
> 两条线共用同一个二进制，差别只在资源画像（独占池 vs 可共存）。
>
> Grassroots field notes: running **two parallel generation lines** on an NVIDIA DRIVE Thor automotive
> domain controller via `stable-diffusion.cpp` — one prebuilt aarch64 binary, no PyTorch/ComfyUI:
> **MiniMax-H3** (joint video+audio, 43 GB weights, pool-exclusive) and **Z-Image-Turbo** (text-to-image,
> ~7 GB weights, coexists with a resident LLM). Full measured performance on real hardware.

**语言 / Language**：中文（英文版规划中）

## TL;DR

| 项 | 结果 |
|---|---|
| 框架 | `stable-diffusion.cpp`（`-M vid_gen`），无需 PyTorch / ComfyUI |
| 权重总量 | **43 017 MB** = 文本编码器 17 376 + DiT 20 083 + VAE 5 558（全在池内） |
| 864×480×56 帧（2.33 s 片，8 步） | **314 s**（≈5.2 分钟） |
| 864×480×124 帧（5.17 s 片，8 步） | **953 s**（≈15.9 分钟） |
| **1344×768×124 帧**（模型原生档，需 `te=disk`） | **4228 s**（≈70 min）—— 见 [§04](docs/04-resolution-vram/resolution-vram-measured.md) |
| 峰值池占用 | **42.8 / 46 GiB** —— 必须独占池 |
| 有效算力（反推） | int8 GEMM ≈ **7.6 TFLOPS**（同平台 bf16 峰值 ≈60） |
| 温度 | 全程 48–54 °C（被动散热，无压力） |
| 音轨 | 与视频同步产出，无需二次配音 |

**一句话**：能跑，但它是"**独占池 + 分钟级**"的负载 —— 适合后台批量出片，不适合交互式秒回。

## 文档导航

> 两条线共用同一个二进制（`sd-cli`，见 [Releases](https://github.com/isenlink/thor-minimax-h3/releases)）。
> 先读 [05 预编译二进制](docs/05-prebuilt-binary/prebuilt-binary-quickstart.md) 下载上板，再按你要的线走对应 quickstart。

| 文档 | 内容 |
|---|---|
| **视频线（MiniMax-H3，带音轨）** | |
| [docs/02-h3-video-audio/minimax-h3-quickstart.md](docs/02-h3-video-audio/minimax-h3-quickstart.md) | **视频线下载即用**：四件套权重来源与校验值、**独占池**流程、带 `--audio-vae` 的完整命令、产出验声、实测、排障 |
| [docs/02-h3-video-audio/h3-video-audio.md](docs/02-h3-video-audio/h3-video-audio.md) | 视频线深入篇：内存预算、权重选型与两个格式陷阱、GGUF 头部比对法、池门禁脚本、实测性能、踩坑清单、复现清单 |
| [docs/04-resolution-vram/resolution-vram-measured.md](docs/04-resolution-vram/resolution-vram-measured.md) | 分辨率上限与显存腾挪实测：`te=disk` 腾出 15.88 GiB、三档分辨率 × 耗时 × 显存、1344×768 跑通、提示词配方、交付前自检 |
| **图片线（Z-Image-Turbo，文生图）** | |
| [docs/07-z-image/z-image-quickstart.md](docs/07-z-image/z-image-quickstart.md) | **图片线下载即用**：三件套权重与校验值、**不需独占池**（可与常驻 LLM 共存）、完整参数速查、批量脚本、多档位实测、排障、调优 |
| **通用（两条线共用）** | |
| [docs/05-prebuilt-binary/prebuilt-binary-quickstart.md](docs/05-prebuilt-binary/prebuilt-binary-quickstart.md) | 预编译二进制下载即用：平台判断、下载校验、上板 5 步、模型获取与 GGUF 头部比对、常见错误对照、构建来源 |
| [docs/06-http-api/http-api-deployment.md](docs/06-http-api/http-api-deployment.md) | 把出图部署成 HTTP 服务（给别的程序/Agent 调用）：前置检查 → 上传 → systemd → 自检 → 排障 + 单文件实现 |
| [docs/03-troubleshooting/troubleshooting.md](docs/03-troubleshooting/troubleshooting.md) | 按症状索引的踩坑速查表（与主文档 §五 同步） |
| [scripts/sd-pool-gate.sh](scripts/sd-pool-gate.sh) | 池门禁入口脚本：检测 GPU 占用 → 自动让池 → 跑完自动恢复（`status` / `zimage` / `h3`） |
| [scripts/zimage-api.py](scripts/zimage-api.py) + [zimage-api.service](scripts/zimage-api.service) | 按需 HTTP 出图服务（OpenAI/A1111 兼容、异步 job、产物直链、空闲零显存）+ systemd 单元 |
| [samples/](samples/README.md) | 两条线实测产物：视频线三档分辨率 + 56/124 帧成片（带音轨）、图片线 512²/1024² 出图 |

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
