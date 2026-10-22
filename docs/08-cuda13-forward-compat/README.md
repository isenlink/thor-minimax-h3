# CUDA 13 前向兼容线（ComfyUI cu130）—— H3 提速 3.63×

> 本文档给 **MiniMax-H3** 的 **CUDA 13 前向兼容**路线（ComfyUI + torch 2.13 + comfy_kitchen），
> 与仓库原有的 `stable-diffusion.cpp`（CUDA 12.8，无 PyTorch）是**两条并列技术路线**。
> 本线**没有自编译二进制**——全部是官方 wheel + NVIDIA 官方 compat 包，故本文档只提供
> 「官方下载地址 + SHA256 + 版本锁 + 安装顺序」的**复现清单**，不提供 ModelScope 二进制。

> **方法来源致谢**：CUDA 13 forward-compat 在 Thor 上的方法来自网友
> [souyou233 (EnkiMa)](https://github.com/souyou233) 的公开工作
> [Minimax-H3-on-thor-u](https://github.com/souyou233/Minimax-H3-on-thor-u)。
> 共用底座（compat + runtime）见 [thor-fp8-llm docs/11](https://github.com/isenlink/thor-fp8-llm/tree/main/docs/11-cuda132-shared-base)。

---

## 0. 结论（TL;DR）

| 项 | 结果 |
|---|---|
| 路线 | ComfyUI 0.38 + torch 2.13.0+cu130 + comfy_kitchen 0.2.39，CUDA 13 前向兼容 |
| 编译件 | **无自编译二进制**（纯官方 wheel + NVIDIA 官方 `cuda-compat-orin-13-2` deb） |
| 系统 CUDA | 仍 12.8；仅在测试进程内隔离加载 `cuda-compat-orin-13-2`，不改系统默认 |
| 实测提速 | 8 步生产 LoRA 同口径 **289.4 s/段** vs sd.cpp 1049 s/段 = **3.63×** |
| 双帧链 | 347–348 s/段，接缝像素差 2.3–2.9；20 段整批 **19/20** 一次过门禁 |
| 状态 | **已成功复现** |

---

## 1. 方案本质（为什么 Thor 能用上 CUDA 13）

- 不是刷系统、不是换内核驱动。**CUDA 13 前向兼容**：内核驱动保持 CUDA-12 时代原样，
  用户态放一份 CUDA 13 的 `libcuda.so.1` + PTX JIT 伴生库（`cuda-compat-orin-13-2`），
  `LD_LIBRARY_PATH` 里 `/compat` 排最前、系统 `libnvrm_*` 殿后。
- 生效判据（日志）：`cuDriverGetVersion = 13020` + `comfy_kitchen backend cuda: available=True` +
  `Native ops: … int8_tensorwise … float8_e4m3fn …`（原生 int8/fp8 生效，无 eager 反量化）。
- 前提：宿主内核驱动 ≥ CUDA-13 前向兼容下限（Thor 同源板实测满足）。

---

## 2. 软件栈清单（官方件，含 SHA256）

> 全部为官方发行件，**无我方编译产物**。下载后先核对 SHA256 再使用。

| # | 件 | 大小 | sha256 | 来源 |
|---|---|---|---|---|
| 1 | `torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_aarch64.whl` | 427,082,459 B | `5ebd552c887e707c8e64927aceb8377ca2e81588c4e7494bcd23cb8ac0aca14d` | PyTorch 官方，`pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130` |
| 2 | `comfy_kitchen-0.2.39-cp312-abi3-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl` | 37,219,419 B | `6f76b1291e4d5dc8c5afc0c2b31e41545470e167e6ac90330360a1818cac3647` | PyPI 官方，`pip install comfy_kitchen==0.2.39` |
| 3 | `cuda-compat-orin-13-2_13.2.44290101-1_arm64.deb` | 82,021,804 B | `3331f3ac94eecc0e3c0d1976c8fa306c2bf33dd6f771e0d61bb583cab3590037` | NVIDIA 官方 CUDA 13.2 forward-compat 包，仅 `dpkg-deb` 解包后进程内隔离加载，未重编译 |
| 4 | `ComfyUI 0.38` | Python 源码 | — | [Comfy-Org/ComfyUI](https://github.com/Comfy-Org/ComfyUI) @ 0.38 |

- `compat-orin-13.2.tar.gz` = 上表 #3 官方 deb 的解包归档；`bench-runtime-wheels.tar` = #1/#2 官方 wheel 的集合。
  二者都不是自编译产物，只是官方件的封装。

---

## 3. 安装顺序（版本锁）

```bash
# 1) Python 3.12 环境（cp312）
# 2) torch（cu130 官方 wheel）
pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
# 3) comfy_kitchen（PyPI 官方 wheel，aarch64）
pip install comfy_kitchen==0.2.39
# 4) ComfyUI 0.38 源码 + H3 节点
git clone https://github.com/Comfy-Org/ComfyUI  # 切 0.38
# 5) CUDA 13.2 forward-compat：dpkg-deb 解包到独立目录，进程内 LD_LIBRARY_PATH 隔离加载
#    （compat 排最前，宿主 libcuda 绝不 cp 进 compat）
```

> ⚠️ **compat 顺序铁律**：宿主 `libcuda` 抢先 = 悄悄退回 CUDA 12.8；宿主 `libcuda` 绝不复制进 compat 目录。

---

## 4. 实测数据（Thor02:8189，验证口径）

| 配置 | CUDA | 8 步生产 LoRA 单段 | 相对 sd.cpp |
|---|---|---|---|
| sd.cpp（对照，同日） | 12.8 | 1049 s/段 | 1.00× |
| ComfyUI cu130 | 13 前向兼容 | **289.4 s/段**（288.8–289.8） | **3.63×** |

- 双帧链 347–348 s/段，接缝像素差 2.3–2.9
- 20 段整批 **19/20** 一次过门禁
- 证据：`RESULT-20261021.md` / `ROUND-SUMMARY-20261022.md`（内部验证留痕，未对外发布）

---

## 5. 状态与口径

- **已成功复现**。
- 本线**没有单独的对外发布仓**——无 GitHub 独立仓、无 ModelScope 一键包。
  对外入口只有本文档 + 共用底座（见 §0 致谢）。
- 权重来源：HF 官方（Comfy-Org/MiniMax-H3 系列），不随本文档分发。

---

## 6. 与仓库原有 sd.cpp 线的关系

| | sd.cpp 线（本仓库主文档） | CUDA 13 前向兼容线（本文档） |
|---|---|---|
| 框架 | `stable-diffusion.cpp`（自编译二进制） | ComfyUI + torch + comfy_kitchen |
| CUDA | 12.8 | 13 前向兼容（系统仍 12.8） |
| 编译件 | 有（`sd-cli`/`sd-server`） | **无（纯官方件）** |
| 单段耗时（8 步 LoRA） | 1049 s/段 | 289.4 s/段（**3.63×**） |
| 状态 | 生产口径 | 已成功复现 |
