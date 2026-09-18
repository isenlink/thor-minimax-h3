# 预编译二进制：下载即用（DRIVE Thor / stable-diffusion.cpp）

> One-liner: Prebuilt aarch64 binaries of `stable-diffusion.cpp` for NVIDIA DRIVE Thor
> (Tegra264 / sm_101a / DriveOS 7.0.3 / CUDA 12.8). Download → verify → run on the board.
> No cross-compilation, no PyTorch, no ComfyUI, no docker, no on-board compiler.
>
> 这份文档的目标：**让你少折腾**。从下载到出第一张图，照着做约 10 分钟（不含下载时间）。

---

> **单位口径（本文档统一）**
> - **GiB** 用于池／显存口径（本平台大页池 46 GiB 级；`sd.cpp` 打印的 "MB" 实际是 **MiB**）
> - **GB** 用于磁盘文件口径（1 GB = 10⁹ B）
> - 例：H3 权重磁盘占 **45.0 GB**（= 41.9 GiB），运行时占 VRAM **42.0 GiB**（框架报 `43 017 MB`）

## 0. 先花 1 分钟确认平台（避免白折腾）

| 你的平台 | 能否直接用这份二进制 |
|---|---|
| **NVIDIA DRIVE Thor**（Tegra264 / sm_101a / DriveOS 7.0.3 / CUDA 12.8） | ✅ **可以**，实测跑通 |
| DRIVE Thor，但 CUDA 版本不是 12.8 | ⚠️ 需 CUDA 12.8 运行库（DriveOS 自带即符合） |
| Jetson Orin（sm_87）/ Xavier（sm_72）/ 其它 Jetson | ❌ **不可以**——算力版本不同，需自行编译 |
| 桌面 / 服务器 GPU | ❌ 不可以（本包是 aarch64） |
| x86_64 主机 | ❌ 不可以（架构不同） |

**判断方法**（在板上执行）：本包的 CUDA kernel 只编了 `sm_101`。
如果平台不对，运行时会出现：

```
no kernel image is available for execution on the device
```

遇到这条就是算力不匹配 ⇒ 走"自己编译"的路线（见 §8）。

---

## 1. 下载与校验

| 文件 | 体积 | 说明 | sha256 |
|---|---|---|---|
| `sd-cli` | 100,813,608 B（96.1 MiB） | **命令行出图**（推荐；本仓库的按需 HTTP 服务也是调它） | `02cee738e66d4a0d798a7d778d6bbaeea064ce37f06c3f0bf5b536f7addf1f10` |
| `sd-server` | 101,320,080 B（96.6 MiB） | HTTP + 内置 Web UI（常驻方案用；按需方案见 [`scripts/zimage-api.py`](../../scripts/zimage-api.py)） | `f8cd7e0d1f9bfa3b78d144df0a4d1d830ce4be84315f8012a692c3b95863e567` |

> 从本仓库的 **Releases** 页面下载（`Releases → 最新版本 → Assets`）。
>
> **网盘镜像**（GitHub 下载慢/不通时备用）——`sd-cpp-thor-sm101-20260917.tar.gz`（111 MB，内容 = 上面两个二进制 + SHA256SUMS 打包）：
>
> - 百度网盘：<https://pan.baidu.com/s/1-FiOwmedrh6bMgVGnM14mA?pwd=vxrw>（提取码 `vxrw`）
> - 解包后对包内 `SHA256SUMS` 执行 `sha256sum -c`，或直接用下方校验命令核对单个二进制。

**使用前务必校验**（下载损坏是"上板后各种奇怪报错"的常见来源）：

```bash
sha256sum -c <<'EOF'
02cee738e66d4a0d798a7d778d6bbaeea064ce37f06c3f0bf5b536f7addf1f10  sd-cli
f8cd7e0d1f9bfa3b78d144df0a4d1d830ce4be84315f8012a692c3b95863e567  sd-server
EOF
# 两行都应输出 "OK"
```

---

## 2. 上板（5 步，照抄）

```bash
# ① 板上建目录（数据分区顶层通常属主不是你，需要 sudo）
ssh user@<board> 'sudo mkdir -p <DATA_DIR>/sd-cpp/{bin,models,out,logs} && \
                  sudo chown -R $USER <DATA_DIR>/sd-cpp'

# ② 传二进制（板上没有 rsync，用 scp；关掉压缩更快）
scp -o Compression=no sd-cli sd-server user@<board>:<DATA_DIR>/sd-cpp/bin/

# ③ 给执行权限（scp 不带权限位）
ssh user@<board> 'chmod 755 <DATA_DIR>/sd-cpp/bin/sd-*'

# ④ 冒烟测试：能打印帮助就是二进制没问题
ssh user@<board> '<DATA_DIR>/sd-cpp/bin/sd-cli --help | head -5'

# ⑤ 依赖自检：缺失依赖数应为 0
ssh user@<board> 'ldd <DATA_DIR>/sd-cpp/bin/sd-cli | grep -c "not found"'
```

`ldd` 里出现 `not found` 一般是 CUDA 运行库路径问题；DriveOS 7.0.3 自带 CUDA 12.8
运行库，正常情况下无需设置 `LD_LIBRARY_PATH`。

---

## 3. 模型文件（二进制**不含**权重）

本包只发**二进制**（权重体积太大，且上游有现成分发）。模型来源以
`stable-diffusion.cpp` 上游文档为准（`docs/z_image.md`、`docs/minimax_h3.md`）：

### 3.1 Z-Image-Turbo（文生图，三件套）

| 角色 | 文件 | 体积（本仓库实测） | 上游指定来源 |
|---|---|---|---|
| DiT | `z_image_turbo-Q4_K_S.gguf`（低显存档）<br>`z_image_turbo-Q8_0.gguf`（画质档） | 4.4 GB / 7.2 GB | GGUF：`huggingface.co/leejet/Z-Image-Turbo-GGUF`<br>safetensors：`Comfy-Org/z_image_turbo` → `split_files/diffusion_models` |
| VAE | `ae.safetensors` | 320 MB | `black-forest-labs/FLUX.1-schnell`（Z-Image 复用该 VAE） |
| 文本编码器 | `Qwen3-4B-Q4_K_M.gguf` | 2.4 GB | GGUF：`unsloth/Qwen3-4B-Instruct-2507` → GGUF 版<br>safetensors：`Comfy-Org/z_image_turbo` → `split_files/text_encoders` |

> ⚠️ 这里的 `Qwen3-4B` 是**扩散模型流水线自己的文本编码器**（4B 级小模型），
> 与同机上可能常驻的**文本推理 LLM 服务不是同一个模型**，两者互不依赖。

### 3.2 MiniMax-H3（视频 + 音频，四件套）

| 角色 | 文件 | 体积（本仓库实测） | 上游指定来源 |
|---|---|---|---|
| DiT | `minimax_h3_fl2va_pruned_int8_convrot.safetensors`（int8 剪枝档）<br>或 `minimax_h3_fl2va-Q4_K_M.gguf`（量化档，约 10.6 GB，更省内存） | 19.5 GB | GGUF：`huggingface.co/leejet/MiniMax-H3-GGUF`<br>safetensors：`Comfy-Org/MiniMax-H3` → `diffusion_models` |
| 视频 VAE | `minimax_h3_video_vae_fp16.safetensors` | 4.85 GB | `Comfy-Org/MiniMax-H3` → `vae` |
| 音频 VAE | `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GB | `Comfy-Org/MiniMax-H3` → `vae` |
| 文本编码器 | `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | 17.0 GB | GGUF：`leejet/MiniMax-H3-GGUF`<br>safetensors：`Comfy-Org/MiniMax-H3` → `text_encoders` |

> ⚠️ **H3 的文本编码器必须是 MiniMax-H3 专用变体**（上游明确要求）：
> Qwen3-VL-32B 截断到 **50 层语言层**、且**不带最终 LM norm**；视觉塔（含 3 个 DeepStack merger）必须齐全。
> 张量结构本仓库实测为 **902 个张量 = model 551 + visual 351**，层 0–49。
> 拿错版本会加载失败——**先用下面的"头部比对"确认，再传输**。

### 3.3 我们实测通过的一组文件校验值（供核对）

下面这组文件在 DRIVE Thor（DriveOS 7.0.3 / CUDA 12.8）上**实测出图通过**（512² 与 1024²）。
下载完可以先比一下——**一致就说明你手上是和本仓库验证过的同一批文件**，能少走弯路：

| 文件 | 字节数 | sha256 |
|---|---|---|
| `z_image_turbo-Q4_K_S.gguf` | 4,658,972,736 | `c193f86867daf4e57007ec2fdb7bee8e7273b0f6ea20153ef2cb4037c4930257` |
| `Qwen3-4B-Q4_K_M.gguf`（Z-Image 文本编码器） | 2,497,280,256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| `ae.safetensors`（VAE） | 335,304,388 | `afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38` |

> 注：哈希不一致**不一定**代表文件有问题（可能是不同量化档/不同转换批次）；
> 但如果你用的正是上面这几档，比对一致是最省事的确认方式。
> 校验命令：`sha256sum <文件>`（核对其 sha256），或与上表逐字符比对。

### 3.4 动手前先比对 GGUF 头部（省几 GB 传输的关键习惯）

不必下全量就能确认"这个文件对不对"：GGUF 的张量清单就在文件头部，
用 HTTP Range 拉几 MB 即可解析（`.gguf` 用此法；`.safetensors` 同样有头部 JSON）。

> ⚠️ **拉取量别太小**：头部里含 tokenizer 等大字段——实测 Z-Image 的文本编码器头部一直延伸到
> **5.93 MB**（H3 的文本编码器则只有 24 B）。按 400 KB 拉会直接解析失败（`struct.error`）。
> 本文示例统一拉 **6 MB**，稳。

```bash
curl -sL -r 0-6000000 "<hf-url-of-file>" -o head.bin
python3 - <<'PY'
import struct
f = open("head.bin", "rb"); assert f.read(4) == b"GGUF"
f.read(4)                                     # version
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
def rs():
    n = struct.unpack("<Q", f.read(8))[0]; return f.read(n).decode("utf-8", "replace")
def rv(t):
    m = {0:("<B",1),1:("<b",1),2:("<H",2),3:("<h",2),4:("<I",4),5:("<i",4),
         6:("<f",4),7:("<?",1),10:("<Q",8),11:("<q",8),12:("<d",8)}
    if t == 8: return rs()
    if t == 9:
        et = struct.unpack("<I", f.read(4))[0]; n = struct.unpack("<Q", f.read(8))[0]
        vals = [rv(et) for _ in range(n)]      # 必须读完整，少读会让后续字段全部错位
        return vals if n <= 4 else vals[:4]    # 只展示前 4 个
    fmt, sz = m[t]; return struct.unpack(fmt, f.read(sz))[0]
kvs = {rs(): rv(struct.unpack("<I", f.read(4))[0]) for _ in range(n_kv)}
print("arch =", kvs.get("general.architecture"), "| tensors =", n_tensors)
for _ in range(min(n_tensors, 8)):
    name = rs(); nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    struct.unpack("<IQ", f.read(12))
    print(" ", name, dims)
PY
```

两个用途：① **看 `general.architecture`**——某些第三方打包版会把它改成自家 loader 的值
（例：Comfy-Org 的 H3 DiT GGUF 标成 `wan`），`sd.cpp` 会当成别的模型加载而失败。
> 注意**别把合法的 arch 当异常**：不同模型系列本来就用不同的 arch 值——例如 Z-Image 的 DiT
> 正常就是 `lumina2`，这是对的，不是被改过；
② **逐项比对张量清单**，判断手上已有的文件是不是与目标同构（本仓库就靠这招
复用了已有的 17 GB 文本编码器，省掉一次下载）。

---

## 4. 跑通验证（Z-Image 512²，约十几秒）

```bash
ssh user@<board>
cd <DATA_DIR>/sd-cpp

./bin/sd-cli \
  --diffusion-model models/z_image_turbo-Q4_K_S.gguf \
  --vae models/ae.safetensors \
  --llm models/Qwen3-4B-Q4_K_M.gguf \
  --cfg-scale 1.0 --steps 8 --diffusion-fa \
  -W 512 -H 512 \
  -p "a red apple on a wooden table, studio light" \
  -o out/first_test.png
```

**期望**：日志尾部出现 `generate_image completed in ...` 与 `save result image ... success`，
`out/first_test.png` 生成（约 400 KB），全过程十几秒（本平台 512²/8 步：池空闲 **11.9 s**；同机常驻 LLM 空闲时 ≈13.5 s）。

如果这一步成功，剩下的（1024²、批量、HTTP 服务）都是参数与封装问题，见仓库顶层 [README](../../README.md)。

**长任务记得脱离终端**（断线不会杀掉任务）：

```bash
{ cd <DATA_DIR>/sd-cpp && setsid nohup ./bin/sd-cli ... > logs/run.log 2>&1 < /dev/null & }
```

---

## 5. 关于显存：本平台的"大页池"就是 GPU 显存

这是本平台最容易踩的坑，**单独强调**：

```bash
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages     # 总页数（每页 2 MiB）
cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages  # 空闲页数
```

- **池 = CUDA 能用的显存**（本机 46 GiB 级）。池外那几 GiB 普通内存 CUDA 拿不到；
- ⚠️ **绝对不要为了"腾显存"去写 `nr_hugepages=0`** —— 那会把 CUDA 可用的池清掉，
  表现为"假 OOM"（我们会花半天排查过一次）；
- 池被同机常驻服务占满时，正确做法是**停掉占用方**把池让出来（跑完再启回），
  或用更小的量化档 / 更低分辨率；
- 机制细节见 [主文档 §一 内存预算](../02-h3-video-audio/h3-video-audio.md#一内存预算h3-为什么必须独占池)。

**本平台实测的占用参考**：

| 负载 | CUDA 侧占用 |
|---|---|
| Z-Image-Turbo（Q4_K_S + TE + VAE） | **7.49 GB 磁盘**（6.98 GiB）／**6.82 GiB VRAM** |
| MiniMax-H3（int8 剪枝 DiT + TE Q4 + 双 VAE） | **45.0 GB 磁盘**（41.9 GiB）／**42.0 GiB VRAM**（贴近 46 GiB 池上限，必须独占池） |

---

## 6. 常见错误对照表

| 症状 | 原因 | 处理 |
|---|---|---|
| `no kernel image is available for execution on the device` | 算力不是 `sm_101`（平台不对） | 换平台或自行编译（§8） |
| `cudaMalloc failed: out of memory`（内存"看起来"够） | 池被占 / 或被写成 0 | 停占用方；**不要清池**（§5） |
| `ldd` 报 `not found` | CUDA 运行库路径问题 | 正常情况下 DriveOS 自带；必要时设 `LD_LIBRARY_PATH` 指向 CUDA 的 aarch64 运行库目标路径 |
| `vae decode compute failed` + 随后 `retrying with spatial tiling` 且成功 | 触发了自动分块解码（池余量偏紧） | **正常回退**，无需处理；但说明已贴近上限 |
| `vae encode compute failed` / hires 放大失败 | 放大阶段要额外显存，池余量不足 | 不开 hires；或先腾池 |
| `--hires-upscaler lanczos` 秒退 | 该参数**取值区分大小写** | 写 `Lanczos` |
| `rsync: command not found` | 板上无 rsync | 用 `scp -o Compression=no` |
| 建目录 `Permission denied` | 数据分区顶层属主不是你 | `sudo mkdir -p ... && sudo chown -R $USER ...` |
| 用 `/usr/bin/time` 计时，命令静默秒退 | 板上**没有** `/usr/bin/time` | 改用 `$SECONDS` 或 shell 计时 |
| SSH 起后台任务后命令挂住不返回 | 长任务没脱离终端 | `setsid nohup ... < /dev/null &` |
| 加载 H3 报架构/张量不符 | 文本编码器或 DiT 版本不对 / 被第三方改过 `arch` | 按 §3.4 比对 GGUF 头部 |

更多症状见 [踩坑速查表](../03-troubleshooting/troubleshooting.md)。

---

## 7. 这包是怎么来的（可复现）

| 项 | 值 |
|---|---|
| 上游源码 | `leejet/stable-diffusion.cpp` @ `59c23bc`（2026-09-15） |
| 目标平台 | aarch64 (GNU/Linux) + CUDA **12.8 (sbsa)** + `CMAKE_CUDA_ARCHITECTURES=101` |
| 编译参数 | `-DSD_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DSD_BUILD_SHARED_LIBS=OFF` |
| 工具链 | aarch64 GNU 14 + CUDA 12.8 sbsa（`targets/sbsa-linux`）+ `qemu-aarch64-static` 跑 codegen |
| 路径处理 | 编译期使用 `-ffile-prefix-map`，**二进制内不含构建机的绝对路径** |
| 实测验证 | 在 DRIVE Thor（DriveOS 7.0.3 / CUDA 12.8）上：Z-Image 512²/1024² 出图、MiniMax-H3 视频+音轨生成均跑通 |

> 小知识：`__FILE__` 字符串会随 `assert` 等宏进二进制，`strip` **去不掉**（实测无效）——
> 想让编译产物内外一致、无构建机路径，只能编译期重映射。

---

## 8. 想自己编译 / 想跑视频

- **自己交叉编译**（改版本、改算力目标）：参考上游 `stable-diffusion.cpp` 的构建文档，
  目标平台 aarch64 + CUDA 12.8 sbsa + `CMAKE_CUDA_ARCHITECTURES=101`（见本文 §7 构建来源）；
- **跑 MiniMax-H3 视频 + 音频**：见 [主文档](../02-h3-video-audio/h3-video-audio.md)
  （必须独占池、56/124 帧实测、池门禁脚本）与 [分辨率与显存腾挪实测](../04-resolution-vram/resolution-vram-measured.md)；
- **做成按需 HTTP 服务给别的程序/AI 调用**：见 [`scripts/zimage-api.py`](../../scripts/zimage-api.py)
  （单文件标准库实现，空闲零显存、异步 job、产物直链）+ [`scripts/zimage-api.service`](../../scripts/zimage-api.service)。
---

[整理者注] 本文由真实部署过程整理；内部主机名、账号、凭据与业务标识已按公开分享规范移除或代称化
（如板端数据分区统一写作 `/brand_data`）。文中的命令与数据均来自实际运行记录。
