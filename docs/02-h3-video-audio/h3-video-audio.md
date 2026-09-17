# 在 DRIVE Thor 上跑 MiniMax-H3（视频 + 音频双轨生成）

> One-liner: Running MiniMax-H3 (joint video+audio generation) on an NVIDIA DRIVE Thor
> board — how a **45 GB** weight set (**42.0 GiB** once loaded) fits into a **46 GiB** pool, why the
> pool must be **exclusively** owned, the model-format traps that cost us a wasted
> download, and full measured performance (56/124-frame clips) on real hardware.
>
> 适用：DRIVE Thor（p3960-0010 / Tegra264 / sm_101a），DriveOS 7.0.3，CUDA 12.8
> 本仓库版：脱敏整理

---

## TL;DR

| 项 | 结果 |
|---|---|
| 框架 | `stable-diffusion.cpp`（`-M vid_gen`，一个交叉编译好的 aarch64 二进制，**无需 PyTorch / ComfyUI**） |
| 权重总量（全 VRAM） | **43 017 MB** = 文本编码器 17 376 + DiT 20 083 + VAE 5 558 |
| 实测 864×480×**56 帧**（2.33 s 片，8 步） | **314 s**（≈5.2 分钟） |
| 实测 864×480×**124 帧**（5.17 s 片，8 步） | **953 s**（≈15.9 分钟） |
| 峰值池占用 | **42.8 / 46 GiB** —— 所以 **必须独占池**（跑前停掉同机常驻 LLM 服务） |
| 有效算力（反推） | int8 GEMM ≈ **7.6 TFLOPS** |
| 温度 | 全程 48–54 °C（被动散热，无压力） |
| 音轨 | **模型联合生成**（与画面同一个 DiT + 独立音频 VAE）——输出文件**自带音轨**，不是后期配音。实测产物为 `h264 + aac 32 kHz 立体声`（`sd-cli` 直出的 avi 为 `mjpeg + pcm_s16le`） |

**一句话**：能跑，但它是"**独占池 + 分钟级**"的负载——适合后台批量出片，不适合交互式秒回。

> **🚀 最快路径（不想自己交叉编译）**
> 本仓库提供**预编译好的 aarch64 二进制**（`sd-cli` / `sd-server`，sm_101，含 sha256），
> 下载 → 校验 → 上板即可，跳过本文的编译环节。
> 步骤见 **[prebuilt-binary-quickstart.md](../05-prebuilt-binary/prebuilt-binary-quickstart.md)**。

---

> **单位口径（本文档统一）**
> - **GiB** 用于池／显存口径（本平台大页池 46 GiB 级；`sd.cpp` 打印的 "MB" 实际是 **MiB**）
> - **GB** 用于磁盘文件口径（1 GB = 10⁹ B）
> - 例：H3 权重磁盘占 **45.0 GB**（= 41.9 GiB），运行时占 VRAM **42.0 GiB**（框架报 `43 017 MB`）

## 一、内存预算：H3 为什么必须独占池

### 1.1 前提：本平台上"大页池 = GPU 显存"

Thor 是统一内存架构（无独立显存）。实测口径（`ctypes` 直调 `libcudart`，不依赖任何模型）：

```
cudaMemGetInfo: free=45.93 GiB  total=46.00 GiB     ← 池 23552 页 × 2 MiB 全空时
 8/16/24/32/36/40/42/44 GiB malloc+memset: OK
46 GiB malloc: FAIL rc=2 (out of memory)
```

框架侧报同一数字：`ggml_cuda_init: Total VRAM: 47104 MiB` = 23552 × 2 MiB。

⇒ **CUDA 侧一切占用（权重 + 计算缓冲）必须落在池内**；池外那几 GiB 普通内存 CUDA 拿不到。
⇒ 推论（很重要的操作纪律）：**不要为了"腾显存"去写 `nr_hugepages=0`** —— 那会把 CUDA 可用的池清掉，
表现为"假 OOM"（我们第一次 Z-Image 失败就是这么来的，排查了半天）。

查池：

```bash
cat /sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages     # 总页数（每页 2 MiB）
cat /sys/kernel/mm/hugepages/hugepages-2048kB/free_hugepages  # 空闲页数
sudo cat /sys/kernel/debug/nvmap/iovmm/clients                 # 谁在占 GPU 内存、占多少
```

### 1.2 本机预算（实测）

| 占用者 | 池内占用 | 备注 |
|---|---|---|
| 同机常驻 LLM 服务（单槽、长上下文） | ≈ 36.7 GiB | 权重 + KV cache + 缓冲 |
| **MiniMax-H3（本配置）** | **42.0 GiB**（VRAM 口径） | 46 GiB 池装不下"两者之和" |
| 池总量 | 46 GiB | — |

⇒ **H3 与常驻 LLM 服务无法共存**：必须先把 LLM 服务停掉，把池整块让出来。

实测释放/恢复时间：

| 动作 | 实测 |
|---|---|
| `systemctl stop <llm-service>` → nvmap 占用 36.7 GB → 0、池释放 | **20 秒内** |
| 跑完 `systemctl start <llm-service>` → 服务 READY、nvmap 回到 36.7 GB | **20 秒内** |

注：即使池全空，单进程能拿到的上限也是 **≈44-45 GiB**（实测 44 GiB 分配 + 实写 ✅、46 GiB ❌），
H3 权重运行时占 **42.0 GiB**，相对 46 GiB 的池**基本是贴着上限**在跑——所以留给"池外普通内存"的余量要盯紧（`MemAvailable`），
别让系统侧（SSH/监控/日志）没内存。

---

## 二、权重选型：四个文件、三套现成件的取舍

H3 在 `stable-diffusion.cpp` 里需要 **4 个文件**（DiT + 文本编码器 + 视频 VAE + 音频 VAE）。
社区/本地通常已有若干套，**先算尺寸再决定下不下**，能省掉十几 GB 的传输：

| 文件 | 体积 | 说明 |
|---|---|---|
| DiT `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | **19.53 GiB** | 剪枝 int8 档（画质优先；本机已有） |
| 文本编码器 `qwen3vl_32b_minimax_h3-Q4_K_M.gguf` | **16.97 GiB** | Q4_K_M 档（本机 ComfyUI 件，**实测与官方同构，直接复用**） |
| 视频 VAE `minimax_h3_video_vae_fp16.safetensors` | 4.85 GiB | — |
| 音频 VAE `minimax_h3_audio_vae_fp32.safetensors` | 0.56 GiB | — |
| **合计** | **42.0 GiB**（磁盘 41.9 GiB = 45.0 GB） | 与运行时实报 `43 017 MB`（MiB）吻合 |

**更省内存的替代档**（需要时再下，本次未用）：官方 GGUF 量化版 DiT 走 Q4_K_M 约 **10.64 GiB**
⇒ 可把总量压到 ≈ 33 GiB，为 124 帧档多留 ~9 GiB 余量。**快慢/画质需自行 A/B**，不要凭直觉认为
"量化一定更快"——本平台瓶颈在算力与带宽，不在权重读取（Z-Image 上实测 Q4 与 Q8 速度几乎相同）。

### 2.1 陷阱一：ComfyUI 版的 DiT GGUF **不能直接用**

我们手上有 Comfy-Org 打包的 H3 DiT GGUF（18.5 GiB），加载即失败。原因：

```
GGUF 头部:  general.architecture = wan
```

Comfy-Org 为自家 loader 打了兼容标（`wan`），而 **`sd.cpp` 源码里没有任何 comfy 兼容分支**
⇒ 它会把 H3 权重当成 Wan 模型去分派，必然失败。

**结论**：DiT 要么用**官方/leejet 版**的 GGUF，要么用 **safetensors**（int8 剪枝档）。
**不要**拿"ComfyUI 能跑"的 GGUF 直接喂 `sd.cpp`。

### 2.2 陷阱二：同名的文本编码器 GGUF **可以直接复用**（省 17 GiB）

同一个 ComfyUI 目录里的 TE GGUF（`qwen3vl_32b_minimax_h3-Q4_K_M.gguf`）**可以用**：
与官方同名文件逐项比对，**完全同构**：

```
902 个张量 = model 551 + visual 351（含 **3 组** DeepStack merger：`visual.deepstack_merger_list.{0,1,2}`，层 0–49）
```

⇒ 省掉一次 **12–17 GiB** 的下载与传输。

**♻️ 可复用手法（强烈推荐）：远程比对 GGUF 头部，比下载快 100 倍**

不要为了确认"两个文件是不是同一个"去下全量。GGUF 的张量清单就在文件头部，
用 HTTP Range 拉几 MB 就能解析出全部 kv 与张量名。

> ⚠️ **拉取量别太小**：头部含 tokenizer 等大字段——实测 Z-Image 的文本编码器头部到 **5.93 MB**
> （H3 的文本编码器只有 24 B）。按 400 KB 拉会解析失败；示例统一拉 **6 MB**。

```bash
# 只拉头部（示例走镜像站；换成你的实际 URL）
curl -sL -r 0-6000000 "https://<hf-mirror>/<repo>/resolve/main/<file>.gguf" -o head.bin
```

```python
# 解析 GGUF 头部：版本、kv、张量名与维度
import struct, json
f = open("head.bin", "rb")
assert f.read(4) == b"GGUF"
ver = struct.unpack("<I", f.read(4))[0]
n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

def rs():
    ln = struct.unpack("<Q", f.read(8))[0]
    return f.read(ln).decode("utf-8", "replace")

def rv(t):                       # GGUF 值类型
    import struct as s
    m = {0: ("<B",1), 1: ("<b",1), 2: ("<H",2), 3: ("<h",2), 4: ("<I",4), 5: ("<i",4),
         6: ("<f",4), 7: ("<?",1), 10: ("<Q",8), 11: ("<q",8), 12: ("<d",8)}
    if t == 8:  return rs()
    if t == 9:  # 数组
        et = s.unpack("<I", f.read(4))[0]; n = s.unpack("<Q", f.read(8))[0]
        vals = [rv(et) for _ in range(n)]            # 必须读完整，少读会让后续字段错位
        return vals if n <= 4 else vals[:4]          # 只展示前几项
    fmt, sz = m[t]; return s.unpack(fmt, f.read(sz))[0]

kvs = {rs(): rv(struct.unpack("<I", f.read(4))[0]) for _ in range(n_kv)}
for _ in range(n_tensors):
    name = rs(); nd = struct.unpack("<I", f.read(4))[0]
    dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
    ttype, off = struct.unpack("<IQ", f.read(12))
    print(name, dims)
print("arch =", kvs.get("general.architecture"))
```

**这一步同时解决了两个陷阱**：① 一眼看出 `general.architecture` 是不是被第三方改过
（注意**别误判合法值**：不同模型系列 arch 本来就不同，例如 Z-Image 的 DiT 正常就是 `lumina2`）；
② 逐项比对张量清单，判断"疑似缺件"是否真的同构（能省十几 GB 传输）。

---

## 三、怎么跑：池门禁脚本（**推荐入口**）

手动停服务容易忘恢复。用一段门禁脚本包起来——**检测到 GPU 被占 → 自动让池 → 跑完自动恢复**：

```bash
#!/bin/bash
# sd-pool-gate.sh — 扩散模型统一入口（Z-Image 可与 LLM 服务共存；H3 必须独占池）
# 用法: sd-pool-gate.sh h3 <DiT> <TE> <W> <H> <帧数> <步数> <输出名> [prompt]
#       sd-pool-gate.sh zimage -W 1024 -H 1024 --steps 8 -p "..." -o out/x.png
#       sd-pool-gate.sh status
set -u
SD=${SD_DIR:-/brand_data/sd-cpp}; cd "$SD" || exit 1
M=models; LLM_UNIT=${LLM_UNIT:-llama-server}
DIT=$M/minimax_h3_fl2va_pruned_int8_convrot.safetensors
TE=$M/qwen3vl_32b_minimax_h3-Q4_K_M.gguf
VAE=$M/minimax_h3_video_vae_fp16.safetensors
AVAE=$M/minimax_h3_audio_vae_fp32.safetensors
HP=/sys/kernel/mm/hugepages/hugepages-2048kB

pool_state() {
  echo "池: Total=$(cat $HP/nr_hugepages) Free=$(cat $HP/free_hugepages)" \
       "| GPU 占用: $(sudo -n cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null \
                      | awk '/total/{gsub("K","",$2); print $2"K"}')" \
       "| 温度 $(($(cat /sys/class/thermal/thermal_zone0/temp)/1000))C"
}
gpu_busy() { pgrep -f "bin/(sd-cli|sd-server) " >/dev/null || pgrep -f "$LLM_UNIT" >/dev/null; }

start_llm() {
  sudo -n systemctl start "$LLM_UNIT"
  for i in $(seq 1 12); do
    systemctl is-active --quiet "$LLM_UNIT" && { echo "  LLM 服务已恢复($((i*2))s)"; return 0; }
    sleep 2
  done
  echo "  !!! LLM 服务恢复超时"
}
stop_llm() {
  sudo -n systemctl stop "$LLM_UNIT"
  for i in $(seq 1 15); do           # 等 nvmap 占用归零 = 池真正释放
    nv=$(sudo -n cat /sys/kernel/debug/nvmap/iovmm/clients 2>/dev/null \
         | awk '/total/{gsub("K","",$2); print $2+0}')
    [ "${nv:-1}" = "0" ] && { echo "  池已释放($((i*2))s)"; return 0; }
    sleep 2
  done
  echo "  !!! 池释放等待超时（继续，风险自负）"
}

case "${1:-status}" in
  status) pool_state; gpu_busy && echo "GPU: 有进程在跑" || echo "GPU: 空闲"; exit 0 ;;
  zimage)
    shift; pool_state
    [ "$(free -m | awk '/Mem:/{print $7}')" -lt 9000 ] && echo "警告: 池外可用内存偏低"
    T0=$SECONDS
    ./bin/sd-cli --diffusion-model $M/z_image_turbo-Q4_K_S.gguf --vae $M/ae.safetensors \
      --llm $M/Qwen3-4B-Q4_K_M.gguf --cfg-scale 1.0 --steps 8 "$@"
    echo "rc=$? wall=$((SECONDS-T0))s"; exit 0 ;;
  h3)
    shift
    DIT=${1:-$DIT}; TE=${2:-$TE}; W=${3:-864}; H=${4:-480}
    FR=${5:-56}; ST=${6:-8}; OUT=${7:-h3_out}
    shift 7 || true
    echo "=== H3 $(basename "$DIT") + $(basename "$TE") ${W}x${H} ${FR}f ${ST}step ==="
    pool_state
    RESTORE=0
    if gpu_busy; then echo "检测到 GPU 占用 → 停 LLM 服务让出池（跑完自动恢复）"; RESTORE=1; stop_llm; fi
    T0=$SECONDS
    ./bin/sd-cli -M vid_gen --diffusion-model "$DIT" --vae $VAE --audio-vae $AVAE --llm "$TE" \
      -p "${PROMPT:-a cat surfing on a tropical ocean wave, cinematic}" \
      --cfg-scale 1.0 -W "$W" -H "$H" --video-frames "$FR" --steps "$ST" \
      --diffusion-fa --fps 24 -o "out/$OUT" "$@"
    RC=$?; echo "rc=$RC wall=$((SECONDS-T0))s"; pool_state
    [ "$RESTORE" = 1 ] && { echo "恢复生产服务"; start_llm; pool_state; }
    exit $RC ;;
  *) echo "用法: sd-pool-gate.sh {zimage|h3|status} ..."; exit 1 ;;
esac
```

要点：

- **等 `nvmap` 占用归零**再开跑——`systemctl stop` 返回 ≠ 显存已释放（实测 20 秒内释放完）；
- **跑完自动 `start`**，不让生产服务躺在地上（脚本里 `RESTORE` 分支）；
- H3 是长任务（最长一档 16 分钟），**务必脱离 SSH**（`setsid nohup ... < /dev/null &`），
  否则断线会被 SIGHUP 连带杀掉。

---

## 四、实测性能（2026-09-16，864×480，8 步，`--cfg-scale 1.0 --diffusion-fa`）

| 档位 | 总耗时 | 文本条件 | 采样 | s/step | 视频 VAE | 音频 VAE | 峰值池占用 |
|---|---|---|---|---|---|---|---|
| 256×256×5 帧（2 步，POC） | **29 s** | 10.3 s | 13.3 s | 6.6 s | 4.4 s | 0.7 s | — |
| 864×480×**56 帧**（2.33 s 片） | **314 s** | 10.0 s | 240.8 s | 30.1 s | 56.7 s | 4.1 s | 41.8 / 46 GiB |
| 864×480×**124 帧**（5.17 s 片） | **953 s** | 10.1 s | 802.1 s | 100.3 s | 129.6 s | 8.4 s | **42.8 / 46 GiB** |

- **权重内存明细**（框架实报 `total params memory size`）：
  **43 017.20 MB** = text_encoders 17 375.93 + diffusion_model 20 083.17 + vae 5 558.10（**全部在 VRAM**）
- **载入耗时**：DiT 冷载 ≈ 11 s、TE ≈ 11 s（热载 < 1 s）；磁盘顺序读 ≈ 1.0 GB/s
- **有效算力反推**：124 帧档 758 TFLOP/step ÷ 100.3 s ≈ **7.6 TFLOPS**（int8 GEMM 现实值；
  同平台 bf16 峰值实测 ≈ 60 TFLOPS）——**这就是瓶颈所在**，换量化档位不会显著改变量级
- **性价比档位**：**56 帧档最优**（5.2 分钟换 2.33 秒成片）；124 帧档耗时是 3 倍、只换到 2.2 倍时长
- **音轨**：视频与音频在同一流程内产出（独立 audio VAE），无需二次配音
- **温度**：48–54 °C，远低于红线，无散热压力

> 横比参考：同一档位在一台 x86 + 高功耗 GPU 的机器上跑 244.6 s（≈2.5 分钟），
> **Thor 慢 1.3×（56 帧档）～3.9×（124 帧档）**——但它是 60 W 级的车规 SoC，且整机可离线部署。权衡点在这里。

### 4.1 输出：文件自带音轨（ffprobe 实证）

**必须传 `--audio-vae <音频VAE>`**，否则只有画面没有声音。上游文档原文：

> *Omitting `--audio-vae` still runs the joint diffusion model but produces video without a decoded audio track.*

实测产物（`ffprobe -show_entries stream=codec_type,codec_name,channels,sample_rate,duration` 原样）：

| 产物 | 视频流 | 音频流 |
|---|---|---|
| `h3_864x480_56f_2s33.mp4` | h264，**2.333 s** | **aac，32 000 Hz，2 声道，2.325 s** |
| `h3_864x480_124f_5s17.mp4` | h264，**5.167 s** | **aac，32 000 Hz，2 声道，5.175 s** |
| `h3_864x480_56f.avi`（`sd-cli` 直出格式） | mjpeg | **pcm_s16le，32 000 Hz，2 声道** |

⇒ **音视频时长对齐**（2.325 vs 2.333 s；5.175 vs 5.167 s），音轨随画面一起由模型生成。
⇒ 若你拿到的文件没有声音：先确认命令行里带了 `--audio-vae`（常见遗漏），再用
`ffprobe <文件>` 看是否存在 audio 流。


---

## 五、踩坑清单

| 症状 | 根因 | 处理 |
|---|---|---|
| 传 `--video-frames 60` 却得到 73 帧 | 本模型帧数按 **`17k+5` 网格向上对齐**（源码实测） | 要精确控制就用网格值：**5 / 22 / 39 / 56 / 73 / 90 / 107 / 124** |
| 加载即报 architecture 不符 / 被当 Wan 模型 | ComfyUI 版 DiT GGUF 头部 `general.architecture = wan` | 换官方 GGUF 或 safetensors 版 DiT |
| 分配失败 `cudaMalloc failed: out of memory`（明明内存"看起来"够） | 池被同机服务占满、或你**把池写成了 0** | 停服务让池；**永远不要 `nr_hugepages=0`** |
| 传输/下载多花了十几 GB | 没先比对 GGUF 头部就重下 | 用 §2.2 的 Range 头部解析先比对 |
| `hf download --include "a" "b"` 只下了第一个 | CLI 把 `--include` 的多个实参当"显式文件名清单" | 分次下载，或用 `--include "*.gguf"` |
| 用 `/usr/bin/time` 计时命令静默秒退 | 板上**没有** `/usr/bin/time`，找不到命令→看起来像加载失败 | 用 `$SECONDS` 或 shell 内置计时 |
| SSH 起后台任务后命令挂住不返回 | 长任务未脱离终端 | `{ cd DIR && setsid nohup cmd > log 2>&1 < /dev/null & }` |
| 板上没有 `rsync` | 未安装 | 用 `scp -o Compression=no`（实测 ~112 MB/s，够用） |
| 数据盘顶层建目录 `Permission denied` | 顶层属主不是你的账号 | `sudo mkdir -p ... && sudo chown -R $USER ...` |

---

## 六、复现清单

1. **拿到二进制**：用本仓库的预编译包（[prebuilt-binary-quickstart.md](../05-prebuilt-binary/prebuilt-binary-quickstart.md)），
   或自己交叉编译 `stable-diffusion.cpp`（aarch64 + CUDA 12.8 sbsa + `sm_101`，见 `../02-cross-compile/`）。
2. **拿权重**：DiT（官方 GGUF 或 int8 safetensors）+ TE（Q4_K_M GGUF）+ 视频/音频 VAE，
   逐个 `sha256` 校验；**动手前先按 §2.2 比对 GGUF 头部**。
3. **算池**：`nr_hugepages × 2 MiB` 必须 **≥ 45 GiB**（H3 运行时占 42.0 GiB，贴近 46 GiB 池上限），
   并确认 `MemAvailable` 还有几 GB 余量给系统。
4. **让池**：停掉同机占 GPU 的服务，等 `nvmap` 占用归零（脚本已包）。
5. **跑**：`sd-cli -M vid_gen --diffusion-model <DiT> --vae <video-vae> --audio-vae <audio-vae> --llm <TE> -W 864 -H 480 --video-frames 56 --steps 8 --diffusion-fa --fps 24 -o out/clip`
6. **恢复**：`systemctl start <llm-service>`，确认服务 READY、池占用回到原值。
---

[整理者注] 本文由真实部署过程整理；内部主机名、账号、凭据与业务标识已按公开分享规范移除或代称化
（如板端数据分区统一写作 `/brand_data`）。文中的命令与数据均来自实际运行记录。
