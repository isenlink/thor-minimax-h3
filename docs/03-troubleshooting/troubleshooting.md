# 踩坑速查表（按症状索引）

扩散模型（MiniMax-H3 视频+音频 / Z-Image）在 DRIVE Thor 上运行的实测踩坑。
详细根因分析见 [主文档](../02-h3-video-audio/h3-video-audio.md)。

| 症状 | 根因 | 处理 |
|---|---|---|
| 加载即报 architecture 不符 / 被当 Wan 模型 | ComfyUI 版 DiT GGUF 头部 `general.architecture = wan`，`sd.cpp` 无 comfy 兼容分支 | 换官方 GGUF 或 safetensors 版 DiT |
| 分配失败 `cudaMalloc failed: out of memory`（明明内存"看起来"够） | 大页池被同机服务占满、或把池写成了 0 | 停服务让池（等 nvmap 归零）；**永远不要 `nr_hugepages=0`** |
| 传输/下载多花了十几 GB | 没先比对 GGUF 头部就重下 | HTTP Range 拉头部解析张量清单先比对（主文档 §2.2） |
| `hf download --include "a" "b"` 只下了第一个 | CLI 把 `--include` 的多个实参当"显式文件名清单" | 分次下载，或用 `--include "*.gguf"` |
| 用 `/usr/bin/time` 计时命令静默秒退 | 板上没有 `/usr/bin/time`，找不到命令→看起来像加载失败 | 用 `$SECONDS` 或 shell 内置计时 |
| SSH 起后台任务后命令挂住不返回 | 长任务未脱离终端 | `{ cd DIR && setsid nohup cmd > log 2>&1 < /dev/null & }` |
| 板上没有 `rsync` | 未安装 | 用 `scp -o Compression=no`（实测 ~112 MB/s，够用） |
| 数据盘顶层建目录 `Permission denied` | 顶层属主不是你的账号 | `sudo mkdir -p ... && sudo chown -R $USER ...` |
