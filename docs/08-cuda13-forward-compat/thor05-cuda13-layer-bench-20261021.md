# Thor05 CUDA13：本地模型需求与速度配对结果

日期：2026-10-21。仅 Thor05，原 6.1.119-rt45-prod-rt-tegra 内核、nvgpu、系统 libnvrm 不变。

## 结论

已确认本地 H3、Music3、Qwen-Image2.1 的 INT8 ConvRot 权重可从 ComfyUI CUDA13 原生量化后端获益。模型可在 CUDA12 回退运行，明确要求 cu130 的是当前 ComfyUI 的优化后端，不是权重文件本身。普通GGUF/BF16、现有LLM FP8权重没有因此被认定必须升级。

Music3完整36层DiT forward提升3.19–4.20倍；H3与Qwen-Image本轮只测真实权重层，不能外推为视频/生图端到端倍数。

## 完整 Music3 DiT forward

| Latent长度 | cu12.8 eager ms | cu13 eager ms | cu13 native ms | 12→13原生倍数 | 同cu13后端倍数 | 输出NRMSE | Cosine |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 242.667 | 236.977 | 57.773 | 4.20 | 4.10 | 3.76% | 0.999292 |
| 512 | 315.542 | 310.120 | 99.000 | 3.19 | 3.13 | 4.10% | 0.999159 |

完整载入现有2.50GB INT8 ConvRot DiT，36层，所有state_dict键匹配。固定CPU随机种子20261021、x=[1,128,L]、context=[1,16,32768]、t=0.5、conditioning_scale=1、BF16。两环境输入哈希一致；checkpoint SHA256=d6b959633e69899f99f3a92d6741c0fe79f26958a30811e50e372ef978b24d5f。预热2次、5次墙钟同步计时取中位数。只测一次DiT forward，不含文本编码、采样循环、VAE和音频生成。

## 三模型真实权重层

| 模型 | M | cu12.8 eager ms | cu13 eager ms | cu13 native ms | 12→13倍数 | 输出NRMSE |
|---|---:|---:|---:|---:|---:|---:|
| H3 | 256 | 2.988 | 2.960 | 0.454 | 6.58 | 0.441% |
| H3 | 1024 | 4.373 | 4.324 | 1.539 | 2.84 | 0.450% |
| Music3 | 256 | 0.563 | 0.536 | 0.154 | 3.65 | 0.502% |
| Music3 | 1024 | 0.841 | 0.795 | 0.414 | 2.03 | 0.503% |
| QwenImage21 | 256 | 1.353 | 1.333 | 0.247 | 5.47 | 0.466% |
| QwenImage21 | 1024 | 2.117 | 2.056 | 0.840 | 2.52 | 0.462% |

真实checkpoint抽取原始weight/weight_scale/comfy_quant，不重新量化。Music3 N×K=2048×2048；H3=5376×7168；Qwen-Image2.1=4096×4096。各权重层与输入哈希在三环境逐项一致；预热3次、CUDA event 7次取中位数，记录同步墙钟。输入是固定随机BF16激活，不是完整工作流采集的激活。

## 环境与准确性边界

- CUDA12：既有custom torch2.13.0/cu12.8；CUDA13：官方AArch64 torch2.13.0+cu130；均为comfy-kitchen0.2.31、CPU线程4，禁用Triton。不同PyTorch构建的影响通过第三组同cu130 eager对照分离。
- CUDA13驱动来自已验证官方cuda-compat-orin13.2；只在测试进程优先加载，不改系统默认CUDA。独立环境及下载包位于 /zeekr_data/ai_workspace/h3-cu13-compat-20261021/bench 与宿主内部目录（略）。
- 当前 ComfyUI quant_ops.py 在 torch.version.cuda<13 时明确禁用CUDA后端；版本commit=8f04fd6ecf4a548937971e840af2a985529fab76。本轮显式选择eager/native，原生路由不允许静默退回。
- 层级预设数值门NRMSE≤2%、cosine≥0.999；完整DiT门NRMSE≤5%、cosine≥0.995。全部通过。大tensor余弦使用CPU float64累加复核，避免float32归约误差。
- 完整DiT差异3.76–4.10%不等于感知质量已通过。尚未做实际音频/视频/图像质量验收，不据此直接推荐部署。
- 本轮为短基准，未锁频、未做长稳或端到端生成评测。

## 恢复与证据

6个基准进程、24组计时记录；6份结束dmesg均与起始逐字节一致。所有进程退出后25600个大页全部归还。原vLLM27B/MTP4已通过 transient thor05-vllm032-compat-restore2.service 恢复，argv/env与原始快照完全一致，8081 health=200，默认思考模式实际1+1问答content为2。未更改其启动参数、系统库或开机配置。

权威原始证据：宿主内部证据目录（略）；归档 tar 同目录。本目录layer-comparison.json、music-comparison.json可直接复查汇总；脚本及wheel/fixture清单同目录。私有vLLM环境快照仅留板端0600文件，不进入报告或归档。

模型需求细分见 MODEL-CUDA13-INVENTORY.md；下一步若推进部署，先做实际生成与质量验收。
