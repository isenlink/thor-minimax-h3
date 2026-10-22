# Thor05 原内核 CUDA13 前向兼容：基础计算复现成功

## 结论

2026-10-21，Thor05保留 6.1.119-rt45-prod-rt-tegra 内核及原 nvgpu、系统 libnvrm_*，隔离加载 NVIDIA 官方 cuda-compat-orin-13-2 后，CUDA13.2 cubin 与 PTX JIT 两臂真实数值计算均通过。无需先更换 6.8 内核。尚未验证 PyTorch cu130、ComfyUI、FP8/INT8、H3、长时间稳定性和性能，不能宣称 H3 已复现，也不能认定此库等于作者私下分发工件。

## 输入来源

官方包：https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-compat-orin-13-2_13.2.44290101-1_arm64.deb
包 SHA256：3331f3ac94eecc0e3c0d1976c8fa306c2bf33dd6f771e0d61bb583cab3590037，与 NVIDIA Packages.gz 相同。
libcuda.so.1.1 SHA256：e4c849be8e1bca055c13c8ee0866abfbc58a7041db32c9534ea07853da4f801d，与 内部缓存（略）相同。
宿主解包（内部目录，略）/packages/orin-13.2；仅 dpkg-deb -x，不运行安装脚本。

复用已有 CUDA13.2.86 libcudart、旧 a1-kernel-sm110.cubin、a1-smoke.c oracle。适配源码见 compat-smoke.c：移除 /a1/lib 硬编码，允许记录 SM101/110，记录实际库映射；数值判据不变。PTX 使用旧完整 x86_64/toolkit/bin/nvcc13.2.86 对同一 a1-kernel.cu 生成，PTX .version9.2、target sm_110；设置 CUDA_CACHE_DISABLE=1。

## 验收

| 项目 | cubin | PTX JIT |
|---|---|---|
| Driver / Runtime API | 13020 / 13020 | 13020 / 13020 |
| cuInit / context / alloc / copy / launch / sync / copyback / cleanup | PASS | PASS |
| 设备 | Thor SM110、1个 | Thor SM110、1个 |
| 数值 | 65536 uint32 逐项等于 i+17 | 65536 uint32 逐项等于 i+17 |
| 标记 / exit | COMPAT_CUDA13_COMPUTE_PASS / 0 | COMPAT_CUDA13_COMPUTE_PASS / 0 |
| 前后 dmesg 差异 | 0行 | 0行 |

原始记录：cubin-oracle.log、ptx-oracle.log、dmesg-before/after-cubin.txt、dmesg-before/after-ptx.txt。两臂同 boot_id=75a607ab-1908-47cb-b059-354550f6f1c4。vLLM 暂停后25600个大页全部归还；cubin 后仍全部空闲。

## 板端隔离与复现命令

仅 Thor05，目录 /zeekr_data/ai_workspace/h3-cu13-compat-20261021。运行前必须确认GPU独占；以下命令会调用GPU，不能对正在使用中的板卡直接照跑。

```bash
cd /zeekr_data/ai_workspace/h3-cu13-compat-20261021
sudo -n timeout --signal=TERM --kill-after=5 45 env \
  LD_LIBRARY_PATH="$PWD/compat_orin:$PWD:/usr/lib:/usr/local/cuda-12.8/thor/lib64:/usr/local/cuda-12.8/lib64" \
  ./compat-smoke ./a1-kernel-sm110.cubin
sudo -n timeout --signal=TERM --kill-after=5 45 env CUDA_CACHE_DISABLE=1 \
  LD_LIBRARY_PATH="$PWD/compat_orin:$PWD:/usr/lib:/usr/local/cuda-12.8/thor/lib64:/usr/local/cuda-12.8/lib64" \
  ./compat-smoke ./a1-kernel-compute110.ptx
```

## 恢复与下一步

用户授权暂停的旧 vLLM27B/MTP4 进程 argv/env/cwd 已在板端0600私有快照保存。实验结束后通过 transient thor05-vllm032-compat-restore.service 执行 restore-vllm.py 恢复；未改变原参数及系统默认CUDA，不启用新开机服务。恢复后 argv/env 与快照逐项完全一致；8081 /health=200；按原默认思考配置实测 1+1，content 为“2”，finish_reason=stop，质量 PASS。最初尝试 enable_thinking=False 时答案“2”落在 reasoning 而 content=None，已记录为原服务 deepseek_r1 解析配置行为；未因此修改原服务配置。

下一阶段基于此已验证用户态组合，隔离准备 PyTorch cu130 + comfy_kitchen，再做 FP8/INT8 和 H3 原配方；不混入6.8/R39路线，不先上稀疏注意力。6.8路线独立等待作者补充资料。
