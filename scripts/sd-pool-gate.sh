#!/bin/bash
# sd-pool-gate.sh — 扩散模型统一入口（Z-Image 可与 LLM 服务共存；H3 必须独占池）
# 用法: sd-pool-gate.sh h3 <DiT> <TE> <W> <H> <帧数> <步数> <输出名> [prompt]
#       sd-pool-gate.sh zimage -W 1024 -H 1024 --steps 8 -p "..." -o out/x.png
#       sd-pool-gate.sh status
set -u
# NOTE: /brand_data/ 是品牌中立代称，指板载数据盘工作目录（真实名含品牌字样，ls / 可见）。
#       可用环境变量 SD_DIR 覆盖：SD_DIR=/path/to/workdir ./sd-pool-gate.sh ...
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
