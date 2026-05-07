#!/bin/bash
# YOCO-U Origin Arch: F16 vs I2_S CPU Benchmark
# Models use YOCO-U config: 14 layers (7 stages x 2), 24/48 heads, 4 kv_heads, head_dim=128
#
# Usage: cd /home/huangxin/code_list/BitNet && bash benchmark_yoco_u_origin_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_yoco_u_origin_cpu.sh 8         # 8 threads, no CPU pinning
#   bash benchmark_yoco_u_origin_cpu.sh 8 0       # 8 threads, pinned to NUMA 0 (cores 0-31,64-95)
#   bash benchmark_yoco_u_origin_cpu.sh 8 1       # 8 threads, pinned to NUMA 1 (cores 32-63,96-127)

set -e

F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch/yoco-u-1.5b-f16/ggml-model-f16.gguf"
I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch/yoco-u-bitnet-1.5b-i2s/ggml-model-i2_s.gguf"
BENCH="./build/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

# Build CPU affinity prefix
TASKSET=""
if [ -n "$NUMA_NODE" ]; then
    TASKSET="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    CPUS=$(numactl --hardware 2>/dev/null | grep "node $NUMA_NODE cpus:" | sed 's/.*cpus: //')
    echo "CPU Pinning: NUMA node $NUMA_NODE (cores: $CPUS)"
fi

echo "========================================================"
echo "  YOCO-U Origin Arch: F16 vs I2_S Benchmark"
echo "  Config: 14 layers (7 stages x 2), 24/48 heads, 4 kv_heads, head_dim=128"
echo "  Embedding: Q8_0 quantized"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

echo "--- Model Sizes ---"
echo "  F16:  $(du -h $F16_MODEL | cut -f1)"
echo "  I2_S: $(du -h $I2S_MODEL | cut -f1)"
echo

echo "--- F16 Benchmark ---"
$TASKSET $BENCH -m $F16_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0

echo
echo "--- I2_S Benchmark ---"
$TASKSET $BENCH -m $I2S_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0 || echo "⚠ I2_S crashed. Try threads>=2."

echo
echo "Done."
