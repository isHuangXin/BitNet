#!/bin/bash
# YOCO-U DRAM-Only Benchmark: F16 vs I2_S
#
# Purpose: Determine whether the fast speed is due to small model size or
#          L3 cache residency. Uses llama-bench --flush-cache to evict L3
#          before each repetition, so every rep measures cold-cache (DRAM) latency.
#
# Usage: cd /home/huangxin/code_list/BitNet && bash benchmark_yoco_u_DRAM_only_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_yoco_u_DRAM_only_cpu.sh 8 0    # 8 threads, NUMA 0

set -e

F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch_L3/yoco-u-L3-f16/ggml-model-f16.gguf"
I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch_L3/yoco-u-L3-bitnet-i2s/ggml-model-i2_s.gguf"
BENCH="./build/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

TASKSET=""
if [ -n "$NUMA_NODE" ]; then
    TASKSET="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
    CPUS=$(numactl --hardware 2>/dev/null | grep "node $NUMA_NODE cpus:" | sed 's/.*cpus: //')
fi

F16_SIZE=$(du -h "$F16_MODEL" | cut -f1)
I2S_SIZE=$(du -h "$I2S_MODEL" | cut -f1)

echo "========================================================"
echo "  YOCO-U DRAM-Only Benchmark: F16 vs I2_S"
echo "  L3 cache flushed before each rep (--flush-cache)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA $NUMA_NODE (cores: $CPUS)"
echo "========================================================"
echo
echo "--- Model Sizes ---"
echo "  F16:  $F16_SIZE"
echo "  I2_S: $I2S_SIZE"
echo "  L3 Cache: 48 MiB per NUMA node (will be flushed)"
echo

echo "--- F16 Benchmark (DRAM, cold cache) ---"
$TASKSET $BENCH -m $F16_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0 --flush-cache

echo
echo "--- I2_S Benchmark (DRAM, cold cache) ---"
$TASKSET $BENCH -m $I2S_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0 --flush-cache || echo "⚠ I2_S crashed."

echo
echo "Done."
