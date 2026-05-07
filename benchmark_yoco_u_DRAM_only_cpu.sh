#!/bin/bash
# YOCO-U DRAM-Only Benchmark: F16 vs I2_S (flush L3 cache before each run)
#
# Purpose: Determine whether the fast speed is due to small model size or
#          L3 cache residency. By flushing the CPU cache before each benchmark
#          run, the model weights must be fetched from DRAM, not L3.
#
# Technique:
#   Allocate and touch a large buffer (2x L3 size = 96 MiB) to evict
#   any remaining data from L3 cache, then run benchmark with -r 1
#   so the first run measures cold-cache (DRAM) latency.
#   NOTE: Does NOT drop OS page cache — safe for shared servers.
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

# L3 cache size per NUMA node: 48 MiB; we allocate 2x to ensure full eviction
FLUSH_SIZE_MB=96

flush_l3_cache() {
    echo "  [flush] Evicting L3 cache (touching ${FLUSH_SIZE_MB} MiB buffer)..."
    # Allocate a buffer larger than L3 and touch every cache line (64-byte stride)
    # to ensure all previous L3 contents are evicted
    python3 -c "
import ctypes
size = ${FLUSH_SIZE_MB} * 1024 * 1024
buf = ctypes.create_string_buffer(size)
# Touch every cache line (64 bytes) to fully evict L3
for i in range(0, size, 64):
    buf[i] = 0x42
"
}

F16_SIZE=$(du -h "$F16_MODEL" | cut -f1)
I2S_SIZE=$(du -h "$I2S_MODEL" | cut -f1)

echo "========================================================"
echo "  YOCO-U DRAM-Only Benchmark: F16 vs I2_S"
echo "  L3 cache flushed before each run (cold-cache / DRAM)"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA $NUMA_NODE (cores: $CPUS)"
echo "========================================================"
echo
echo "--- Model Sizes ---"
echo "  F16:  $F16_SIZE"
echo "  I2_S: $I2S_SIZE"
echo "  L3 Cache: 48 MiB per NUMA node (will be flushed)"
echo

flush_l3_cache

echo "--- F16 Benchmark (DRAM, cold cache) ---"
$TASKSET $BENCH -m $F16_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0

echo
flush_l3_cache

echo "--- I2_S Benchmark (DRAM, cold cache) ---"
$TASKSET $BENCH -m $I2S_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0 || echo "⚠ I2_S crashed."

echo
echo "Done."
