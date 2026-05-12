#!/bin/bash
# Embedding Model CPU Benchmark: F32 vs I2_S vs TL2 (with optional Q6_K embedding)
#
# Models: multilingual-e5-base (278M) and multilingual-e5-large (559M)
#
# Usage: cd /home/huangxin/code_list/BitNet && bash benchmark_embedding_cpu.sh [threads] [numa_node]
# Examples:
#   bash benchmark_embedding_cpu.sh 8         # 8 threads, no CPU pinning
#   bash benchmark_embedding_cpu.sh 8 0       # 8 threads, pinned to NUMA 0
#   bash benchmark_embedding_cpu.sh 8 1       # 8 threads, pinned to NUMA 1
#
# NOTE: TL2 requires building with -DBITNET_X86_TL2=ON and generating kernels via:
#   python utils/codegen_tl2.py --model "multilingual-e5-large,multilingual-e5-base" \
#     --BM 128,128,128,128,128,128 --BK 96,96,96,96,96,96 --bm 32,32,32,32,32,32

set -e

BENCH="./build/bin/llama-bench"
THREADS=${1:-4}
NUMA_NODE=${2:-""}

# --- Model paths ---
BASE_DIR="/data2/huangxin/model_list"

# e5-base (278M params, hidden=768)
E5_BASE_F32="${BASE_DIR}/intfloat--multilingual-e5-base/ggml-model-f32.gguf"
E5_BASE_I2S="${BASE_DIR}/intfloat--multilingual-e5-base/ggml-model-i2_s.gguf"
E5_BASE_I2S_Q6K="${BASE_DIR}/intfloat--multilingual-e5-base/ggml-model-i2_s-q6k_embd.gguf"
E5_BASE_TL2="${BASE_DIR}/intfloat--multilingual-e5-base/ggml-model-tl2.gguf"
E5_BASE_TL2_Q6K="${BASE_DIR}/intfloat--multilingual-e5-base/ggml-model-tl2-q6k_embd.gguf"

# e5-large (559M params, hidden=1024)
E5_LARGE_F32="${BASE_DIR}/intfloat--multilingual-e5-large/ggml-model-f32.gguf"
E5_LARGE_I2S="${BASE_DIR}/intfloat--multilingual-e5-large/ggml-model-i2_s.gguf"
E5_LARGE_I2S_Q6K="${BASE_DIR}/intfloat--multilingual-e5-large/ggml-model-i2_s-q6k_embd.gguf"
E5_LARGE_TL2="${BASE_DIR}/intfloat--multilingual-e5-large/ggml-model-tl2.gguf"
E5_LARGE_TL2_Q6K="${BASE_DIR}/intfloat--multilingual-e5-large/ggml-model-tl2-q6k_embd.gguf"

# Build CPU affinity prefix
TASKSET=""
if [ -n "$NUMA_NODE" ]; then
    if ! command -v numactl &>/dev/null; then
        echo "⚠ numactl not found, ignoring NUMA node parameter. Install with: apt install numactl"
        NUMA_NODE=""
    else
        TASKSET="numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE"
        CPUS=$(numactl --hardware 2>/dev/null | grep "node $NUMA_NODE cpus:" | sed 's/.*cpus: //')
        echo "CPU Pinning: NUMA node $NUMA_NODE (cores: $CPUS)"
    fi
fi

# Common bench args: embedding mode, CPU only, 3 repetitions
BENCH_ARGS="-t $THREADS -p 128,256,512 -n 0 -r 3 -ngl 0 -embd 1"

echo "========================================================"
echo "  Embedding Model CPU Benchmark"
echo "  Models: multilingual-e5-base (278M), e5-large (559M)"
echo "  Formats: F32, I2_S, I2_S+Q6K_embd, TL2, TL2+Q6K_embd"
echo "  Threads: $THREADS"
[ -n "$NUMA_NODE" ] && echo "  CPU Affinity: NUMA node $NUMA_NODE"
echo "========================================================"
echo

# --- Model sizes ---
echo "=== Model Sizes ==="
echo "--- e5-base ---"
[ -f "$E5_BASE_F32" ]     && echo "  F32:          $(du -h "$E5_BASE_F32" | cut -f1)"
[ -f "$E5_BASE_I2S" ]     && echo "  I2_S:         $(du -h "$E5_BASE_I2S" | cut -f1)"
[ -f "$E5_BASE_I2S_Q6K" ] && echo "  I2_S+Q6K:     $(du -h "$E5_BASE_I2S_Q6K" | cut -f1)"
[ -f "$E5_BASE_TL2" ]     && echo "  TL2:          $(du -h "$E5_BASE_TL2" | cut -f1)"
[ -f "$E5_BASE_TL2_Q6K" ] && echo "  TL2+Q6K:      $(du -h "$E5_BASE_TL2_Q6K" | cut -f1)"
echo "--- e5-large ---"
[ -f "$E5_LARGE_F32" ]    && echo "  F32:          $(du -h "$E5_LARGE_F32" | cut -f1)"
[ -f "$E5_LARGE_I2S" ]    && echo "  I2_S:         $(du -h "$E5_LARGE_I2S" | cut -f1)"
[ -f "$E5_LARGE_I2S_Q6K" ] && echo "  I2_S+Q6K:     $(du -h "$E5_LARGE_I2S_Q6K" | cut -f1)"
[ -f "$E5_LARGE_TL2" ]    && echo "  TL2:          $(du -h "$E5_LARGE_TL2" | cut -f1)"
[ -f "$E5_LARGE_TL2_Q6K" ] && echo "  TL2+Q6K:      $(du -h "$E5_LARGE_TL2_Q6K" | cut -f1)"
echo

# ============================================================
#  e5-base (109M)
# ============================================================
echo "========================================================"
echo "  e5-base (278M params, hidden=768, intermediate=3072)"
echo "========================================================"

echo
echo "--- [e5-base] F32 ---"
[ -f "$E5_BASE_F32" ] && $TASKSET $BENCH -m "$E5_BASE_F32" $BENCH_ARGS || echo "⚠ Model not found: $E5_BASE_F32"

echo
echo "--- [e5-base] I2_S ---"
[ -f "$E5_BASE_I2S" ] && $TASKSET $BENCH -m "$E5_BASE_I2S" $BENCH_ARGS || echo "⚠ Model not found: $E5_BASE_I2S"

echo
echo "--- [e5-base] I2_S + Q6K embedding ---"
[ -f "$E5_BASE_I2S_Q6K" ] && $TASKSET $BENCH -m "$E5_BASE_I2S_Q6K" $BENCH_ARGS || echo "⚠ Model not found: $E5_BASE_I2S_Q6K"

echo
echo "--- [e5-base] TL2 ---"
if [ -f "$E5_BASE_TL2" ]; then
    $TASKSET $BENCH -m "$E5_BASE_TL2" $BENCH_ARGS || echo "⚠ TL2 crashed. Make sure kernels are generated for e5-base dimensions."
else
    echo "⚠ Model not found: $E5_BASE_TL2"
fi

echo
echo "--- [e5-base] TL2 + Q6K embedding ---"
if [ -f "$E5_BASE_TL2_Q6K" ]; then
    $TASKSET $BENCH -m "$E5_BASE_TL2_Q6K" $BENCH_ARGS || echo "⚠ TL2+Q6K crashed. Make sure kernels are generated for e5-base dimensions."
else
    echo "⚠ Model not found: $E5_BASE_TL2_Q6K"
fi

# ============================================================
#  e5-large (335M)
# ============================================================
echo
echo "========================================================"
echo "  e5-large (559M params, hidden=1024, intermediate=4096)"
echo "========================================================"

echo
echo "--- [e5-large] F32 ---"
[ -f "$E5_LARGE_F32" ] && $TASKSET $BENCH -m "$E5_LARGE_F32" $BENCH_ARGS || echo "⚠ Model not found: $E5_LARGE_F32"

echo
echo "--- [e5-large] I2_S ---"
[ -f "$E5_LARGE_I2S" ] && $TASKSET $BENCH -m "$E5_LARGE_I2S" $BENCH_ARGS || echo "⚠ Model not found: $E5_LARGE_I2S"

echo
echo "--- [e5-large] I2_S + Q6K embedding ---"
[ -f "$E5_LARGE_I2S_Q6K" ] && $TASKSET $BENCH -m "$E5_LARGE_I2S_Q6K" $BENCH_ARGS || echo "⚠ Model not found: $E5_LARGE_I2S_Q6K"

echo
echo "--- [e5-large] TL2 ---"
if [ -f "$E5_LARGE_TL2" ]; then
    $TASKSET $BENCH -m "$E5_LARGE_TL2" $BENCH_ARGS || echo "⚠ TL2 crashed. Make sure kernels are generated for e5-large dimensions."
else
    echo "⚠ Model not found: $E5_LARGE_TL2"
fi

echo
echo "--- [e5-large] TL2 + Q6K embedding ---"
if [ -f "$E5_LARGE_TL2_Q6K" ]; then
    $TASKSET $BENCH -m "$E5_LARGE_TL2_Q6K" $BENCH_ARGS || echo "⚠ TL2+Q6K crashed. Make sure kernels are generated for e5-large dimensions."
else
    echo "⚠ Model not found: $E5_LARGE_TL2_Q6K"
fi

echo
echo "========================================================"
echo "  Done. All benchmarks completed."
echo "========================================================"
