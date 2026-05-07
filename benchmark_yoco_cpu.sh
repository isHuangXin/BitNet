#!/bin/bash
# YOCO-1.5B vs YOCO-BITNET-1.5B CPU Inference Benchmark
# Using BitNet's llama.cpp with LUT kernel for i2_s
#
# Usage: cd /home/huangxin/code_list/BitNet && bash benchmark_yoco_cpu.sh [threads]
# Note: I2_S LUT kernel has a known segfault with threads=1, minimum 2 threads recommended.

set -e

F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/yoco-1.5b-f16/ggml-model-f16.gguf"
I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/yoco-bitnet-1.5b-i2s/ggml-model-i2_s.gguf"
BENCH="./build/bin/llama-bench"
THREADS=${1:-4}

echo "========================================================"
echo "  YOCO-1.5B (f16) vs YOCO-BITNET-1.5B (i2_s) Benchmark"
echo "  Threads: $THREADS"
echo "========================================================"
echo

if [ "$THREADS" -lt 2 ]; then
  echo "⚠ WARNING: I2_S LUT kernel may segfault with threads=1."
  echo "  Running F16 and I2_S separately to avoid losing results."
  echo
fi

echo "--- Model Sizes ---"
echo "  YOCO-1.5B (f16):        $(du -h $F16_MODEL | cut -f1)"
echo "  YOCO-BITNET-1.5B (i2_s): $(du -h $I2S_MODEL | cut -f1)"
echo

# Run models separately so one crash doesn't lose the other's results
echo "--- F16 Benchmark ---"
$BENCH \
  -m $F16_MODEL \
  -t $THREADS \
  -p 128,256,512 \
  -n 128 \
  -r 3 \
  -ngl 0

echo
echo "--- I2_S Benchmark ---"
$BENCH \
  -m $I2S_MODEL \
  -t $THREADS \
  -p 128,256,512 \
  -n 128 \
  -r 3 \
  -ngl 0 || echo "⚠ I2_S benchmark crashed (likely LUT kernel segfault with threads=$THREADS). Try threads>=2."

echo
echo "Done."
