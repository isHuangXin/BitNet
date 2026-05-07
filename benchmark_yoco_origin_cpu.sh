#!/bin/bash
# YOCO Origin Arch: F16 vs I2_S CPU Benchmark
# Models use true YOCO config: 14 layers, 24 heads, 4 kv_heads, head_dim=128
#
# Usage: cd /home/huangxin/code_list/BitNet && bash benchmark_yoco_origin_cpu.sh [threads]

set -e

F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_models_arch/yoco-1.5b-f16/ggml-model-f16.gguf"
I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_models_arch/yoco-bitnet-1.5b-i2s/ggml-model-i2_s.gguf"
BENCH="./build/bin/llama-bench"
THREADS=${1:-4}

echo "========================================================"
echo "  YOCO Origin Arch: F16 vs I2_S Benchmark"
echo "  Config: 14 layers, 24 heads, 4 kv_heads, head_dim=128"
echo "  Embedding: I2_S quantized (reduced IO)"
echo "  Threads: $THREADS"
echo "========================================================"
echo

echo "--- Model Sizes ---"
echo "  F16:  $(du -h $F16_MODEL | cut -f1)"
echo "  I2_S: $(du -h $I2S_MODEL | cut -f1)"
echo

echo "--- F16 Benchmark ---"
$BENCH -m $F16_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0

echo
echo "--- I2_S Benchmark ---"
$BENCH -m $I2S_MODEL -t $THREADS -p 128,256,512 -n 128 -r 3 -ngl 0 || echo "⚠ I2_S crashed. Try threads>=2."

echo
echo "Done."
