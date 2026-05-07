#!/bin/bash
# Inspect YOCO-BITNET-1.5B I2_S model: show each tensor's quantization type and size
#
# Usage: cd /home/huangxin/code_list/BitNet && bash inspect_yoco_i2s_model.sh

set -e

I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/yoco-bitnet-1.5b-i2s/ggml-model-i2_s.gguf"
F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/yoco-1.5b-f16/ggml-model-f16.gguf"
QUANTIZE="./build/bin/llama-quantize"

echo "========================================================"
echo "  YOCO-BITNET-1.5B (I2_S) Tensor Inspection"
echo "========================================================"
echo

echo "--- I2_S Model Tensor Details ---"
echo "(Requantize to F16 to show per-tensor type and size)"
echo
$QUANTIZE --allow-requantize \
  "$I2S_MODEL" \
  /dev/null \
  F16 2>&1 | grep -E "^\[|type =|^llama_model_loader: - type"

echo
echo "--- Model File Sizes ---"
echo "  YOCO-1.5B (F16):        $(du -h "$F16_MODEL" | cut -f1)"
echo "  YOCO-BITNET-1.5B (I2_S): $(du -h "$I2S_MODEL" | cut -f1)"

echo
echo "Done."
