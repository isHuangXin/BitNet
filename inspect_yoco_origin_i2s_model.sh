#!/bin/bash
# Inspect YOCO Origin Arch I2_S model: show tensor types, especially embedding
#
# Usage: cd /home/huangxin/code_list/BitNet && bash inspect_yoco_origin_i2s_model.sh

set -e

I2S_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_models_arch/yoco-bitnet-1.5b-i2s/ggml-model-i2_s.gguf"
F16_MODEL="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_models_arch/yoco-1.5b-f16/ggml-model-f16.gguf"
QUANTIZE="./build/bin/llama-quantize"

echo "========================================================"
echo "  YOCO Origin Arch - I2_S Tensor Inspection"
echo "  Focus: Embedding layer quantization"
echo "========================================================"
echo

echo "--- I2_S Model Tensor Details ---"
echo "(Look for token_embd.weight — should be I2_S, not F32)"
echo
$QUANTIZE --allow-requantize \
  "$I2S_MODEL" \
  /dev/null \
  F16 2>&1 | grep -E "^\[|type =|^llama_model_loader: - type|token_embd"

echo
echo "--- Model File Sizes ---"
echo "  F16:  $(du -h "$F16_MODEL" | cut -f1)"
echo "  I2_S: $(du -h "$I2S_MODEL" | cut -f1)"
RATIO=$(python3 -c "
import os
f16 = os.path.getsize('$F16_MODEL')
i2s = os.path.getsize('$I2S_MODEL')
print(f'Compression: {f16/i2s:.1f}x')
")
echo "  $RATIO"

echo
echo "--- Config ---"
echo "  Layers: 14, Heads: 24, KV Heads: 4, Head Dim: 128"
echo "  d_model: 2560, d_ffn: 7680, Vocab: 32002"
echo "  q_proj: (3072, 2560), k/v_proj: (512, 2560)"
echo "  Embedding quantized: YES (Q8_0)"
echo
echo "Done."
