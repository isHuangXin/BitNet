#!/usr/bin/env python3
"""
Generate YOCO-1.5B equivalent dummy models in GGUF format for CPU inference benchmarking.

Creates two models:
  1. YOCO-1.5B (f16) — bf16 baseline, stored as GGUF f16
  2. YOCO-BITNET-1.5B (i2_s) — ternary quantized, stored as GGUF i2_s with LUT kernel

Architecture mapping:
  YOCO-1.5B (d_model=2560, d_ffn=7680, 14 layers, 1.857B params)
  → Standard BitnetForCausalLM (hidden=2560, ffn=7680, 21 layers, ~1.871B params)

Usage:
  cd /home/huangxin/code_list/BitNet
  python generate_yoco_dummy_models.py
"""
from __future__ import annotations
import sys
import os
import json
import argparse
import logging
from pathlib import Path

import numpy as np
import torch

# Use BitNet's custom gguf
sys.path.insert(0, str(Path(__file__).parent / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf

logger = logging.getLogger("generate-yoco-models")

# YOCO-1.5B equivalent config for standard BitNet architecture
# Original YOCO: d_model=2560, d_ffn=7680, 14 layers (7 self + 7 cross), 1.857B params
# Mapped: hidden=2560, ffn=7680, 21 layers (all standard decoder), ~1.871B params
YOCO_EQUIV_CONFIG = {
    "hidden_size": 2560,
    "intermediate_size": 7680,
    "num_hidden_layers": 21,
    "num_attention_heads": 20,       # 2560 / 128 = 20 heads
    "num_key_value_heads": 20,       # no GQA in standard BitNet
    "vocab_size": 32002,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "architectures": ["BitnetForCausalLM"],
    "model_type": "bitnet",
    "torch_dtype": "float32",
}


def create_hf_model_dir(output_dir: Path, config: dict):
    """Create a minimal HuggingFace model directory with config.json and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config.json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer from cached BitNet-3B model
    tokenizer_src = Path("/data2/docker-root/overlay2/93f931a377e06c53b900422e6f3d5603ec3fa954ee819bdcdb84b78134cdbfd0/merged/data1/xiaoxinyu/.cache/huggingface/hub/models--hxbgsyxh--bitnet_b1_58-3B/snapshots/54766069621e3326b126992088e60b0c16aa0aac/tokenizer.model")
    if tokenizer_src.exists():
        import shutil
        shutil.copy(tokenizer_src, output_dir / "tokenizer.model")
    else:
        logger.error(f"Tokenizer not found at {tokenizer_src}")
        sys.exit(1)

    return output_dir


def generate_f16_gguf(model_dir: Path, output_path: Path, config: dict):
    """Generate GGUF f16 model (YOCO-1.5B bf16 equivalent)."""
    logger.info(f"Generating f16 GGUF model: {output_path}")

    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    vocab = config["vocab_size"]
    head_dim = hidden // n_heads

    writer = gguf.GGUFWriter(output_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])

    # Set model parameters
    writer.add_name("yoco-1.5b-f16")
    writer.add_block_count(n_layers)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(vocab)
    writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    writer.add_rope_scaling_factor(1.0)
    writer.add_file_type(gguf.GGMLQuantizationType.F16)

    # Set vocab (sentencepiece)
    from sentencepiece import SentencePieceProcessor
    tokenizer_path = model_dir / "tokenizer.model"
    tokenizer = SentencePieceProcessor(str(tokenizer_path))

    tokens = []
    scores = []
    toktypes = []
    for i in range(tokenizer.vocab_size()):
        tokens.append(tokenizer.id_to_piece(i).encode("utf-8"))
        scores.append(tokenizer.get_score(i))
        if tokenizer.is_unknown(i):
            toktypes.append(gguf.TokenType.UNKNOWN)
        elif tokenizer.is_control(i):
            toktypes.append(gguf.TokenType.CONTROL)
        elif tokenizer.is_unused(i):
            toktypes.append(gguf.TokenType.UNUSED)
        elif tokenizer.is_byte(i):
            toktypes.append(gguf.TokenType.BYTE)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

    # Pad to vocab_size
    while len(tokens) < vocab:
        tokens.append(f"[PAD{len(tokens)}]".encode())
        scores.append(-1000.0)
        toktypes.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, n_vocab=len(tokens))
    special_vocab.add_to_gguf(writer)

    # Generate random tensors
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

    # Embedding
    data = np.random.randn(vocab, hidden).astype(np.float32)
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)
    logger.info(f"  {name}: {data.shape} -> f32")

    for i in range(n_layers):
        layer_tensors = {
            f"model.layers.{i}.input_layernorm.weight": (hidden,),
            f"model.layers.{i}.self_attn.q_proj.weight": (n_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.k_proj.weight": (n_kv_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.v_proj.weight": (n_kv_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.o_proj.weight": (hidden, n_heads * head_dim),
            f"model.layers.{i}.self_attn.inner_attn_ln.weight": (hidden,),
            f"model.layers.{i}.post_attention_layernorm.weight": (hidden,),
            f"model.layers.{i}.mlp.gate_proj.weight": (ffn, hidden),
            f"model.layers.{i}.mlp.up_proj.weight": (ffn, hidden),
            f"model.layers.{i}.mlp.down_proj.weight": (hidden, ffn),
            f"model.layers.{i}.mlp.ffn_layernorm.weight": (ffn,),
        }
        for tensor_name, shape in layer_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                logger.warning(f"  Skipping unmapped tensor: {tensor_name}")
                continue

            n_dims = len(shape)
            # For f16: 2D weight tensors -> f16, 1D norms -> f32
            if n_dims >= 2 and not mapped.endswith("_norm.weight"):
                data = data.astype(np.float16)
            writer.add_tensor(mapped, data)
            logger.info(f"  {mapped}: {shape} -> {data.dtype}")

    # Final norm
    data = np.random.randn(hidden).astype(np.float32)
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = output_path.stat().st_size / (1024**3)
    logger.info(f"  f16 model saved: {output_path} ({file_size:.2f} GB)")


def weight_quant_ternary(weight: np.ndarray) -> np.ndarray:
    """Quantize weight to ternary {-1, 0, 1} using per-tensor scale."""
    w = weight.astype(np.float32)
    s = 1.0 / max(np.abs(w).mean(), 1e-5)
    return np.clip(np.round(w * s), -1, 1) / s


def generate_i2s_gguf(model_dir: Path, output_path: Path, config: dict):
    """Generate GGUF f32 model first, then quantize to i2_s using llama-quantize."""
    logger.info(f"Generating f32 GGUF model for i2_s quantization: {output_path}")

    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_layers = config["num_hidden_layers"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    vocab = config["vocab_size"]
    head_dim = hidden // n_heads

    # First generate f32 model
    f32_path = output_path.parent / "ggml-model-f32.gguf"
    writer = gguf.GGUFWriter(f32_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])

    writer.add_name("yoco-bitnet-1.5b")
    writer.add_block_count(n_layers)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(ffn)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(vocab)
    writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    writer.add_rope_scaling_factor(1.0)
    writer.add_file_type(gguf.GGMLQuantizationType.F32)

    # Set vocab
    from sentencepiece import SentencePieceProcessor
    tokenizer_path = model_dir / "tokenizer.model"
    tokenizer = SentencePieceProcessor(str(tokenizer_path))

    tokens = []
    scores = []
    toktypes = []
    for i in range(tokenizer.vocab_size()):
        tokens.append(tokenizer.id_to_piece(i).encode("utf-8"))
        scores.append(tokenizer.get_score(i))
        if tokenizer.is_unknown(i):
            toktypes.append(gguf.TokenType.UNKNOWN)
        elif tokenizer.is_control(i):
            toktypes.append(gguf.TokenType.CONTROL)
        elif tokenizer.is_unused(i):
            toktypes.append(gguf.TokenType.UNUSED)
        elif tokenizer.is_byte(i):
            toktypes.append(gguf.TokenType.BYTE)
        else:
            toktypes.append(gguf.TokenType.NORMAL)

    while len(tokens) < vocab:
        tokens.append(f"[PAD{len(tokens)}]".encode())
        scores.append(-1000.0)
        toktypes.append(gguf.TokenType.UNUSED)

    writer.add_tokenizer_model("llama")
    writer.add_tokenizer_pre("default")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(toktypes)

    special_vocab = gguf.SpecialVocab(model_dir, n_vocab=len(tokens))
    special_vocab.add_to_gguf(writer)

    # Generate random tensors with ternary quantization applied
    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

    # Embedding (not quantized)
    data = np.random.randn(vocab, hidden).astype(np.float32)
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    for i in range(n_layers):
        layer_tensors = {
            f"model.layers.{i}.input_layernorm.weight": (hidden,),
            f"model.layers.{i}.self_attn.q_proj.weight": (n_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.k_proj.weight": (n_kv_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.v_proj.weight": (n_kv_heads * head_dim, hidden),
            f"model.layers.{i}.self_attn.o_proj.weight": (hidden, n_heads * head_dim),
            f"model.layers.{i}.self_attn.inner_attn_ln.weight": (hidden,),
            f"model.layers.{i}.post_attention_layernorm.weight": (hidden,),
            f"model.layers.{i}.mlp.gate_proj.weight": (ffn, hidden),
            f"model.layers.{i}.mlp.up_proj.weight": (ffn, hidden),
            f"model.layers.{i}.mlp.down_proj.weight": (hidden, ffn),
            f"model.layers.{i}.mlp.ffn_layernorm.weight": (ffn,),
        }
        quant_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        for tensor_name, shape in layer_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            # Apply ternary quantization to weight matrices
            should_quant = any(qn in tensor_name for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                continue
            writer.add_tensor(mapped, data)

    # Final norm
    data = np.random.randn(hidden).astype(np.float32)
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    f32_size = f32_path.stat().st_size / (1024**3)
    logger.info(f"  f32 model saved: {f32_path} ({f32_size:.2f} GB)")

    # Quantize to i2_s using llama-quantize
    quantize_bin = Path(__file__).parent / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        logger.error(f"llama-quantize not found at {quantize_bin}")
        sys.exit(1)

    import subprocess
    logger.info(f"  Quantizing f32 -> i2_s ...")
    cmd = [str(quantize_bin), str(f32_path), str(output_path), "I2_S", "1"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Quantization failed:\n{result.stderr}")
        sys.exit(1)

    i2s_size = output_path.stat().st_size / (1024**3)
    logger.info(f"  i2_s model saved: {output_path} ({i2s_size:.2f} GB)")
    logger.info(f"  Compression ratio: {f32_size/i2s_size:.1f}x")

    # Clean up f32
    # f32_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Generate YOCO-equivalent dummy BitNet models")
    parser.add_argument("--output-dir", type=str,
                        default="/data2/huangxin/model_list/Yoco-YocoBitNet-Project",
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_base = Path(args.output_dir)

    # Create HF model directory (for tokenizer/config)
    hf_dir = output_base / "yoco-1.5b-hf"
    create_hf_model_dir(hf_dir, YOCO_EQUIV_CONFIG)
    logger.info(f"HF model dir created: {hf_dir}")

    # Generate f16 model (YOCO-1.5B bf16 equivalent)
    f16_dir = output_base / "yoco-1.5b-f16"
    f16_dir.mkdir(parents=True, exist_ok=True)
    f16_path = f16_dir / "ggml-model-f16.gguf"
    generate_f16_gguf(hf_dir, f16_path, YOCO_EQUIV_CONFIG)

    # Generate i2_s model (YOCO-BITNET-1.5B equivalent)
    i2s_dir = output_base / "yoco-bitnet-1.5b-i2s"
    i2s_dir.mkdir(parents=True, exist_ok=True)
    i2s_path = i2s_dir / "ggml-model-i2_s.gguf"
    generate_i2s_gguf(hf_dir, i2s_path, YOCO_EQUIV_CONFIG)

    print("\n" + "=" * 70)
    print("Models generated successfully!")
    print(f"  YOCO-1.5B (f16):       {f16_path}")
    print(f"  YOCO-BITNET-1.5B (i2_s): {i2s_path}")
    print("=" * 70)
    print("\nRun benchmarks:")
    print(f"  # f16 baseline")
    print(f"  ./build/bin/llama-bench -m {f16_path} -t 4 -n 128 -p 512")
    print(f"  # i2_s BitNet (LUT kernel)")
    print(f"  ./build/bin/llama-bench -m {i2s_path} -t 4 -n 128 -p 512")


if __name__ == "__main__":
    main()
