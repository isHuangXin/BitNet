#!/usr/bin/env python3
"""
Generate YOCO-U (Universal YOCO) dummy models with full YOCO-U architecture.

YOCO-U Architecture (from "Universal YOCO for Efficient Depth Scaling"):
  Unlike original YOCO which splits layers into two halves (self-decoder | cross-decoder),
  YOCO-U divides layers into U stages. Each stage contains:
    - A group of self-decoder layers (standard self-attention with SWA)
    - A group of cross-decoder layers (cross-attention with shared KV)
  Each stage has its own shared K^,V^ computed from the last self-decoder layer in that stage.

  This enables efficient depth scaling by adding more stages, and allows cross-decoder
  layers to attend to more recent self-decoder representations.

Default config (matching YOCO-BITNET 1.5B parameters):
  d_model=2560, d_ffn=7680, head=24, cross_head=48, kv_head=4, head_dim=128,
  n_layers=14, U=7 stages (each stage: 1 self-decoder + 1 cross-decoder layer)
  qk_norm=True, gated_attention=True, weight_tying=True, yoco_window_size=512

Stage layout (U=7, 14 layers total):
  Stage 0: layer 0 (self),  layer 1 (cross, shared KV from layer 0 output)
  Stage 1: layer 2 (self),  layer 3 (cross, shared KV from layer 2 output)
  Stage 2: layer 4 (self),  layer 5 (cross, shared KV from layer 4 output)
  Stage 3: layer 6 (self),  layer 7 (cross, shared KV from layer 6 output)
  Stage 4: layer 8 (self),  layer 9 (cross, shared KV from layer 8 output)
  Stage 5: layer 10 (self), layer 11 (cross, shared KV from layer 10 output)
  Stage 6: layer 12 (self), layer 13 (cross, shared KV from layer 12 output)

Creates:
  1. yoco-u-1.5b-f16/ggml-model-f16.gguf
  2. yoco-u-bitnet-1.5b-i2s/ggml-model-i2_s.gguf (with embedding quantized to Q8_0)

Usage:
  cd /home/huangxin/code_list/BitNet
  python generate_yoco_u_origin_arch_models.py
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

sys.path.insert(0, str(Path(__file__).parent / "3rdparty" / "llama.cpp" / "gguf-py"))
import gguf

logger = logging.getLogger("generate-yoco-u-origin-arch")

# ---------------------------------------------------------------------------
# YOCO-U config — same parameter budget as YOCO-BITNET 1.5B, but with U stages
# ---------------------------------------------------------------------------
# Stage layout: list of (n_self_layers, n_cross_layers) per stage
# Default: U=7 stages, each with 1 self + 1 cross = 14 layers total
YOCO_U_STAGES = [(1, 1)] * 7  # 7 stages x (1 self + 1 cross) = 14 layers

YOCO_U_CONFIG = {
    "hidden_size": 2560,         # d_model
    "intermediate_size": 7680,   # d_ffn
    "num_hidden_layers": 14,     # n_layers (sum of all stages)
    "num_attention_heads": 24,   # head (self-decoder)
    "num_cross_attention_heads": 48,  # cross_head (cross-decoder)
    "num_key_value_heads": 4,    # kv_head (GQA)
    "head_dim": 128,             # head_dim
    "vocab_size": 32002,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "architectures": ["BitnetForCausalLM"],
    "model_type": "bitnet",
    "torch_dtype": "float32",
    # YOCO-U specific
    "yoco_u_stages": YOCO_U_STAGES,  # [(n_self, n_cross), ...] per stage
    "yoco_u_num_stages": len(YOCO_U_STAGES),  # U
    "yoco_window_size": 512,
    "qk_norm": True,
    "gated_attention": True,
    "weight_tying": True,
}


def build_layer_map(stages):
    """Build a mapping from layer index to (stage_id, layer_type).

    Returns:
        layer_map: list of (stage_id, "self"|"cross") for each layer index
        stage_self_last: dict mapping stage_id -> last self-decoder layer index in that stage
    """
    layer_map = []
    stage_self_last = {}
    for stage_id, (n_self, n_cross) in enumerate(stages):
        last_self_idx = len(layer_map) + n_self - 1
        for _ in range(n_self):
            layer_map.append((stage_id, "self"))
        stage_self_last[stage_id] = last_self_idx
        for _ in range(n_cross):
            layer_map.append((stage_id, "cross"))
    return layer_map, stage_self_last


def create_hf_model_dir(output_dir: Path, config: dict):
    """Create a minimal HuggingFace model directory with config.json and tokenizer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Serialize stages as nested list for JSON
    cfg_json = dict(config)
    cfg_json["yoco_u_stages"] = [list(s) for s in config["yoco_u_stages"]]
    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg_json, f, indent=2)

    tokenizer_src = Path("/data2/docker-root/overlay2/93f931a377e06c53b900422e6f3d5603ec3fa954ee819bdcdb84b78134cdbfd0/merged/data1/xiaoxinyu/.cache/huggingface/hub/models--hxbgsyxh--bitnet_b1_58-3B/snapshots/54766069621e3326b126992088e60b0c16aa0aac/tokenizer.model")
    if tokenizer_src.exists():
        import shutil
        shutil.copy(tokenizer_src, output_dir / "tokenizer.model")
    else:
        logger.error(f"Tokenizer not found at {tokenizer_src}")
        sys.exit(1)

    return output_dir


def add_vocab(writer, model_dir, vocab_size):
    """Add tokenizer vocab to GGUF writer."""
    from sentencepiece import SentencePieceProcessor
    tokenizer_path = model_dir / "tokenizer.model"
    tokenizer = SentencePieceProcessor(str(tokenizer_path))

    tokens, scores, toktypes = [], [], []
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

    while len(tokens) < vocab_size:
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


def add_model_params(writer, config, name, file_type):
    """Add model hyperparameters to GGUF writer."""
    hidden = config["hidden_size"]
    n_heads_self = config["num_attention_heads"]
    n_heads_cross = config["num_cross_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    n_layers = config["num_hidden_layers"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]

    layer_map, _ = build_layer_map(stages)

    writer.add_name(name)
    writer.add_block_count(n_layers)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(config["intermediate_size"])

    # Per-layer head counts: self-decoder=24, cross-decoder=48
    n_head_arr = []
    n_head_kv_arr = []
    for i in range(n_layers):
        _, ltype = layer_map[i]
        if ltype == "self":
            n_head_arr.append(n_heads_self)
        else:
            n_head_arr.append(n_heads_cross)
        n_head_kv_arr.append(n_kv_heads)

    writer.add_head_count(n_head_arr)
    writer.add_head_count_kv(n_head_kv_arr)

    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(config["vocab_size"])
    writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    writer.add_rope_scaling_factor(1.0)
    writer.add_file_type(file_type)

    # YOCO-U specific: sliding window for self-decoder layers
    writer.add_sliding_window(config["yoco_window_size"])


def generate_self_layer_tensors(config):
    """Return dict of {tensor_name: shape} for one self-decoder layer."""
    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    q_dim = n_heads * head_dim      # 24 * 128 = 3072
    kv_dim = n_kv_heads * head_dim  # 4 * 128 = 512

    return {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (kv_dim, hidden),
        "self_attn.v_proj.weight": (kv_dim, hidden),
        "self_attn.o_proj.weight": (hidden, q_dim),
        "self_attn.inner_attn_ln.weight": (q_dim,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (ffn, hidden),
        "mlp.up_proj.weight": (ffn, hidden),
        "mlp.down_proj.weight": (hidden, ffn),
        "mlp.ffn_layernorm.weight": (ffn,),
    }


def generate_cross_layer_tensors(config):
    """Return dict of {tensor_name: shape} for one cross-decoder layer.
    Cross-decoder layers have their own Q/O and also K/V placeholders
    (the shared K^,V^ from the stage is used at runtime, but we include K/V tensors
    to satisfy the quantizer's tensor counting assertion).
    """
    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_heads_cross = config["num_cross_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]

    q_dim = n_heads_cross * head_dim  # 48 * 128 = 6144
    kv_dim = n_kv_heads * head_dim    # 4 * 128 = 512

    return {
        "input_layernorm.weight": (hidden,),
        "self_attn.q_proj.weight": (q_dim, hidden),
        "self_attn.k_proj.weight": (kv_dim, hidden),  # placeholder for quantizer
        "self_attn.v_proj.weight": (kv_dim, hidden),  # placeholder for quantizer
        "self_attn.o_proj.weight": (hidden, q_dim),
        "self_attn.inner_attn_ln.weight": (q_dim,),
        "post_attention_layernorm.weight": (hidden,),
        "mlp.gate_proj.weight": (ffn, hidden),
        "mlp.up_proj.weight": (ffn, hidden),
        "mlp.down_proj.weight": (hidden, ffn),
        "mlp.ffn_layernorm.weight": (ffn,),
    }


def generate_stage_shared_cross_tensors(config, stage_id):
    """Return dict of {tensor_name: shape} for per-stage shared cross KV tensors.

    YOCO-U difference: each stage has its own shared K^,V^ (unlike YOCO which has one global set).
    Naming: yoco_u_stage_{stage_id}_cross_kv_norm, yoco_u_stage_{stage_id}_cross_k, etc.
    """
    hidden = config["hidden_size"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    kv_dim = n_kv_heads * head_dim  # 4 * 128 = 512

    return {
        f"yoco_u_stage_{stage_id}_cross_kv_norm.weight": (hidden,),
        f"yoco_u_stage_{stage_id}_cross_k.weight": (kv_dim, hidden),
        f"yoco_u_stage_{stage_id}_cross_v.weight": (kv_dim, hidden),
    }


def generate_f16_gguf(model_dir: Path, output_path: Path, config: dict):
    """Generate GGUF f16 model with YOCO-U architecture."""
    logger.info(f"Generating YOCO-U f16 GGUF: {output_path}")

    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]

    layer_map, stage_self_last = build_layer_map(stages)

    writer = gguf.GGUFWriter(output_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])
    add_model_params(writer, config, "yoco-u-1.5b-origin-f16", gguf.GGMLQuantizationType.F16)
    add_vocab(writer, model_dir, vocab)

    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

    # Embedding - f16
    data = np.random.randn(vocab, hidden).astype(np.float16)
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)
    logger.info(f"  {name}: {data.shape} -> f16")

    # Per-layer tensors
    for i in range(n_layers):
        stage_id, ltype = layer_map[i]
        is_cross = (ltype == "cross")
        layer_template = generate_cross_layer_tensors(config) if is_cross else generate_self_layer_tensors(config)

        for suffix, shape in layer_template.items():
            tensor_name = f"model.layers.{i}.{suffix}"
            data = np.random.randn(*shape).astype(np.float32)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                logger.warning(f"  Skipping unmapped: {tensor_name}")
                continue
            n_dims = len(shape)
            if n_dims >= 2:
                data = data.astype(np.float16)
            writer.add_tensor(mapped, data)
            logger.info(f"  {mapped}: {shape} -> {data.dtype} (stage {stage_id}, {ltype})")

    # Per-stage shared cross KV tensors
    for stage_id in range(n_stages):
        shared_tensors = generate_stage_shared_cross_tensors(config, stage_id)
        for tname, shape in shared_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            if len(shape) >= 2:
                data = data.astype(np.float16)
            writer.add_tensor(tname, data)
            logger.info(f"  {tname}: {shape} -> {data.dtype} (stage {stage_id} shared cross KV)")

    # Final norm
    data = np.random.randn(hidden).astype(np.float32)
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_gb = output_path.stat().st_size / (1024**3)
    logger.info(f"  f16 model saved: {output_path} ({size_gb:.2f} GB)")

    # Count params
    total_params = vocab * hidden  # embedding
    for i in range(n_layers):
        _, ltype = layer_map[i]
        lt = generate_cross_layer_tensors(config) if ltype == "cross" else generate_self_layer_tensors(config)
        for shape in lt.values():
            total_params += int(np.prod(shape))
    for stage_id in range(n_stages):
        shared = generate_stage_shared_cross_tensors(config, stage_id)
        for shape in shared.values():
            total_params += int(np.prod(shape))
    total_params += hidden  # final norm
    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e9:.3f}B)")


def weight_quant_ternary(weight: np.ndarray) -> np.ndarray:
    """Quantize weight to ternary {-1, 0, 1}."""
    w = weight.astype(np.float32)
    s = 1.0 / max(np.abs(w).mean(), 1e-5)
    return np.clip(np.round(w * s), -1, 1) / s


def generate_i2s_gguf(model_dir: Path, output_path: Path, config: dict):
    """Generate f32 GGUF with ternary weights, then quantize to i2_s."""
    logger.info(f"Generating YOCO-U f32 GGUF for i2_s quantization: {output_path}")

    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]

    layer_map, stage_self_last = build_layer_map(stages)

    f32_path = output_path.parent / "ggml-model-f32.gguf"
    writer = gguf.GGUFWriter(f32_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])
    add_model_params(writer, config, "yoco-u-bitnet-1.5b-origin", gguf.GGMLQuantizationType.F32)
    add_vocab(writer, model_dir, vocab)

    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

    # Embedding - f32 (will be quantized to Q8_0 by llama-quantize)
    data = np.random.randn(vocab, hidden).astype(np.float32)
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    quant_names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

    for i in range(n_layers):
        stage_id, ltype = layer_map[i]
        is_cross = (ltype == "cross")
        layer_template = generate_cross_layer_tensors(config) if is_cross else generate_self_layer_tensors(config)

        for suffix, shape in layer_template.items():
            tensor_name = f"model.layers.{i}.{suffix}"
            data = np.random.randn(*shape).astype(np.float32)
            should_quant = any(qn in tensor_name for qn in quant_names)
            if should_quant and len(shape) == 2:
                data = weight_quant_ternary(data)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                continue
            writer.add_tensor(mapped, data)

    # Per-stage shared cross KV tensors (ternary quantized)
    for stage_id in range(n_stages):
        shared_tensors = generate_stage_shared_cross_tensors(config, stage_id)
        for tname, shape in shared_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            if len(shape) == 2:
                data = weight_quant_ternary(data)
            writer.add_tensor(tname, data)

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

    # Quantize to i2_s with Q8_0 embedding
    quantize_bin = Path(__file__).parent / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        logger.error(f"llama-quantize not found at {quantize_bin}")
        sys.exit(1)

    import subprocess
    logger.info(f"  Quantizing f32 -> i2_s (with embedding Q8_0) ...")
    cmd = [
        str(quantize_bin),
        "--token-embedding-type", "Q8_0",
        str(f32_path),
        str(output_path),
        "I2_S",
        "1",
    ]
    logger.info(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Quantization failed:\n{result.stderr}\n{result.stdout}")
        sys.exit(1)
    logger.info(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    i2s_size = output_path.stat().st_size / (1024**3)
    logger.info(f"  i2_s model saved: {output_path} ({i2s_size:.2f} GB)")
    logger.info(f"  Compression ratio: {f32_size/i2s_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Generate YOCO-U origin arch dummy BitNet models")
    parser.add_argument("--output-dir", type=str,
                        default="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch",
                        help="Output directory")
    parser.add_argument("--stages", type=str, default=None,
                        help="Stage layout as comma-separated pairs, e.g. '1:1,1:1,1:1,...' "
                             "meaning (n_self:n_cross) per stage. Default: 7 stages of 1:1")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Parse custom stages if provided
    if args.stages:
        stages = []
        for pair in args.stages.split(","):
            n_self, n_cross = pair.strip().split(":")
            stages.append((int(n_self), int(n_cross)))
        YOCO_U_CONFIG["yoco_u_stages"] = stages
        YOCO_U_CONFIG["yoco_u_num_stages"] = len(stages)
        total_layers = sum(s + c for s, c in stages)
        YOCO_U_CONFIG["num_hidden_layers"] = total_layers
        logger.info(f"Custom stages: {stages}, total layers: {total_layers}")

    config = YOCO_U_CONFIG
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]
    n_layers = config["num_hidden_layers"]
    layer_map, stage_self_last = build_layer_map(stages)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    # Create HF dir (for tokenizer)
    hf_dir = output_base / "yoco-u-1.5b-origin-hf"
    create_hf_model_dir(hf_dir, config)
    logger.info(f"HF model dir: {hf_dir}")

    # Generate f16 model
    f16_dir = output_base / "yoco-u-1.5b-f16"
    f16_dir.mkdir(parents=True, exist_ok=True)
    f16_path = f16_dir / "ggml-model-f16.gguf"
    generate_f16_gguf(hf_dir, f16_path, config)

    # Generate i2_s model
    i2s_dir = output_base / "yoco-u-bitnet-1.5b-i2s"
    i2s_dir.mkdir(parents=True, exist_ok=True)
    i2s_path = i2s_dir / "ggml-model-i2_s.gguf"
    generate_i2s_gguf(hf_dir, i2s_path, config)

    print("\n" + "=" * 70)
    print("YOCO-U Origin Arch Models Generated (Full YOCO-U Architecture)!")
    print(f"  YOCO-U-1.5B (f16):          {f16_path}")
    print(f"  YOCO-U-BITNET-1.5B (i2_s):  {i2s_path}")
    print("=" * 70)
    print(f"\nYOCO-U Config Used:")
    print(f"  d_model={config['hidden_size']}, d_ffn={config['intermediate_size']}")
    print(f"  n_layers={n_layers}, U={n_stages} stages")
    print(f"  Self-decoder: head={config['num_attention_heads']}, kv_head={config['num_key_value_heads']}, head_dim={config['head_dim']}")
    print(f"    q_proj shape: ({config['num_attention_heads']*config['head_dim']}, {config['hidden_size']}) = (3072, 2560)")
    print(f"    k/v_proj shape: ({config['num_key_value_heads']*config['head_dim']}, {config['hidden_size']}) = (512, 2560)")
    print(f"    SWA window_size={config['yoco_window_size']}")
    print(f"  Cross-decoder: cross_head={config['num_cross_attention_heads']}, shared kv_head={config['num_key_value_heads']}")
    print(f"    q_proj shape: ({config['num_cross_attention_heads']*config['head_dim']}, {config['hidden_size']}) = (6144, 2560)")
    print(f"  Embedding quantized in i2_s: Q8_0")
    print(f"\n  Stage layout (YOCO-U interleaved):")
    layer_idx = 0
    for stage_id, (n_self, n_cross) in enumerate(stages):
        self_layers = list(range(layer_idx, layer_idx + n_self))
        cross_layers = list(range(layer_idx + n_self, layer_idx + n_self + n_cross))
        print(f"    Stage {stage_id}: self layers {self_layers}, cross layers {cross_layers}")
        print(f"      Shared K^,V^ computed from layer {self_layers[-1]} output")
        layer_idx += n_self + n_cross


if __name__ == "__main__":
    main()
