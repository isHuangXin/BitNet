#!/usr/bin/env python3
"""
Generate tiny YOCO-U models that fit entirely in CPU L3 Cache (single NUMA node).

Target: Intel Xeon Platinum 8358P, L3 = 48 MiB per NUMA node.
  - F16 model:  ~30 MiB  (fits in 48 MiB L3)
  - I2_S model: ~5 MiB   (easily fits in L3)

Tiny YOCO-U config:
  d_model=256, d_ffn=768, head=2, cross_head=4, kv_head=1, head_dim=128,
  n_layers=8 (4 stages x 2), yoco_window_size=512

Stage layout (U=4, 8 layers total):
  Stage 0: layer 0 (self),  layer 1 (cross)
  Stage 1: layer 2 (self),  layer 3 (cross)
  Stage 2: layer 4 (self),  layer 5 (cross)
  Stage 3: layer 6 (self),  layer 7 (cross)

Usage:
  cd /home/huangxin/code_list/BitNet
  python generate_yoco_u_L3_cache_models.py
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

logger = logging.getLogger("generate-yoco-u-L3")

# ---------------------------------------------------------------------------
# Tiny YOCO-U config — designed to fit in L3 cache (48 MiB per NUMA node)
# ---------------------------------------------------------------------------
YOCO_U_STAGES = [(1, 1)] * 4  # 4 stages x (1 self + 1 cross) = 8 layers

YOCO_U_CONFIG = {
    "hidden_size": 256,          # d_model (small)
    "intermediate_size": 768,    # d_ffn = 3x d_model
    "num_hidden_layers": 8,      # n_layers
    "num_attention_heads": 2,    # head (self-decoder)
    "num_cross_attention_heads": 4,  # cross_head (cross-decoder)
    "num_key_value_heads": 1,    # kv_head (GQA)
    "head_dim": 128,             # head_dim
    "vocab_size": 32002,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "architectures": ["BitnetForCausalLM"],
    "model_type": "bitnet",
    "torch_dtype": "float32",
    # YOCO-U specific
    "yoco_u_stages": YOCO_U_STAGES,
    "yoco_u_num_stages": len(YOCO_U_STAGES),
    "yoco_window_size": 512,
    "qk_norm": True,
    "gated_attention": True,
    "weight_tying": True,
}


def build_layer_map(stages):
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
    output_dir.mkdir(parents=True, exist_ok=True)
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
    hidden = config["hidden_size"]
    n_heads_self = config["num_attention_heads"]
    n_heads_cross = config["num_cross_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    n_layers = config["num_hidden_layers"]
    stages = config["yoco_u_stages"]

    layer_map, _ = build_layer_map(stages)

    writer.add_name(name)
    writer.add_block_count(n_layers)
    writer.add_context_length(config["max_position_embeddings"])
    writer.add_embedding_length(hidden)
    writer.add_feed_forward_length(config["intermediate_size"])

    n_head_arr = []
    n_head_kv_arr = []
    for i in range(n_layers):
        _, ltype = layer_map[i]
        n_head_arr.append(n_heads_cross if ltype == "cross" else n_heads_self)
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
    writer.add_sliding_window(config["yoco_window_size"])


def generate_self_layer_tensors(config):
    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

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
    hidden = config["hidden_size"]
    ffn = config["intermediate_size"]
    n_heads_cross = config["num_cross_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    q_dim = n_heads_cross * head_dim
    kv_dim = n_kv_heads * head_dim

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


def generate_stage_shared_cross_tensors(config, stage_id):
    hidden = config["hidden_size"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config["head_dim"]
    kv_dim = n_kv_heads * head_dim

    return {
        f"yoco_u_stage_{stage_id}_cross_kv_norm.weight": (hidden,),
        f"yoco_u_stage_{stage_id}_cross_k.weight": (kv_dim, hidden),
        f"yoco_u_stage_{stage_id}_cross_v.weight": (kv_dim, hidden),
    }


def estimate_model_size(config):
    """Estimate model size in bytes for F16 and I2_S."""
    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]
    layer_map, _ = build_layer_map(stages)

    total_params = vocab * hidden  # embedding
    for i in range(n_layers):
        _, ltype = layer_map[i]
        lt = generate_cross_layer_tensors(config) if ltype == "cross" else generate_self_layer_tensors(config)
        for shape in lt.values():
            total_params += int(np.prod(shape))
    for s in range(n_stages):
        shared = generate_stage_shared_cross_tensors(config, s)
        for shape in shared.values():
            total_params += int(np.prod(shape))
    total_params += hidden  # final norm

    f16_bytes = total_params * 2
    # I2_S: 2-bit weights (~0.25 bytes) for 2D tensors, f32 for 1D norms, Q8_0 for embedding
    i2s_bytes = vocab * hidden  # Q8_0 embedding ~1 byte/param
    for i in range(n_layers):
        _, ltype = layer_map[i]
        lt = generate_cross_layer_tensors(config) if ltype == "cross" else generate_self_layer_tensors(config)
        for shape in lt.values():
            n = int(np.prod(shape))
            if len(shape) == 2:
                i2s_bytes += n // 4  # ~2 bits per param
            else:
                i2s_bytes += n * 4   # f32 norm
    for s in range(n_stages):
        shared = generate_stage_shared_cross_tensors(config, s)
        for shape in shared.values():
            n = int(np.prod(shape))
            if len(shape) == 2:
                i2s_bytes += n // 4
            else:
                i2s_bytes += n * 4

    return total_params, f16_bytes, i2s_bytes


def generate_f16_gguf(model_dir: Path, output_path: Path, config: dict):
    logger.info(f"Generating YOCO-U L3 f16 GGUF: {output_path}")

    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]
    layer_map, _ = build_layer_map(stages)

    writer = gguf.GGUFWriter(output_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])
    add_model_params(writer, config, "yoco-u-L3-f16", gguf.GGMLQuantizationType.F16)
    add_vocab(writer, model_dir, vocab)

    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

    # Embedding
    data = np.random.randn(vocab, hidden).astype(np.float16)
    name = tensor_map.get_name("model.embed_tokens.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)
    logger.info(f"  {name}: {data.shape} -> f16")

    for i in range(n_layers):
        stage_id, ltype = layer_map[i]
        is_cross = (ltype == "cross")
        layer_template = generate_cross_layer_tensors(config) if is_cross else generate_self_layer_tensors(config)

        for suffix, shape in layer_template.items():
            tensor_name = f"model.layers.{i}.{suffix}"
            data = np.random.randn(*shape).astype(np.float32)
            mapped = tensor_map.get_name(tensor_name, try_suffixes=(".weight",))
            if mapped is None:
                continue
            if len(shape) >= 2:
                data = data.astype(np.float16)
            writer.add_tensor(mapped, data)
            logger.info(f"  {mapped}: {shape} -> {data.dtype} (stage {stage_id}, {ltype})")

    for stage_id in range(n_stages):
        shared_tensors = generate_stage_shared_cross_tensors(config, stage_id)
        for tname, shape in shared_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            if len(shape) >= 2:
                data = data.astype(np.float16)
            writer.add_tensor(tname, data)
            logger.info(f"  {tname}: {shape} -> {data.dtype}")

    data = np.random.randn(hidden).astype(np.float32)
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_path.stat().st_size / (1024**2)
    logger.info(f"  f16 model saved: {output_path} ({size_mb:.1f} MiB)")

    total_params, _, _ = estimate_model_size(config)
    logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")


def weight_quant_ternary(weight: np.ndarray) -> np.ndarray:
    w = weight.astype(np.float32)
    s = 1.0 / max(np.abs(w).mean(), 1e-5)
    return np.clip(np.round(w * s), -1, 1) / s


def generate_i2s_gguf(model_dir: Path, output_path: Path, config: dict):
    logger.info(f"Generating YOCO-U L3 f32 GGUF for i2_s quantization: {output_path}")

    hidden = config["hidden_size"]
    n_layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]
    layer_map, _ = build_layer_map(stages)

    f32_path = output_path.parent / "ggml-model-f32.gguf"
    writer = gguf.GGUFWriter(f32_path, gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.BITNET])
    add_model_params(writer, config, "yoco-u-L3-bitnet", gguf.GGMLQuantizationType.F32)
    add_vocab(writer, model_dir, vocab)

    tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.BITNET, n_layers)

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

    for stage_id in range(n_stages):
        shared_tensors = generate_stage_shared_cross_tensors(config, stage_id)
        for tname, shape in shared_tensors.items():
            data = np.random.randn(*shape).astype(np.float32)
            if len(shape) == 2:
                data = weight_quant_ternary(data)
            writer.add_tensor(tname, data)

    data = np.random.randn(hidden).astype(np.float32)
    name = tensor_map.get_name("model.norm.weight", try_suffixes=(".weight",))
    writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    f32_size = f32_path.stat().st_size / (1024**2)
    logger.info(f"  f32 model saved: {f32_path} ({f32_size:.1f} MiB)")

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

    i2s_size = output_path.stat().st_size / (1024**2)
    logger.info(f"  i2_s model saved: {output_path} ({i2s_size:.1f} MiB)")
    logger.info(f"  Compression ratio: {f32_size/i2s_size:.1f}x")


def main():
    parser = argparse.ArgumentParser(description="Generate tiny YOCO-U models for L3 cache benchmark")
    parser.add_argument("--output-dir", type=str,
                        default="/data2/huangxin/model_list/Yoco-YocoBitNet-Project/origin_yoco_u_models_arch_L3",
                        help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = YOCO_U_CONFIG

    # Print estimated sizes
    total_params, f16_bytes, i2s_bytes = estimate_model_size(config)
    l3_per_numa = 48  # MiB
    print("=" * 60)
    print("  YOCO-U L3 Cache Model Size Estimation")
    print(f"  L3 Cache per NUMA node: {l3_per_numa} MiB")
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"  Estimated F16 size: {f16_bytes/1024**2:.1f} MiB {'✅' if f16_bytes/1024**2 < l3_per_numa else '❌'}")
    print(f"  Estimated I2_S size: {i2s_bytes/1024**2:.1f} MiB {'✅' if i2s_bytes/1024**2 < l3_per_numa else '❌'}")
    print("=" * 60)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    hf_dir = output_base / "yoco-u-L3-hf"
    create_hf_model_dir(hf_dir, config)
    logger.info(f"HF model dir: {hf_dir}")

    f16_dir = output_base / "yoco-u-L3-f16"
    f16_dir.mkdir(parents=True, exist_ok=True)
    f16_path = f16_dir / "ggml-model-f16.gguf"
    generate_f16_gguf(hf_dir, f16_path, config)

    i2s_dir = output_base / "yoco-u-L3-bitnet-i2s"
    i2s_dir.mkdir(parents=True, exist_ok=True)
    i2s_path = i2s_dir / "ggml-model-i2_s.gguf"
    generate_i2s_gguf(hf_dir, i2s_path, config)

    stages = config["yoco_u_stages"]
    n_stages = config["yoco_u_num_stages"]
    n_layers = config["num_hidden_layers"]

    f16_actual = f16_path.stat().st_size / (1024**2)
    i2s_actual = i2s_path.stat().st_size / (1024**2)

    print("\n" + "=" * 60)
    print("YOCO-U L3 Cache Models Generated!")
    print(f"  YOCO-U-L3 (f16):          {f16_path} ({f16_actual:.1f} MiB)")
    print(f"  YOCO-U-L3-BITNET (i2_s):  {i2s_path} ({i2s_actual:.1f} MiB)")
    print("=" * 60)
    print(f"\n  L3 per NUMA: {l3_per_numa} MiB")
    print(f"  F16 fits in L3:  {'✅ YES' if f16_actual < l3_per_numa else '❌ NO'} ({f16_actual:.1f}/{l3_per_numa} MiB)")
    print(f"  I2_S fits in L3: {'✅ YES' if i2s_actual < l3_per_numa else '❌ NO'} ({i2s_actual:.1f}/{l3_per_numa} MiB)")
    print(f"\n  Config: d_model={config['hidden_size']}, d_ffn={config['intermediate_size']}")
    print(f"  n_layers={n_layers}, U={n_stages} stages")
    print(f"  Self head={config['num_attention_heads']}, Cross head={config['num_cross_attention_heads']}")
    print(f"  kv_head={config['num_key_value_heads']}, head_dim={config['head_dim']}")
    print(f"  Total params: {total_params:,} ({total_params/1e6:.2f}M)")


if __name__ == "__main__":
    main()
