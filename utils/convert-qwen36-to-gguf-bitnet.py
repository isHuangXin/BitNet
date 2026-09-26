#!/usr/bin/env python3
"""Convert the Qwen3.6 MoE text backbone to native row-scaled I2_S GGUF.

Large text matrices, including embeddings and the output head, are ternary.
Routers, shared-expert gates, GDN alpha/beta, convolutions, norms, biases and
A/dt parameters stay F32. Vision and MTP are excluded. I2_S stores two bits
per weight plus row scales; conversion alone does not guarantee model quality.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

LLAMA_CPP = Path(__file__).resolve().parents[1] / "3rdparty" / "llama.cpp"
sys.path.insert(0, str(LLAMA_CPP))
sys.path.insert(0, str(LLAMA_CPP / "gguf-py"))

import gguf
from conversion.base import LazyTorchTensor
from conversion.qwen import Qwen3_5MoeTextModel

logger = logging.getLogger("qwen36-bitnet")
I2_S_LAYOUT = "i2_s_row_scale_v1"
T = gguf.MODEL_TENSOR
EXPERT_ROLES = {T.FFN_GATE_EXP, T.FFN_UP_EXP, T.FFN_DOWN_EXP}
I2_S_ROLES = EXPERT_ROLES | {
    T.FFN_GATE_SHEXP, T.FFN_UP_SHEXP, T.FFN_DOWN_SHEXP,
    T.ATTN_Q, T.ATTN_K, T.ATTN_V, T.ATTN_OUT,
    T.ATTN_QKV, T.ATTN_GATE, T.SSM_OUT,
    T.TOKEN_EMBD, T.OUTPUT,
}
F32_ROLES = {
    T.FFN_GATE_INP, T.FFN_GATE_INP_SHEXP,
    T.SSM_ALPHA, T.SSM_BETA, T.SSM_CONV1D, T.SSM_A, T.SSM_DT, T.SSM_NORM,
    T.OUTPUT_NORM, T.ATTN_NORM, T.ATTN_POST_NORM, T.ATTN_Q_NORM, T.ATTN_K_NORM,
}


def quantize_to_i2_s(w: np.ndarray) -> np.ndarray:
    """Pack [M,K] or [E,M,K] into flat uint8, with K divisible by 128.

    Each slice contains M*K/4 interleaved bytes, M little-endian F32 absmean
    scales, then zero padding to 32 bytes. Rounding is ties-to-even; zero rows
    have zero scale and code 1. Codes 0/1/2 represent -1/0/+1.
    """
    w = np.asarray(w)
    if w.ndim not in (2, 3) or any(n <= 0 for n in w.shape) or w.shape[-1] % 128:
        raise ValueError(f"I2_S requires [M,K] or [E,M,K] with K divisible by 128, got {w.shape}")
    if not np.issubdtype(w.dtype, np.floating):
        raise ValueError(f"I2_S requires floating-point weights, got {w.dtype}")
    matrices = w[np.newaxis, ...] if w.ndim == 2 else w
    rows, cols = w.shape[-2:]
    packed_size = rows * cols // 4
    stride = (packed_size + 4 * rows + 31) // 32 * 32
    result = np.zeros((len(matrices), stride), dtype=np.uint8)
    # Bound temporary float/code arrays independently of the matrix size.
    rows_per_chunk = max(1, (1 << 20) // cols)
    for expert, matrix in enumerate(matrices):
        scales_out = result[expert, packed_size:packed_size + 4 * rows].view("<f4")
        for first in range(0, rows, rows_per_chunk):
            block = np.array(matrix[first:first + rows_per_chunk], dtype=np.float32, copy=True)
            if not np.isfinite(block).all():
                raise ValueError("I2_S cannot quantize non-finite weights")
            scales = np.abs(block).mean(axis=1, dtype=np.float64).astype(np.float32)
            if not np.isfinite(scales).all():
                raise ValueError("I2_S row absmean overflow")
            scales_out[first:first + len(block)] = scales
            np.divide(block, scales[:, None], out=block, where=scales[:, None] != 0)
            block[scales == 0] = 0
            np.rint(block, out=block)
            np.clip(block, -1, 1, out=block)
            codes = (block + 1).astype(np.uint8).reshape(-1, 4, 32)
            packed = ((codes[:, 0] << 6) | (codes[:, 1] << 4)
                      | (codes[:, 2] << 2) | codes[:, 3])
            start = first * cols // 4
            result[expert, start:start + packed.size] = packed.reshape(-1)
    return result.reshape(-1)


class Qwen36BitNetModel(Qwen3_5MoeTextModel):
    model_arch = gguf.MODEL_ARCH.QWEN35MOE
    no_mtp = True

    def __init__(self, dir_model: Path, fname_out: Path):
        hparams = self.load_hparams(dir_model, is_mistral_format=False)
        text_config = hparams.get("text_config", hparams)
        if text_config.get("model_type") not in ("qwen3_5_moe", "qwen3_5_moe_text"):
            raise ValueError("Expected a Qwen3.5/3.6 MoE HF checkpoint")
        if (hparams.get("quantization_config") or text_config.get("quantization_config")
                or (dir_model / "hf_quant_config.json").exists()):
            raise ValueError("Use the original floating-point HF checkpoint, not prequantized weights")
        self.inventory: Counter[tuple[str, str]] = Counter()
        self.inventory_bytes: Counter[tuple[str, str]] = Counter()
        super().__init__(dir_model, gguf.LlamaFileType.MOSTLY_I2_S, fname_out,
                         use_temp_file=True, hparams=hparams)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        for new_name, transformed in super().modify_tensors(data_torch, name, bid):
            role = self.tensor_map.get_type(new_name, try_suffixes=(".weight", ".bias"))
            if role in I2_S_ROLES and new_name.endswith(".weight"):
                expected_ndim = 3 if role in EXPERT_ROLES else 2
                if transformed.ndim != expected_ndim:
                    raise ValueError(f"Unsupported native I2_S shape for {new_name}: {tuple(transformed.shape)}")
                qtype = gguf.GGMLQuantizationType.I2_S
            elif role in F32_ROLES or (role in I2_S_ROLES and new_name.endswith(".bias")):
                qtype = gguf.GGMLQuantizationType.F32
            else:
                raise ValueError(f"Tensor role is not whitelisted: {name} -> {new_name} ({role})")
            data = LazyTorchTensor.to_eager(transformed).float().numpy()
            shape = tuple(data.shape)
            if qtype == gguf.GGMLQuantizationType.I2_S:
                data = quantize_to_i2_s(data)
            elif not np.isfinite(data).all():
                raise ValueError(f"Non-finite F32 tensor: {new_name}")
            self.gguf_writer.add_tensor(new_name, data, raw_shape=shape, raw_dtype=qtype)
            key = (role.name, qtype.name)
            self.inventory[key] += 1
            self.inventory_bytes[key] += data.nbytes
            logger.info("%s: %s, shape=%s, bytes=%d", new_name, qtype.name, shape, data.nbytes)
        # Already spooled with logical shapes; bypass the generic quantizer and its fallback.
        return ()

    def prepare_tensors(self):
        logger.info("Text only: no vision or MTP; routers, GDN alpha/beta, conv, norms, biases and A/dt stay F32")
        super().prepare_tensors()
        for key, count in sorted(self.inventory.items()):
            logger.info("Coverage %s/%s: %d tensors, %d bytes", *key, count, self.inventory_bytes[key])
        if not any(qtype == "I2_S" for _, qtype in self.inventory):
            raise ValueError("No I2_S tensors were emitted")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_tensor_data_layout(I2_S_LAYOUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Local Qwen3.6 MoE HF model directory")
    parser.add_argument("--outfile", type=Path, help="New GGUF path (default: MODEL/Qwen3.6-35B-A3B-1.58bit.gguf)")
    args = parser.parse_args()
    outfile = args.outfile if args.outfile is not None else args.model / "Qwen3.6-35B-A3B-1.58bit.gguf"
    if os.path.lexists(outfile):
        parser.error(f"Refusing existing output or symlink: {outfile}")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    # Same-filesystem staging and exclusive publication also protect against races.
    with tempfile.TemporaryDirectory(prefix=".qwen36-i2-s-", dir=outfile.parent) as staging:
        staged_file = Path(staging) / "model.gguf"
        model = Qwen36BitNetModel(args.model, staged_file)
        try:
            with torch.inference_mode():
                model.write()
            os.link(staged_file, outfile)
        finally:
            model.gguf_writer.close()
            if model.gguf_writer.temp_file is not None:
                model.gguf_writer.temp_file.close()
    logger.info("Wrote %s", outfile)


if __name__ == "__main__":
    main()
