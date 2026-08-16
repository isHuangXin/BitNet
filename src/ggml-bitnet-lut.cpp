#include <vector>
#include <type_traits>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
#include "bitnet-lut-kernels.h"
#endif

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        if (src1->ne[1] <= 1) {
            return true;
        }
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_bitnet_get_type_bits(src0->type);

    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const size_t ne00 = src0->ne[0];
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];

    const int ith = params->ith;
    const int nth = params->nth;
    const int bits = ggml_bitnet_get_type_bits(src0->type);

    struct bitnet_tensor_extra * extra = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(extra != nullptr);

    char * wdata = (char *)params->wdata;
    const size_t wsize_per_thread = ggml_bitnet_mul_mat_get_wsize(src0, src1, dst);

    int8_t  * qlut  = (int8_t  *)(wdata);
    bitnet_float_type * lut_scales = (bitnet_float_type *)(qlut + ne10 * ne11 * 15);
    bitnet_float_type * lut_biases = (bitnet_float_type *)(lut_scales + ne11);

    if (ith == 0) {
        ggml_bitnet_mul_mat_task_init(
            (void *)((char *)src1->data),
            (void *)qlut,
            (void *)lut_scales,
            (void *)lut_biases,
            ne10, ne00, ne11, bits);
    }

    // barrier
    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    ggml_bitnet_mul_mat_task_compute(
        (void *)extra->qweights,
        (void *)extra->scales,
        (void *)qlut,
        (void *)lut_scales,
        (void *)lut_biases,
        (void *)((char *)dst->data),
        ne10, ne00, ne11, bits);
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];

    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int ne00 = src0->ne[0]; // K (weight cols)
    const int ne01 = src0->ne[1]; // M (weight rows)
    const int ne10 = src1->ne[0]; // K (activation cols)
    const int ne11 = src1->ne[1]; // batch size
    const int ne0  = dst->ne[0];

    const int ith = params->ith;
    const int nth = params->nth;

    struct bitnet_tensor_extra * wt = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(wt != nullptr);

    char * wdata = (char *)params->wdata;
    bitnet_float_type * bitnet_f_ptr = (bitnet_float_type *)wdata;
    char * cur_wdata = wdata;
    if (sizeof(bitnet_float_type) == 2) {
        cur_wdata = wdata + std::max((size_t)ne10, (size_t)ne01) * ne11 * sizeof(bitnet_float_type);
    }

    int8_t * three_qlut = (int8_t *)cur_wdata;
    const int total_k = ne10;
    const int three_k = (int)(total_k / wt->BK) * wt->BK;
    const int two_k = total_k - three_k;

    bitnet_float_type * lut_scales = (bitnet_float_type *)(three_qlut + three_k / 3 * 16 * 2 * ne11);
    int8_t * two_qlut = (int8_t *)(lut_scales + ne11);

    if (ith == 0) {
        ggml_bitnet_transform_tensor((struct ggml_tensor *)src0);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        bitnet_float_type * act_input;
        if (sizeof(bitnet_float_type) == 2) {
            ggml_fp32_to_fp16_row((const float *)src1->data, (ggml_fp16_t *)bitnet_f_ptr, ne10 * ne11);
            act_input = bitnet_f_ptr;
        } else {
            act_input = (bitnet_float_type *)src1->data;
        }
        ggml_preprocessor(ne11, ne01, three_k, two_k, act_input, lut_scales, three_qlut, two_qlut);
    }

    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    bitnet_float_type * act_output;
    if (sizeof(bitnet_float_type) == 2) {
        act_output = bitnet_f_ptr;
    } else {
        act_output = (bitnet_float_type *)dst->data;
    }

    const int n_tile_num = wt->n_tile_num;
    GGML_ASSERT(ne0 % n_tile_num == 0);
    const int w_size         = three_k * ne01 / (2 * 3);
    const int w_tile_size    = w_size / n_tile_num;
    const int c_size         = ne01;
    const int c_tile_size    = c_size / n_tile_num;
    const int sign_size      = three_k * ne01 / 24;
    const int sign_tile_size = sign_size / n_tile_num;

    const int th_tile_num = (n_tile_num + nth - 1) / nth;
    const int th_tile_beg = ith * th_tile_num;
    const int th_tile_end = std::min((ith + 1) * th_tile_num, n_tile_num);

    uint8_t * sign = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2;

    const int two_w_size      = ne01 * two_k / (2 * 2);
    const int two_w_tile_size = two_w_size / n_tile_num;
    uint8_t * two_A = ((uint8_t *)(wt->qweights)) + three_k * ne01 / 3 / 2 + three_k * ne01 / 3 / 8;

    // Decompose ne11 into batch sizes: 512, 256, 128, 32, 8, 1
    int iter = ne11;
    int bs512_num = iter / 512; iter -= 512 * bs512_num;
    int bs256_num = iter / 256; iter -= 256 * bs256_num;
    int bs128_num = iter / 128; iter -= 128 * bs128_num;
    int bs32_num  = iter / 32;  iter -= 32 * bs32_num;
    int bs8_num   = iter / 8;   iter -= 8 * bs8_num;
    int bs1_num   = iter;

    int three_qlut_offset = 0;
    int two_qlut_offset = 0;
    int lut_scales_offset = 0;
    int output_offset = 0;

    // Helper macro for batch processing
    #define PROCESS_BATCH(BS, COUNT) \
    for (int i = 0; i < (COUNT); i++) { \
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) { \
            ggml_qgemm_lut(BS, ne01, ne00, three_k, \
                           ((uint8_t *)(wt->qweights) + i_tile * w_tile_size), \
                           sign + i_tile * sign_tile_size, \
                           three_qlut + three_qlut_offset + i * (BS) * three_k / 3 * 32, \
                           wt->scales, \
                           lut_scales + lut_scales_offset + i * (BS), \
                           act_output + i_tile * c_tile_size + output_offset + i * (BS) * ne01); \
        } \
        if (two_k > 0) { \
            for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) { \
                ggml_qgemm_lut(BS, ne01, ne00, two_k, \
                               two_A + i_tile * two_w_tile_size, \
                               NULL, \
                               two_qlut + two_qlut_offset + i * (BS) * two_k / 2 * 32, \
                               wt->scales, \
                               lut_scales + lut_scales_offset + i * (BS), \
                               act_output + i_tile * c_tile_size + output_offset + i * (BS) * ne01); \
            } \
        } \
    }

    // bs 512
    PROCESS_BATCH(512, bs512_num)
    three_qlut_offset += bs512_num * 512 * three_k / 3 * 32;
    two_qlut_offset   += bs512_num * 512 * two_k / 2 * 32;
    lut_scales_offset += bs512_num * 512;
    output_offset     += bs512_num * 512 * ne01;

    // bs 256
    PROCESS_BATCH(256, bs256_num)
    three_qlut_offset += bs256_num * 256 * three_k / 3 * 32;
    two_qlut_offset   += bs256_num * 256 * two_k / 2 * 32;
    lut_scales_offset += bs256_num * 256;
    output_offset     += bs256_num * 256 * ne01;

    // bs 128
    PROCESS_BATCH(128, bs128_num)
    three_qlut_offset += bs128_num * 128 * three_k / 3 * 32;
    two_qlut_offset   += bs128_num * 128 * two_k / 2 * 32;
    lut_scales_offset += bs128_num * 128;
    output_offset     += bs128_num * 128 * ne01;

    // bs 32
    PROCESS_BATCH(32, bs32_num)
    three_qlut_offset += bs32_num * 32 * three_k / 3 * 32;
    two_qlut_offset   += bs32_num * 32 * two_k / 2 * 32;
    lut_scales_offset += bs32_num * 32;
    output_offset     += bs32_num * 32 * ne01;

    // bs 8
    PROCESS_BATCH(8, bs8_num)
    three_qlut_offset += bs8_num * 8 * three_k / 3 * 32;
    two_qlut_offset   += bs8_num * 8 * two_k / 2 * 32;
    lut_scales_offset += bs8_num * 8;
    output_offset     += bs8_num * 8 * ne01;

    // bs 1
    PROCESS_BATCH(1, bs1_num)

    #undef PROCESS_BATCH

    if (sizeof(bitnet_float_type) == 2) {
        // Convert fp16 output to fp32
        for (int i_tile = th_tile_beg; i_tile < th_tile_end; i_tile++) {
            const int dst_offset = i_tile * c_tile_size;
            ggml_fp16_to_fp32_row((const ggml_fp16_t *)(act_output + dst_offset),
                                  (float *)dst->data + dst_offset,
                                  ne01 / n_tile_num * ne11);
        }
    }
}

#endif
