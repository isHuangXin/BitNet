import argparse
import os
from configparser import ConfigParser

def gen_ctor_code():
    kernel_code = "\n\
#include \"ggml-bitnet.h\"\n\
#include <cstring>\n\
#include <immintrin.h>\n\
#define GGML_BITNET_MAX_NODES 8192\n\
static bool initialized = false;\n\
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;\n\
static size_t bitnet_tensor_extras_index = 0;\n\
static void * aligned_malloc(size_t size) {\n\
#if defined(_WIN32)\n\
    return _aligned_malloc(size, 64);\n\
#else\n\
    void * ptr = nullptr;\n\
    posix_memalign(&ptr, 64, size);\n\
    return ptr;\n\
#endif\n\
}\n\
\n\
static void aligned_free(void * ptr) {\n\
#if defined(_WIN32)\n\
    _aligned_free(ptr);\n\
#else\n\
    free(ptr);\n\
#endif\n\
}\n\
#define BK2 32\n\
#if defined __AVX2__\n\
inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi32(va, vb);\n\
    *vh = _mm256_unpackhi_epi32(va, vb);\n\
}\n\
inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi64(va, vb);\n\
    *vh = _mm256_unpackhi_epi64(va, vb);\n\
}\n\
inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));\n\
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));\n\
}\n\
inline void Transpose_8_8(\n\
    __m256i *v0,\n\
    __m256i *v1,\n\
    __m256i *v2,\n\
    __m256i *v3,\n\
    __m256i *v4,\n\
    __m256i *v5,\n\
    __m256i *v6,\n\
    __m256i *v7)\n\
{\n\
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;\n\
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;\n\
    _mm256_merge_epi32(*v0, *v1, &w0, &w1);\n\
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);\n\
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);\n\
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);\n\
    _mm256_merge_epi64(w0, w2, &x0, &x1);\n\
    _mm256_merge_epi64(w1, w3, &x2, &x3);\n\
    _mm256_merge_epi64(w4, w6, &x4, &x5);\n\
    _mm256_merge_epi64(w5, w7, &x6, &x7);\n\
    _mm256_merge_si128(x0, x4, v0, v1);\n\
    _mm256_merge_si128(x1, x5, v2, v3);\n\
    _mm256_merge_si128(x2, x6, v4, v5);\n\
    _mm256_merge_si128(x3, x7, v6, v7);\n\
}\n\
#endif\n\
inline int32_t per_tensor_quant(int k, void* lut_scales_, void* b_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    bitnet_float_type* b = (bitnet_float_type*)b_;\n\
#if defined __AVX2__\n\
    __m256 max_vec = _mm256_set1_ps(0.f);\n\
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);\n\
    for (int i = 0; i < k / 8; i++) {\n\
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);\n\
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);\n\
        max_vec = _mm256_max_ps(vec_babs, max_vec);\n\
    }\n\
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));\n\
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));\n\
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));\n\
    float scales = 127 / _mm_cvtss_f32(max1);\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    #pragma unroll\n\
    for (int i=0; i< bs; i++) {\n\
        lut_scales[i] = 0.0;\n\
    }\n\
    return 0;\n\
}\n\
template<int act_k>\n\
inline int32_t three_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256i vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);\n\
    float scales = *lut_scales;\n\
    __m256i shuffle_mask = _mm256_set_epi8(\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00\n\
                                            );\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 24; ++k) {\n\
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);\n\
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);\n\
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);\n\
\n\
        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
\n\
        vec_lut[15] = _mm256_setzero_si256();\n\
        vec_lut[14] = _mm256_setzero_si256();\n\
        vec_lut[13] = vec_b0i;\n\
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);\n\
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);\n\
        vec_lut[12] = vec_b0i;\n\
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);\n\
        vec_lut[11] = vec_b0i;\n\
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);\n\
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);\n\
        vec_lut[10] = vec_b0i;\n\
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);\n\
        vec_lut[9] = vec_b0i;\n\
        vec_lut[8] = vec_b0i;\n\
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);\n\
        vec_lut[7] = vec_b0i;\n\
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);\n\
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);\n\
        vec_lut[6] = vec_b0i;\n\
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);\n\
        vec_lut[5] = vec_b0i;\n\
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);\n\
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);\n\
        vec_lut[4] = vec_b1i;\n\
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);\n\
        vec_lut[3] = vec_b1i;\n\
        vec_lut[2] = vec_b1i;\n\
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);\n\
        vec_lut[1] = vec_b2i;\n\
        vec_lut[0] = _mm256_setzero_si256();\n\
        __m256i ix[16];\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            ix[g] = vec_lut[g];\n\
        }\n\
\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
        }\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);\n\
\n\
    }\n\
\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
\n\
template<int act_k>\n\
inline int32_t two_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256i vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);\n\
    float scales = *lut_scales;\n\
    __m256i shuffle_mask = _mm256_set_epi8(\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00\n\
                                            );\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 16; ++k) {\n\
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);\n\
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);\n\
\n\
        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        vec_lut[15] = _mm256_setzero_si256();\n\
        vec_lut[14] = _mm256_setzero_si256();\n\
        vec_lut[13] = _mm256_setzero_si256();\n\
        vec_lut[12] = _mm256_setzero_si256();\n\
        vec_lut[11] = _mm256_setzero_si256();\n\
        vec_lut[10] = _mm256_setzero_si256();\n\
        vec_lut[9] = _mm256_setzero_si256();\n\
        vec_lut[8] = vec_b0;\n\
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);\n\
        vec_lut[7] = vec_b0;\n\
        vec_lut[6] = vec_b0;\n\
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);\n\
        vec_lut[5] = vec_b1;\n\
        vec_lut[4] = _mm256_setzero_si256();\n\
        vec_lut[3] = _mm256_setzero_si256();\n\
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);\n\
        vec_lut[2] = _mm256_setzero_si256();\n\
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);\n\
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);\n\
        vec_lut[1] = _mm256_setzero_si256();\n\
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);\n\
        vec_lut[0] = _mm256_setzero_si256();\n\
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);\n\
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);\n\
\n\
        __m256i ix[16];\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            ix[g] = vec_lut[g];\n\
        }\n\
\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
        }\n\
\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);\n\
\n\
    }\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
static bool is_type_supported(enum ggml_type type) {\n\
    if (type == GGML_TYPE_Q4_0 ||\n\
        type == GGML_TYPE_TL2) {\n\
        return true;\n\
    } else {\n\
        return false;\n\
    }\n\
}\n\
"
    return kernel_code


def _gen_avx512_three_inner_block(j):
    """Generate one unrolled inner block for three_tbl_impl AVX512 version."""
    return """\
                __m512i vec_a_{j} = vec_as[k * 4 + {j}];
                __m128i vec_k1_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 48 + K3 / 3 * 32 * bs));
                __m512i vec_sign_left_hi_{j} = _mm512_srai_epi16(_mm512_slli_epi16(vec_sign, (4 * {j})), 15);
                __m512i vec_sign_left_lo_{j} = _mm512_srai_epi16(_mm512_slli_epi16(vec_sign, (4 * {j} + 1)), 15);
                __m512i vec_v_top_{j} = _mm512_and_si512(_mm512_srli_epi16(vec_a_{j}, 4), vec_mask);
                __m512i vec_v_top_fir_{j} = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(vec_k1_{j}), vec_v_top_{j});
                __m512i vec_v_top_sec_{j} = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(vec_k2_{j}), vec_v_top_{j});
                __m512i vec_sign_right_hi_{j} = _mm512_srai_epi16(_mm512_slli_epi16(vec_sign, (4 * {j} + 2)), 15);
                __m512i vec_sign_right_lo_{j} = _mm512_srai_epi16(_mm512_slli_epi16(vec_sign, (4 * {j} + 3)), 15);
                __m512i vec_v_bot_{j} = _mm512_and_si512(vec_a_{j}, vec_mask);
                __m512i vec_v_bot_fir_{j} = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(vec_k3_{j}), vec_v_bot_{j});
                __m512i vec_v_bot_sec_{j} = _mm512_shuffle_epi8(_mm512_broadcast_i32x4(vec_k4_{j}), vec_v_bot_{j});
                __m512i vec_v_top_lo_{j} = _mm512_xor_si512(_mm512_add_epi16(_mm512_unpackhi_epi8(vec_v_top_fir_{j}, vec_v_top_sec_{j}), vec_sign_left_lo_{j}), vec_sign_left_lo_{j});
                __m512i vec_v_top_hi_{j} = _mm512_xor_si512(_mm512_add_epi16(_mm512_unpacklo_epi8(vec_v_top_fir_{j}, vec_v_top_sec_{j}), vec_sign_left_hi_{j}), vec_sign_left_hi_{j});
                __m512i vec_v_bot_lo_{j} = _mm512_xor_si512(_mm512_add_epi16(_mm512_unpackhi_epi8(vec_v_bot_fir_{j}, vec_v_bot_sec_{j}), vec_sign_right_lo_{j}), vec_sign_right_lo_{j});
                __m512i vec_v_bot_hi_{j} = _mm512_xor_si512(_mm512_add_epi16(_mm512_unpacklo_epi8(vec_v_bot_fir_{j}, vec_v_bot_sec_{j}), vec_sign_right_hi_{j}), vec_sign_right_hi_{j});
                vec_c0 = _mm512_add_epi16(vec_c0, vec_v_top_hi_{j});
                vec_c0 = _mm512_add_epi16(vec_c0, vec_v_bot_hi_{j});
                vec_c1 = _mm512_add_epi16(vec_c1, vec_v_top_lo_{j});
                vec_c1 = _mm512_add_epi16(vec_c1, vec_v_bot_lo_{j});
""".format(j=j)


def gen_tbl_impl(pre, BM, BK, bm, k_list):
    """Generate three_tbl_impl and two_tbl_impl with AVX512BW main loop + AVX2 tail."""

    # --- three_tbl_impl ---
    code = '#include <immintrin.h>\n\n'
    code += '#define BM{} {}\n'.format(pre, BM)
    code += '#define BBK{} {}\n'.format(pre, BK)
    code += 'template<int batch_size, int K3>\n'
    code += 'inline void three_tbl_impl_{}(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {{\n'.format(pre)

    # AVX512BW main loop (64 elements)
    code += '#ifdef __AVX512BW__\n'
    code += '    const __m512i vec_mask = _mm512_set1_epi8(0x0f);\n'
    code += '    const int KK = BBK{} / 3;\n'.format(pre)
    code += '    int i = 0;\n'
    code += '    for (; i + 64 <= BM{}; i += 64) {{\n'.format(pre)
    code += '        __m512i vec_as[KK / 2];\n'
    code += '        __m512i vec_signs[KK / 8];\n'
    code += '        #pragma unroll\n'
    code += '        for (int ai = 0; ai < KK / 2; ai++) {\n'
    code += '            vec_as[ai] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(a + i * KK / 2 + ai * 64));\n'
    code += '        }\n'
    code += '        #pragma unroll\n'
    code += '        for (int as = 0; as < KK / 8; as++) {\n'
    code += '            vec_signs[as] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(sign + i * KK / 8 + as * 64));\n'
    code += '        }\n'
    code += '#pragma unroll\n'
    code += '    for (int bs = 0; bs < batch_size; bs++) {\n'
    code += '        __m512i vec_c0 = _mm512_setzero_si512();\n'
    code += '        __m512i vec_c1 = _mm512_setzero_si512();\n'
    code += '#pragma unroll\n'
    code += '        for (int k = 0; k < KK / 8; k++) {\n'
    code += '            __m512i vec_sign = vec_signs[k];\n'
    for j in range(4):
        code += _gen_avx512_three_inner_block(j)
    code += '        }\n'
    # Store: 512-bit accumulators hold 32x16-bit values each. Widen to 32-bit.
    # vec_c0 has 32 int16 values for positions 0..31 (low halves of pairs)
    # vec_c1 has 32 int16 values for positions 32..63 (high halves of pairs)
    # We need to widen each 256-bit half to 8x int32
    code += '        // Widen 16-bit accumulators to 32-bit and store\n'
    code += '        __m256i c0_lo = _mm512_castsi512_si256(vec_c0);\n'
    code += '        __m256i c0_hi = _mm512_extracti64x4_epi64(vec_c0, 1);\n'
    code += '        __m256i c1_lo = _mm512_castsi512_si256(vec_c1);\n'
    code += '        __m256i c1_hi = _mm512_extracti64x4_epi64(vec_c1, 1);\n'
    code += '        // First 32 elements (i+0..i+31)\n'
    code += '        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs));\n'.format(pre)
    code += '        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c0_lo)));\n'
    code += '        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c0_lo, 1)));\n'
    code += '        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c1_lo)));\n'
    code += '        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c1_lo, 1)));\n'
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs), vec_gc0);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs), vec_gc1);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs), vec_gc2);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs), vec_gc3);\n'.format(pre)
    code += '        // Second 32 elements (i+32..i+63)\n'
    code += '        __m256i vec_gc4 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 32 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc5 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 40 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc6 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 48 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc7 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 56 + BM{} * bs));\n'.format(pre)
    code += '        vec_gc4 = _mm256_add_epi32(vec_gc4, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c0_hi)));\n'
    code += '        vec_gc5 = _mm256_add_epi32(vec_gc5, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c0_hi, 1)));\n'
    code += '        vec_gc6 = _mm256_add_epi32(vec_gc6, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(c1_hi)));\n'
    code += '        vec_gc7 = _mm256_add_epi32(vec_gc7, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(c1_hi, 1)));\n'
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 32 + BM{} * bs), vec_gc4);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 40 + BM{} * bs), vec_gc5);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 48 + BM{} * bs), vec_gc6);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 56 + BM{} * bs), vec_gc7);\n'.format(pre)
    code += '    }\n'
    code += '    }\n'  # end batch loop

    # AVX2 tail loop (32 elements)
    code += '    // AVX2 tail for remaining 32-element blocks\n'
    code += '    const __m256i vec_mask_256 = _mm256_set1_epi8(0x0f);\n'
    code += '    for (; i < BM{}; i += 32) {{\n'.format(pre)
    code += '        __m256i vec_as_t[KK / 2];\n'
    code += '        __m256i vec_signs_t[KK / 8];\n'
    code += '        #pragma unroll\n'
    code += '        for (int ai = 0; ai < KK / 2; ai++) {\n'
    code += '            vec_as_t[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));\n'
    code += '        }\n'
    code += '        #pragma unroll\n'
    code += '        for (int as = 0; as < KK / 8; as++) {\n'
    code += '            vec_signs_t[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));\n'
    code += '        }\n'
    code += '#pragma unroll\n'
    code += '    for (int bs = 0; bs < batch_size; bs++) {\n'
    code += '        __m256i vec_c0 = _mm256_setzero_si256();\n'
    code += '        __m256i vec_c1 = _mm256_setzero_si256();\n'
    code += '#pragma unroll\n'
    code += '        for (int k = 0; k < KK / 8; k++) {\n'
    code += '            __m256i vec_sign = vec_signs_t[k];\n'
    # Unrolled inner blocks for AVX2 tail
    for j in range(4):
        code += '                __m256i vec_a_{j} = vec_as_t[k * 4 + {j}];\n'.format(j=j)
        code += '                __m128i vec_k1_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 0  + K3 / 3 * 32 * bs));\n'.format(j=j)
        code += '                __m128i vec_k2_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 16 + K3 / 3 * 32 * bs));\n'.format(j=j)
        code += '                __m128i vec_k3_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 32 + K3 / 3 * 32 * bs));\n'.format(j=j)
        code += '                __m128i vec_k4_{j} = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + {j} * 64 + 48 + K3 / 3 * 32 * bs));\n'.format(j=j)
        code += '                __m256i vec_sign_left_hi_{j} = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * {j})), 15);\n'.format(j=j)
        code += '                __m256i vec_sign_left_lo_{j} = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * {j} + 1)), 15);\n'.format(j=j)
        code += '                __m256i vec_v_top_{j} = _mm256_and_si256(_mm256_srli_epi16(vec_a_{j}, 4), vec_mask_256);\n'.format(j=j)
        code += '                __m256i vec_v_top_fir_{j} = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_{j}, vec_k1_{j}), vec_v_top_{j});\n'.format(j=j)
        code += '                __m256i vec_v_top_sec_{j} = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_{j}, vec_k2_{j}), vec_v_top_{j});\n'.format(j=j)
        code += '                __m256i vec_sign_right_hi_{j} = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * {j} + 2)), 15);\n'.format(j=j)
        code += '                __m256i vec_sign_right_lo_{j} = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * {j} + 3)), 15);\n'.format(j=j)
        code += '                __m256i vec_v_bot_{j} = _mm256_and_si256(vec_a_{j}, vec_mask_256);\n'.format(j=j)
        code += '                __m256i vec_v_bot_fir_{j} = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_{j}, vec_k3_{j}), vec_v_bot_{j});\n'.format(j=j)
        code += '                __m256i vec_v_bot_sec_{j} = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_{j}, vec_k4_{j}), vec_v_bot_{j});\n'.format(j=j)
        code += '                __m256i vec_v_top_lo_{j} = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_{j}, vec_v_top_sec_{j}), vec_sign_left_lo_{j}), vec_sign_left_lo_{j});\n'.format(j=j)
        code += '                __m256i vec_v_top_hi_{j} = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_{j}, vec_v_top_sec_{j}), vec_sign_left_hi_{j}), vec_sign_left_hi_{j});\n'.format(j=j)
        code += '                __m256i vec_v_bot_lo_{j} = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_{j}, vec_v_bot_sec_{j}), vec_sign_right_lo_{j}), vec_sign_right_lo_{j});\n'.format(j=j)
        code += '                __m256i vec_v_bot_hi_{j} = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_{j}, vec_v_bot_sec_{j}), vec_sign_right_hi_{j}), vec_sign_right_hi_{j});\n'.format(j=j)
        code += '                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_{j});\n'.format(j=j)
        code += '                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_{j});\n'.format(j=j)
        code += '                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_{j});\n'.format(j=j)
        code += '                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_{j});\n'.format(j=j)
    code += '        }\n'
    code += '        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs));\n'.format(pre)
    code += '        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n'
    code += '        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n'
    code += '        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n'
    code += '        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n'
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs), vec_gc0);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs), vec_gc1);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs), vec_gc2);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs), vec_gc3);\n'.format(pre)
    code += '    }\n'
    code += '    }\n'  # end batch loop tail
    code += '#endif\n'
    code += '}\n\n'

    # --- two_tbl_impl (keep as AVX2 — it's much less hot, BK2=32 is tiny) ---
    code += 'template<int batch_size, int K2>\n'
    code += 'inline int32_t two_tbl_impl{}(int32_t* c, int8_t* lut, uint8_t* a) {{\n'.format(pre)
    code += '#ifdef __AVX2__\n'
    code += '    const __m256i vec_mask = _mm256_set1_epi8(0x0f);\n'
    code += '    const int KK = BK2 / 2;\n'
    code += '#pragma unroll\n'
    code += '    for (int i = 0; i < BM{}; i += 32) {{\n'.format(pre)
    code += '        __m256i vec_as[KK / 2];\n'
    code += '        #pragma unroll\n'
    code += '        for (int ai = 0; ai < KK / 2; ai++) {\n'
    code += '            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));\n'
    code += '        }\n'
    code += '#pragma unroll\n'
    code += '    for (int bs = 0; bs < batch_size; bs++) {\n'
    code += '        __m256i vec_c0 = _mm256_setzero_si256();\n'
    code += '        __m256i vec_c1 = _mm256_setzero_si256();\n'
    code += '#pragma unroll\n'
    code += '        for (int k = 0; k < KK / 8; k++) {\n'
    code += '            #pragma unroll\n'
    code += '            for (int j = 0; j < 4; j++) {\n'
    code += '                __m256i vec_a = vec_as[k * 4 + j];\n'
    code += '                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));\n'
    code += '                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));\n'
    code += '                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));\n'
    code += '                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));\n'
    code += '                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);\n'
    code += '                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);\n'
    code += '                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);\n'
    code += '                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);\n'
    code += '                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);\n'
    code += '                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);\n'
    code += '                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);\n'
    code += '                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);\n'
    code += '                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);\n'
    code += '                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);\n'
    code += '                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);\n'
    code += '                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);\n'
    code += '                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);\n'
    code += '                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo);\n'
    code += '            }\n'
    code += '        }\n'
    code += '        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs));\n'.format(pre)
    code += '        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs));\n'.format(pre)
    code += '        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n'
    code += '        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n'
    code += '        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n'
    code += '        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n'
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{} * bs), vec_gc0);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{} * bs), vec_gc1);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{} * bs), vec_gc2);\n'.format(pre)
    code += '        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{} * bs), vec_gc3);\n'.format(pre)
    code += '    }\n'
    code += '    }\n'
    code += '#endif\n'
    code += '    return 0;\n'
    code += '}\n\n'

    # --- three_qgemm_lut / two_qgemm_lut wrappers ---
    three_k = k_list[1]
    two_k = k_list[0]

    code += 'template<int BATCH_SIZE>\n'
    code += 'int32_t three_qgemm_lut_{}(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n'.format(pre)
    code += '    alignas(64) uint32_t CBits[BATCH_SIZE * BM{}];\n'.format(pre)
    code += '    memset(&(CBits[0]), 0, BATCH_SIZE * BM{} * sizeof(int32_t));\n'.format(pre)
    code += '#pragma unroll\n'
    code += '    for (int32_t k_outer = 0; k_outer < {} / BBK{}; ++k_outer) {{\n'.format(three_k, pre)
    code += '        three_tbl_impl_{}<BATCH_SIZE, {}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK{} / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK{} / 3 / 2 * BM{})])), (&(((uint8_t*)sign)[(k_outer * BBK{} / 3 / 8 * BM{})])));\n'.format(pre, three_k, pre, pre, pre, pre, pre)
    code += '    }\n'
    code += '#pragma unroll\n'
    code += '    for (int bs = 0; bs < BATCH_SIZE; bs++) {\n'
    code += '#pragma unroll\n'
    code += '        for (int i = 0; i < BM{}; i++) {{\n'.format(pre)
    code += '            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM{}]);\n'.format(pre)
    code += '        }\n'
    code += '  }\n'
    code += '  return 0;\n'
    code += '}\n\n'

    code += 'template<int BATCH_SIZE>\n'
    code += 'int32_t two_qgemm_lut_{}(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n'.format(pre)
    code += '    alignas(64) uint32_t CBits[BATCH_SIZE * BM{}];\n'.format(pre)
    code += '    memset(&(CBits[0]), 0, BATCH_SIZE * BM{} * sizeof(int32_t));\n'.format(pre)
    code += '#pragma unroll\n'
    code += '    for (int32_t k_outer = 0; k_outer < {} / 32; ++k_outer) {{\n'.format(two_k)
    code += '        two_tbl_impl{}<BATCH_SIZE, {}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM{})])));\n'.format(pre, two_k, pre)
    code += '    }\n'
    code += '#pragma unroll\n'
    code += '    for (int bs = 0; bs < BATCH_SIZE; bs++) {\n'
    code += '#pragma unroll\n'
    code += '        for (int i = 0; i < BM{}; i++) {{\n'.format(pre)
    code += '            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM{}]);\n'.format(pre)
    code += '            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[i];\n'
    code += '        }\n'
    code += '    }\n'
    code += '  return 0;\n'
    code += '}\n\n'

    return code


def gen_top_api(kernel_shapes, k_list):

    kernel_code = "void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {{\n\
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));\n\
    if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], k_list[0][0], k_list[0][1])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n".format(kernel_shapes[i][0], k_list[i][0], k_list[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])


    kernel_code = "".join([kernel_code, "void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1], k_list[0][0], k_list[0][1], "{}_{}".format(kernel_shapes[0][0], kernel_shapes[0][1]))])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[i][0], kernel_shapes[i][1], k_list[i][0], k_list[i][1], "{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]))])
    kernel_code = "".join([kernel_code, "}\n"])
    return kernel_code

def gen_transform_code(kernel_shapes):
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {\n\
    if (!(is_type_supported(tensor->type) && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    int bk = 0;\n\
    int bm = 0;\n"

    kernel_code = "".join([kernel_code, "\n\
    if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[0][0], kernel_shapes[0][1])])

    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "else if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])

    kernel_code = "".join([kernel_code, "\n\
    const int n_tile_num = m / bm;\n\
    const int BK = bk;\n\
    uint8_t * qweights;\n\
    bitnet_float_type * scales;\n\
\n\
    qweights = (uint8_t *) tensor->data;\n\
    int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;\n\
    if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;\n\
    float * i2_scales = (float * )(qweights + nbytes);\n\
    // Per-row scales: M floats stored after packed ternary bytes\n\
    scales = (bitnet_float_type *) aligned_malloc(m * sizeof(bitnet_float_type));\n\
    for (int r = 0; r < m; r++) {\n\
        scales[r] = (bitnet_float_type) i2_scales[r];\n\
    }\n\
\n\
    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;\n\
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {\n\
        /* .lut_scales_size = */ 1,\n\
        /* .BK              = */ BK,\n\
        /* .n_tile_num      = */ n_tile_num,\n\
        /* .qweights        = */ qweights,\n\
        /* .scales          = */ scales\n\
    };\n\
}\n"])

    return kernel_code

def get_three_k_two_k(K, bk):
    bk_num = K // bk
    three_k = bk_num * bk
    two_k = K - three_k
    return two_k, three_k

if __name__ == "__main__":
    ModelShapeDict = {
        "bitnet_b1_58-large"                : [[1536, 4096],
                                               [1536, 1536],
                                               [4096, 1536]],
        "bitnet_b1_58-3B"                   : [[3200, 8640],
                                               [3200, 3200],
                                               [8640, 3200]],
        "Llama3-8B-1.58-100B-tokens"        : [[14336, 4096],
                                               [4096, 14336],
                                               [1024, 4096],
                                               [4096, 4096]],
        "yoco-moe-30b-a3b-v3"              : [[8192, 3072],
                                               [3072, 4096],
                                               [1280, 3072],
                                               [3072, 1280],
                                               [1024, 3072],
                                               [3072, 3072],
                                               [128, 3072]]
    }

    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--model',default="input", type=str, dest="model",
                        help="choose from bitnet_b1_58-large/bitnet_b1_58-3B/Llama3-8B-1.58-100B-tokens.")
    parser.add_argument('--BM',default="input", type=str,
                        help="block length when cutting one weight (M, K) into M / BM weights (BM, K).")
    parser.add_argument('--BK',default="input", type=str,
                        help="block length when cutting one weight (M, K) into K / BK weights (M, BK).")
    parser.add_argument('--bm',default="input", type=str,
                        help="using simd instructions to compute (bm, 192 / bm) in one block")
    args = parser.parse_args()

    kernel_shapes = ModelShapeDict[args.model]

    BM_list = [int(item) for item in args.BM.split(',')]
    BK_list = [int(item) for item in args.BK.split(',')]
    bm_list = [int(item) for item in args.bm.split(',')]

    tbl_impl_code = []
    k_list = []

    for i in range(len(kernel_shapes)):
        k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

    for i in range(len(kernel_shapes)):
        tbl_impl_code.append(
            gen_tbl_impl("{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], k_list[i])
        )

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))

    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert (kernel_shapes[i][1] % BK_list[i]) % 32 == 0, "K %% BK %% 32 should be 0"
        assert bm_list[i] in [32], "choose bm from [32]"

    ctor_code = gen_ctor_code()
    api_code = gen_top_api(kernel_shapes, k_list)
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join("#if defined(GGML_BITNET_X86_TL2)"))
        f.write(''.join(ctor_code))
        for code in tbl_impl_code:
            f.write(''.join(code))
        f.write(''.join(api_code))
        f.write(''.join(trans_code))
        f.write(''.join("#endif"))

    config = ConfigParser()

    for i in range(len(kernel_shapes)):
        config.add_section('Kernels_{}'.format(i))
        config.set('Kernels_{}'.format(i), 'M'.format(i), str(kernel_shapes[i][0]))
        config.set('Kernels_{}'.format(i), 'K'.format(i), str(kernel_shapes[i][1]))
        config.set('Kernels_{}'.format(i), 'BM'.format(i), str(BM_list[i]))
        config.set('Kernels_{}'.format(i), 'BK'.format(i), str(BK_list[i]))
        config.set('Kernels_{}'.format(i), 'bmm'.format(i), str(bm_list[i]))

    with open(''.join([output_dir, "/kernel_config.ini"]), 'w') as configfile:
        config.write(configfile)
