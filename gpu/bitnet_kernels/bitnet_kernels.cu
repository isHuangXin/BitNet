#include "bitnet_kernels.h"

extern "C" void bitlinear_int8xint2(int8_t* input0, int8_t* input1, __nv_bfloat16* output0, __nv_bfloat16* s, __nv_bfloat16* ws, int M, int N, int K, cudaStream_t stream){
    if (M == 1 && N == 3840 && K == 2560){
        ladder_int8xint2_kernel<1, 3840, 2560, 3, 8, 16><<<dim3(240, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 2560){
        ladder_int8xint2_kernel<1, 2560, 2560, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 13824 && K == 2560){
        ladder_int8xint2_kernel<1, 13824, 2560, 2, 8, 16><<<dim3(864, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if (M == 1 && N == 2560 && K == 6912){
        ladder_int8xint2_kernel<1, 2560, 6912, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 4800 && K == 3200){
        ladder_int8xint2_kernel<1, 4800, 3200, 6, 8, 16><<<dim3(300, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 3200){
        ladder_int8xint2_kernel<1, 3200, 3200, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 20480 && K == 3200){
        ladder_int8xint2_kernel<1, 20480, 3200, 2, 8, 16><<<dim3(1280, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 3200 && K == 10240){
        ladder_int8xint2_kernel<1, 3200, 10240, 1, 8, 16><<<dim3(200, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }    
    else if(M == 1 && N == 5120 && K == 27648){
        ladder_int8xint2_kernel<1, 5120, 27648, 1, 8, 16><<<dim3(320, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else if(M == 1 && N == 55296 && K == 5120){
        ladder_int8xint2_kernel<1, 55296, 5120, 1, 8, 16><<<dim3(3456, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // ============================================================
    // YOCO-BITNET 1.5B model shapes (all ws_num=1, separate weights)
    // ============================================================
    // k_proj, v_proj, yoco_k_proj, yoco_v_proj: (512, 2560)
    else if (M == 1 && N == 512 && K == 2560){
        ladder_int8xint2_kernel<1, 512, 2560, 1, 8, 16><<<dim3(32, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // self q_proj, self gate_proj_attn: (3072, 2560)
    else if (M == 1 && N == 3072 && K == 2560){
        ladder_int8xint2_kernel<1, 3072, 2560, 1, 8, 16><<<dim3(192, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // self o_proj: (2560, 3072)
    else if (M == 1 && N == 2560 && K == 3072){
        ladder_int8xint2_kernel<1, 2560, 3072, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // up_proj, gate_proj_ffn: (7680, 2560)
    else if (M == 1 && N == 7680 && K == 2560){
        ladder_int8xint2_kernel<1, 7680, 2560, 1, 8, 16><<<dim3(480, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // down_proj: (2560, 7680)
    else if (M == 1 && N == 2560 && K == 7680){
        ladder_int8xint2_kernel<1, 2560, 7680, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // cross q_proj, cross gate_proj_attn: (6144, 2560)
    else if (M == 1 && N == 6144 && K == 2560){
        ladder_int8xint2_kernel<1, 6144, 2560, 1, 8, 16><<<dim3(384, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    // cross o_proj: (2560, 6144)
    else if (M == 1 && N == 2560 && K == 6144){
        ladder_int8xint2_kernel<1, 2560, 6144, 1, 8, 16><<<dim3(160, 1, 1), dim3(8, 16, 1), 0, stream>>>(input0, input1, output0, s, ws);
    }
    else{
        std::cout << "required ladder gemm kernel: M " << M << ", N " << N << ", K " << K << std::endl;
    }
}