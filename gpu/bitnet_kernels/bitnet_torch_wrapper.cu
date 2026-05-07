// PyTorch C++ extension wrapper for bitnet INT8×INT2 kernel
// Eliminates ctypes overhead (~140µs → ~5µs per call)

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Declaration of the kernel entry point (defined in bitnet_kernels.cu)
extern "C" void bitlinear_int8xint2(
    int8_t* input0, int8_t* input1, __nv_bfloat16* output0,
    __nv_bfloat16* s, __nv_bfloat16* ws,
    int M, int N, int K, cudaStream_t stream);

torch::Tensor bitlinear_int8xint2_forward(
    torch::Tensor input,      // INT8, shape (M, K)
    torch::Tensor weight,     // INT2-packed INT8, shape (N, K//4)
    torch::Tensor act_scale,  // BF16, shape (M, 1)
    torch::Tensor w_scale     // BF16, shape (4,)
) {
    TORCH_CHECK(input.is_cuda(), "input must be CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kInt8, "input must be int8");
    TORCH_CHECK(weight.dtype() == torch::kInt8, "weight must be int8");

    int M = input.size(0);
    int K = weight.size(1) * 4;
    int N = weight.size(0);

    auto output = torch::zeros({M, N}, torch::TensorOptions()
        .dtype(torch::kBFloat16).device(input.device()));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    bitlinear_int8xint2(
        input.data_ptr<int8_t>(),
        weight.data_ptr<int8_t>(),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(act_scale.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(w_scale.data_ptr<at::BFloat16>()),
        M, N, K, stream
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &bitlinear_int8xint2_forward,
          "BitLinear INT8xINT2 forward (CUDA)");
}
