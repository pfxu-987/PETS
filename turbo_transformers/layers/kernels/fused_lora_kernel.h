#pragma once

#include <cuda_runtime.h>

namespace turbo_transformers {
namespace layers {
namespace kernels {

// 启动融合的LoRA内核的包装函数
void LaunchFusedLoraKernel(
    const float* input,
    const float* down_scale_w,
    // const float* down_scale_b, // Removed for LoRA
    const float* up_scale_w,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers 