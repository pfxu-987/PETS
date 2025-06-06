#include "turbo_transformers/layers/kernels/fused_lora_kernel.h"
#include "turbo_transformers/core/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace turbo_transformers {
namespace layers {
namespace kernels {

// GELU激活函数 (remains the same)
__device__ __forceinline__ float gelu(float x) {
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  float tanh_inner = tanhf(inner);
  return 0.5f * x * (1.0f + tanh_inner);
}

// 融合的LoRA内核 - 处理单个线程块的一部分数据
template <int BLOCK_SIZE, int WARP_SIZE>
__global__ void FusedLoraKernel(
    const float* input,
    const float* down_scale_w,
    const float* up_scale_w,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size) {
    
    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    const int base_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    const float* input_ptr = input + base_idx;
    float* output_ptr = output + base_idx;
    
    extern __shared__ float shared_mem[];
    float* intermediate = shared_mem; 
    
    // 步骤1: 下采样矩阵乘法 (input * down_scale_w -> intermediate)
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += input_ptr[j] * down_scale_w[j * intermediate_size + i];
        }
        intermediate[i] = sum;
    }
    
    __syncthreads();
    
    // 步骤2: GELU激活 (No bias for LoRA here)
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        intermediate[i] = gelu(intermediate[i]);
    }
    
    __syncthreads();
    
    // 步骤3: 上采样矩阵乘法 (intermediate * up_scale_w -> output)
    // 并添加残差连接 (output += sum_from_lora_branch)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; j++) {
            sum += intermediate[j] * up_scale_w[j * hidden_size + i];
        }
        output_ptr[i] += sum; // output_ptr was initialized with dense_output
    }
}

// 针对较大的隐藏层尺寸优化的LoRA版本
template <int BLOCK_SIZE, int WARP_SIZE>
__global__ void FusedLoraKernelLarge(
    const float* input,
    const float* down_scale_w,
    const float* up_scale_w,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    const int base_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    const float* input_ptr = input + base_idx;
    float* output_ptr = output + base_idx;
    
    extern __shared__ float shared_mem[];
    float* intermediate = shared_mem; 
    
    // 步骤1: 下采样矩阵乘法 (input * down_scale_w -> intermediate)
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int items_per_warp = (intermediate_size + warps_per_block - 1) / warps_per_block;
    const int warp_start = wid * items_per_warp;
    const int warp_end = min(warp_start + items_per_warp, intermediate_size);
    
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += input_ptr[j] * down_scale_w[j * intermediate_size + i];
        }
        intermediate[i] = sum;
    }
    
    __syncthreads();
    
    // 步骤2: GELU激活
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        intermediate[i] = gelu(intermediate[i]);
    }
    
    __syncthreads();
    
    // 步骤3: 上采样矩阵乘法 (intermediate * up_scale_w -> output)
    // 并添加残差连接 (output += sum_from_lora_branch)
    const int items_per_warp_out = (hidden_size + warps_per_block - 1) / warps_per_block;
    const int warp_start_out = wid * items_per_warp_out;
    const int warp_end_out = min(warp_start_out + items_per_warp_out, hidden_size);
    
    for (int i = warp_start_out + lane; i < warp_end_out; i += WARP_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; j++) {
            sum += intermediate[j] * up_scale_w[j * hidden_size + i];
        }
        output_ptr[i] += sum; // output_ptr was initialized with dense_output
    }
}

// 启动融合的LoRA内核的包装函数
void LaunchFusedLoraKernel(
    const float* input,
    const float* down_scale_w,
    const float* up_scale_w,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream) {
    
    const int block_size = 256;
    const int warp_size = 32;
    
    dim3 grid(seq_len, batch_size);
    dim3 block(block_size);
    
    size_t shared_mem_size = intermediate_size * sizeof(float);
    
    if (hidden_size <= 1024 && intermediate_size <= 1024) {
        FusedLoraKernel<block_size, warp_size><<<grid, block, shared_mem_size, stream>>>(
            input, down_scale_w, up_scale_w, output,
            batch_size, seq_len, hidden_size, intermediate_size);
    } else {
        FusedLoraKernelLarge<block_size, warp_size><<<grid, block, shared_mem_size, stream>>>(
            input, down_scale_w, up_scale_w, output,
            batch_size, seq_len, hidden_size, intermediate_size);
    }
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers 