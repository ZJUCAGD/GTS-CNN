/*
Create by Yijie Zhu
@2019.10
*/
#ifndef _UNPOOL3D_GPU_H
#define _UNPOOL3D_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

torch::Tensor MeanInterpolate_wrapper_fast(torch::Tensor input_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);
torch::Tensor MeanInterpolateGrad_wrapper_fast(torch::Tensor input_tensor, torch::Tensor grad_output_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);

void MeanInterpolate_kernel_launcher_fast(int B, int N, int M, int C, int K, 
                            const int* nnIndex, const int* nnCount, const float* input, float* output, 
                            cudaStream_t stream);
void MeanInterpolateGrad_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, float* gradInput, cudaStream_t stream);


torch::Tensor WeightedInterpolate_wrapper_fast(torch::Tensor input_tensor, torch::Tensor weight_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);
torch::Tensor WeightedInterpolateGrad_wrapper_fast(torch::Tensor input_tensor, torch::Tensor grad_output_tensor, torch::Tensor weight_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);

void WeightedInterpolate_kernel_launcher_fast(int B, int N, int M, int C, int K, 
                            const int* nnIndex, const int* nnCount, const float* input, const float* weight,
                            float* output, cudaStream_t stream);

void WeightedInterpolateGrad_kernel_launcher_fast(int B, int N, int M, int C, int K, 
							const int* nnIndex, const int* nnCount, const float* gradOutput, const float* weight,
                           	float* gradInput, cudaStream_t stream);

#endif
