/*
Create by Yijie Zhu
@2019.10
*/
#ifndef _POOL3D_GPU_H
#define _POOL3D_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void MaxPool3d_wrapper_fast(torch::Tensor input_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor,
 							torch::Tensor output_tensor,  //ouput[0]: batch * out_mpoint * in_channels
                            torch::Tensor max_index_tensor);   //ouput[1]: batch * out_mpoint * in_channels);
    
torch::Tensor MaxPool3dGrad_wrapper_fast(torch::Tensor input_tensor, torch::Tensor grad_output_tensor, torch::Tensor max_index_tensor);

void MaxPool3d_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       const float* input, float* output, int* maxIndex, cudaStream_t stream);
void MaxPool3dGrad_kernel_launcher_fast(int B, int N, int M, int C, const int* maxIndex,
                           const float* gradOutput, float* gradInput, cudaStream_t stream);

torch::Tensor AvgPool3d_wrapper_fast(torch::Tensor input_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);

torch::Tensor AvgPool3dGrad_wrapper_fast(torch::Tensor input_tensor, torch::Tensor grad_output_tensor, torch::Tensor nn_index_tensor, torch::Tensor nn_count_tensor);

void AvgPool3d_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                       				const float* input, float* output, cudaStream_t stream);

void AvgPool3dGrad_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           			const float* gradOutput, float* gradInput, cudaStream_t stream);
#endif
