/*
Create by Yijie Zhu
@2019.10
*/
#ifndef _CONV3D_GPU_H
#define _CONV3D_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

torch::Tensor DepthwiseConv3d_wrapper_fast(torch::Tensor input_tensor, // float32, batch * in_npoint * in_channels
                                        torch::Tensor filter_tensor, // float32, convolution: filter_size * in_channels * channel_multiplier
                                        torch::Tensor nn_index_tensor, // int32, neighbor and kernel bin indices: int, batch * mpoint * nn_sample
                                        torch::Tensor nn_count_tensor,  // int32, number of neighbors: int, batch * mpoint
                                        torch::Tensor bin_index_tensor); // int32, kernel bin indices: batch * out_mpoint * nn_sample

void DepthwiseConv3d_kernel_launcher_fast(int B, int N, int M, int C, int r, int K,
	                            const int* nnIndex, const int* nnCount, const int* binIndex,
	                            const float* input, const float* filter, float* output, cudaStream_t stream);

void DepthwiseConv3dGrad_wrapper_fast(torch::Tensor input_tensor, // float32, batch * in_npoint * in_channels
	                                torch::Tensor filter_tensor, // float32, convolution: filter_size * in_channels * channel_multiplier
	                                torch::Tensor grad_output_tensor, //float32, batch * out_mpoint * out_channels
	                                torch::Tensor nn_index_tensor, // int32, neighbor and kernel bin indices: int, batch * mpoint * nn_sample
	                                torch::Tensor nn_count_tensor,  // int32, number of neighbors: int, batch * mpoint
	                                torch::Tensor bin_index_tensor, // int32, kernel bin indices: batch * out_mpoint * nn_sample
	                                torch::Tensor grad_input_tensor, //float32, batch * in_npoint * in_channels
	                                torch::Tensor grad_filter_tensor);//float32, filter_size * in_channels * channel_multiplier

void DepthwiseConv3dGrad_kernel_launcher_fast(int B, int N, int M, int F, int C, int r, int K,
                                const int* nnIndex, const int* nnCount, const int* binIndex,
                                const float* input, const float* filter, const float* gradOutput,
                                float* gradInput, float* gradFilter, cudaStream_t stream);

#endif
