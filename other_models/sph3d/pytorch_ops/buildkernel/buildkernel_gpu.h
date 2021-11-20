/*
Create by Yijie Zhu
@2019.10
*/
#ifndef _BUILDKERNEL_GPU_H
#define _BUILDKERNEL_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

torch::Tensor SphericalKernel_wrapper_fast(torch::Tensor database_tensor, torch::Tensor query_tensor, 
									torch::Tensor nn_index_tensor,torch::Tensor nn_count_tensor, torch::Tensor nn_dist_tensor,
									float radius, int n_azim, int p_elev, int q_radi);

void SphericalKernel_kernel_launcher_fast(int B, int N, int M, int K, int n, int p, int q, float radius,
                                  const float* database, const float* query, const int* nnIndex,
                                  const int* nnCount, const float* nnDist, int* filtIndex, cudaStream_t stream);

#endif
