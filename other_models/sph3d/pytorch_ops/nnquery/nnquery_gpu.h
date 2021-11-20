/*
Create by Yijie Zhu
@2019.10
*/
#ifndef _NNQUERY_GPU_H
#define _NNQUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void BuildSphereNeighbor_wrapper_fast(torch::Tensor database_tensor, torch::Tensor query_tensor, float radius, int nn_sample,
					torch::Tensor nn_index_tensor,torch::Tensor nn_count_tensor, torch::Tensor nn_dist_tensor);

void BuildSphereNeighbor_kernel_launcher_fast(int B, int N, int M, int nnSample, float radius, const float* database,
                                 	const float* query, int* nnIndex, int* nnCount, float* nnDist, cudaStream_t stream);

  
void BuildCubeNeighbor_wrapper_fast(torch::Tensor database_tensor, torch::Tensor query_tensor, float length, int nn_sample, int grid_size,
					torch::Tensor nn_index_tensor,torch::Tensor nn_count_tensor);

void BuildCubeNeighbor_kernel_launcher_fast(int B, int N, int M, int gridSize, int nnSample, float length,
                               const float* database, const float* query, int* nnIndex, int* nnCount, cudaStream_t stream);

#endif
