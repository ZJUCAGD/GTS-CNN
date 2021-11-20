#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../cuda_utils.h"
#include "nnquery_gpu.h"

extern THCState *state;

void BuildSphereNeighbor_wrapper_fast(torch::Tensor database_tensor, //float32, database points: batch * npoint * 3
                                    torch::Tensor query_tensor, //float32, query points: batch * mpoint * 3
                                    float radius, // range search radius
                                    int nn_sample, // max number of neighbors sampled in the range
                                    torch::Tensor nn_index_tensor,//int32 neighbor indices: batch * mpoint * nn_sample
                                    torch::Tensor nn_count_tensor, //int32, number of neighbors: batch * mpoint
                                    torch::Tensor nn_dist_tensor)// float32, distance to the neighbors: batch * mpoint * nn_sample
{                        
    CHECK_INPUT(database_tensor);
    CHECK_INPUT_TYPE(database_tensor, torch::ScalarType::Float);
    CHECK_INPUT(query_tensor);
    CHECK_INPUT_TYPE(query_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_dist_tensor);
    CHECK_INPUT_TYPE(nn_dist_tensor, torch::ScalarType::Float);

    // get the dims required by computations
    int B = database_tensor.size(0);    // batch size
    int N = database_tensor.size(1);    // number of input points
    int M = query_tensor.size(1); // number of output points

    const float *database = database_tensor.data<float>();
    const float *query = query_tensor.data<float>();
    int *nn_index = nn_index_tensor.data<int>();
    int *nn_count = nn_count_tensor.data<int>();
    float *nn_dist = nn_dist_tensor.data<float>();


    cudaStream_t stream = THCState_getCurrentStream(state);
    BuildSphereNeighbor_kernel_launcher_fast(B, N, M, nn_sample, radius,
                           database, query, nn_index, nn_count, nn_dist, stream);
}

void BuildCubeNeighbor_wrapper_fast(torch::Tensor database_tensor, //float32, database points: batch * npoint * 3
                                    torch::Tensor query_tensor, //float32, query points: batch * mpoint * 3
                                    float length, // cube size: length * length * length
                                    int nn_sample, // max number of neighbors sampled in the range
                                    int grid_size, // division along azimuth direction
                                    torch::Tensor nn_index_tensor,//int32 neighbor indices: batch * mpoint * nn_sample
                                    torch::Tensor nn_count_tensor) //int32, number of neighbors: batch * mpoint
{                        
    CHECK_INPUT(database_tensor);
    CHECK_INPUT_TYPE(database_tensor, torch::ScalarType::Float);
    CHECK_INPUT(query_tensor);
    CHECK_INPUT_TYPE(query_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);

    // get the dims required by computations
    int B = database_tensor.size(0);    // batch size
    int N = database_tensor.size(1);    // number of input points
    int M = query_tensor.size(1); // number of output points

    const float *database = database_tensor.data<float>();
    const float *query = query_tensor.data<float>();
    int *nn_index = nn_index_tensor.data<int>();
    int *nn_count = nn_count_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    BuildCubeNeighbor_kernel_launcher_fast(B, N, M, grid_size, nn_sample, length, database, query, nn_index, nn_count, stream);
    // return output_tensor
}

