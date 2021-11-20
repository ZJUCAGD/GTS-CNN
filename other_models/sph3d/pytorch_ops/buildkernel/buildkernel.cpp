#include <torch/serialize/tensor.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../cuda_utils.h"
#include "buildkernel_gpu.h"

extern THCState *state;

torch::Tensor SphericalKernel_wrapper_fast(torch::Tensor database_tensor, // database points: float32, batch * npoint * 3 (x,y,z)
                                        torch::Tensor query_tensor, // query points: float32, batch * mpoint * 3
                                        torch::Tensor nn_index_tensor, // neighbor and kernel bin indices: int, batch * mpoint * nn_sample
                                        torch::Tensor nn_count_tensor,  // number of neighbors: int, batch * mpoint
                                        torch::Tensor nn_dist_tensor, // distance to the neighbors: float32, batch * mpoint * nn_sample
                                        float radius, // range search radius
                                        int n_azim,  // division along azimuth direction
                                        int p_elev,  // division along elevation direction
                                        int q_radi)// division along radius direction
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
    int K = nn_index_tensor.size(2);

    const float *database = database_tensor.data<float>();
    const float *query = query_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    const float *nn_dist = nn_dist_tensor.data<float>();
    torch::Tensor filt_index_tensor = torch::zeros({B, M, K}, torch::CUDA(torch::kInt));// batch * in_npoint * in_channels
    int *filt_index = filt_index_tensor.data<int>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    SphericalKernel_kernel_launcher_fast(B, N, M, K, n_azim, p_elev, q_radi, radius,
                                        database, query, nn_index, nn_count, nn_dist,
                                        filt_index, stream);

    return  filt_index_tensor;   // kernel bin indices:int32, batch * mpoint * nn_sample
}



