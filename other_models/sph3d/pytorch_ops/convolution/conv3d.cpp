#include <torch/serialize/tensor.h>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../cuda_utils.h"
#include "conv3d_gpu.h"

extern THCState *state;
    
torch::Tensor DepthwiseConv3d_wrapper_fast(torch::Tensor input_tensor, // float32, batch * in_npoint * in_channels
                                        torch::Tensor filter_tensor, // float32, convolution: filter_size * in_channels * channel_multiplier
                                        torch::Tensor nn_index_tensor, // int32, neighbor and kernel bin indices: int, batch * mpoint * nn_sample
                                        torch::Tensor nn_count_tensor,  // int32, number of neighbors: int, batch * mpoint
                                        torch::Tensor bin_index_tensor) // int32, kernel bin indices: batch * out_mpoint * nn_sample
{                        
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(filter_tensor);
    CHECK_INPUT_TYPE(filter_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);
    CHECK_INPUT(bin_index_tensor);
    CHECK_INPUT_TYPE(bin_index_tensor, torch::ScalarType::Int);

    int B = input_tensor.size(0);    // batch size
    int N = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int r = filter_tensor.size(2);   // depthwise channel multiplier
    int M = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    const float *input = input_tensor.data<float>();
    const float *filter = filter_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    const int *bin_index = bin_index_tensor.data<int>();
    torch::Tensor output_tensor = torch::zeros({B, M, C*r}, torch::CUDA(torch::kFloat));
    float *output = output_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthwiseConv3d_kernel_launcher_fast(B, N, M, C, r, K, nn_index, nn_count, bin_index, input, filter, output, stream);

    return  output_tensor;   //batch * out_mpoint * out_channels  (out_channels = in_channels * channel_multiplier)
}


void DepthwiseConv3dGrad_wrapper_fast(torch::Tensor input_tensor, // float32, batch * in_npoint * in_channels
                                torch::Tensor filter_tensor, // float32, convolution: filter_size * in_channels * channel_multiplier
                                torch::Tensor grad_output_tensor, //float32, batch * out_mpoint * out_channels
                                torch::Tensor nn_index_tensor, // int32, neighbor and kernel bin indices: int, batch * mpoint * nn_sample
                                torch::Tensor nn_count_tensor,  // int32, number of neighbors: int, batch * mpoint
                                torch::Tensor bin_index_tensor, // int32, kernel bin indices: batch * out_mpoint * nn_sample
                                torch::Tensor grad_input_tensor, //float32, batch * in_npoint * in_channels
                                torch::Tensor grad_filter_tensor)//float32, filter_size * in_channels * channel_multiplier
{                        
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(filter_tensor);
    CHECK_INPUT_TYPE(filter_tensor, torch::ScalarType::Float);
    CHECK_INPUT(grad_output_tensor);
    CHECK_INPUT_TYPE(grad_output_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);
    CHECK_INPUT(bin_index_tensor);
    CHECK_INPUT_TYPE(bin_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(grad_input_tensor);
    CHECK_INPUT_TYPE(grad_input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(grad_filter_tensor);
    CHECK_INPUT_TYPE(grad_filter_tensor, torch::ScalarType::Float);

    // get the dims required by computations
    int B = input_tensor.size(0);    // batch size
    int N = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int F = filter_tensor.size(0);   // filter bin size
    int r = filter_tensor.size(2);   // depthwise channel multiplier
    int M = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    const float *input = input_tensor.data<float>();
    const float *filter = filter_tensor.data<float>();
    const float *grad_output = grad_output_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    const int *bin_index = bin_index_tensor.data<int>();
    // torch::Tensor output_tensor = torch::zeros({B, M, C*r}, torch::CUDA(torch::kFloat));
    float *grad_input = grad_input_tensor.data<float>();
    float *grad_filter = grad_filter_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    DepthwiseConv3dGrad_kernel_launcher_fast(B, N, M, F, C, r, K, nn_index, nn_count, bin_index,
                                        input, filter, grad_output, grad_input, grad_filter, stream);

    // return  grad_input_tensor, grad_filter_tensor
}