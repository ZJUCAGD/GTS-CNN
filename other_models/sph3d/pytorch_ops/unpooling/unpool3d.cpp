#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../cuda_utils.h"
#include "unpool3d_gpu.h"

extern THCState *state;


// for the unpooling modules, we have in_mpoint<out_npoint
torch::Tensor MeanInterpolate_wrapper_fast(torch::Tensor input_tensor,  //input: float32, batch * in_mpoint * in_channels
                                torch::Tensor nn_index_tensor, //neighbor indices: int32, batch * out_npoint * nn_sample
                                torch::Tensor nn_count_tensor //number of neighbors: batch * out_npoint
                                // at::Tensor output_tensor  //output: float32, batch * out_npoint * in_channels
                                ) 
{                        
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);

    // get the dims required by computations
    int B = input_tensor.size(0);    // batch size
    int M = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int N = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    torch::Tensor output_tensor = torch::zeros({B, N, C}, torch::CUDA(torch::kFloat));
    const float *input = input_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    float *output = output_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    MeanInterpolate_kernel_launcher_fast(B, N, M, C, K, nn_index, nn_count, input, output, stream);
    return output_tensor;
}

torch::Tensor MeanInterpolateGrad_wrapper_fast(torch::Tensor input_tensor,
                                torch::Tensor grad_output_tensor,
                                torch::Tensor nn_index_tensor,
                                torch::Tensor nn_count_tensor) 
{
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(grad_output_tensor);
    CHECK_INPUT_TYPE(grad_output_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);

    // get the dims required by computations
    int B = input_tensor.size(0);    // batch size
    int M = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int N = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    const float *gradOutput = grad_output_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    torch::Tensor gradInput_tensor = torch::zeros({B, M, C}, torch::CUDA(torch::kFloat));
    float *gradInput = gradInput_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    MeanInterpolateGrad_kernel_launcher_fast(B, N, M, C, K, nn_index, nn_count, gradOutput, gradInput, stream);
    return gradInput_tensor;
}

torch::Tensor WeightedInterpolate_wrapper_fast(torch::Tensor input_tensor,  //input: float32, batch * in_mpoint * in_channels
                                torch::Tensor weight_tensor,
                                torch::Tensor nn_index_tensor, //neighbor indices: int32, batch * out_npoint * nn_sample
                                torch::Tensor nn_count_tensor //number of neighbors: batch * out_npoint
                                // at::Tensor output_tensor  //output: float32, batch * out_npoint * in_channels
                                ) 
{                        
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT_TYPE(weight_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);

    // get the dims required by computations
    int B = input_tensor.size(0);    // batch size
    int M = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int N = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    torch::Tensor output_tensor = torch::zeros({B, N, C}, torch::CUDA(torch::kFloat));
    const float *input = input_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    float *output = output_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    WeightedInterpolate_kernel_launcher_fast(B, N, M, C, K, nn_index, nn_count, input, weight, output, stream);
    return output_tensor;
}

torch::Tensor WeightedInterpolateGrad_wrapper_fast(torch::Tensor input_tensor,
                                torch::Tensor grad_output_tensor,
                                torch::Tensor weight_tensor,
                                torch::Tensor nn_index_tensor,
                                torch::Tensor nn_count_tensor) 
{
    CHECK_INPUT(input_tensor);
    CHECK_INPUT_TYPE(input_tensor, torch::ScalarType::Float);
    CHECK_INPUT(grad_output_tensor);
    CHECK_INPUT_TYPE(grad_output_tensor, torch::ScalarType::Float);
    CHECK_INPUT(weight_tensor);
    CHECK_INPUT_TYPE(weight_tensor, torch::ScalarType::Float);
    CHECK_INPUT(nn_index_tensor);
    CHECK_INPUT_TYPE(nn_index_tensor, torch::ScalarType::Int);
    CHECK_INPUT(nn_count_tensor);
    CHECK_INPUT_TYPE(nn_count_tensor, torch::ScalarType::Int);

    // get the dims required by computations
    int B = input_tensor.size(0);    // batch size
    int M = input_tensor.size(1);    // number of input points
    int C = input_tensor.size(2);    // number of input channels
    int N = nn_index_tensor.size(1); // number of output points
    int K = nn_index_tensor.size(2); // max number of neighbors sampled

    const float *gradOutput = grad_output_tensor.data<float>();
    const float *weight = weight_tensor.data<float>();
    const int *nn_index = nn_index_tensor.data<int>();
    const int *nn_count = nn_count_tensor.data<int>();
    torch::Tensor gradInput_tensor = torch::zeros({B, M, C}, torch::CUDA(torch::kFloat));
    float *gradInput = gradInput_tensor.data<float>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    WeightedInterpolateGrad_kernel_launcher_fast(B, N, M, C, K, nn_index, nn_count, gradOutput, weight, gradInput, stream);
    return gradInput_tensor;
}
