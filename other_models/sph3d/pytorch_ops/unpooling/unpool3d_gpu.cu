// nnIndex: B*N*K;
// nnCount: B*N;
// input:   B*M*C;
// output:  B*N*C (N>M)
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.h"
#include "unpool3d_gpu.h"

__global__ void mean_interpolate_forward(int B, int N, int M, int C, int K, const int* nnIndex,
                                         const int* nnCount, const float* input, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                output[i*N*C+j] += input[i*M*C+m*C+c]/nnSize;
            }
        }
    }
}




__global__ void mean_interpolate_backward(int B, int N, int M, int C, int K, const int* nnIndex,
                                          const int* nnCount, const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                atomicAdd(&gradInput[i*M*C+m*C+c],gradOutput[i*N*C+j]/nnSize);
            }
        }
    }
}



__global__ void weighted_interpolate_forward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                             const float* input, const float* weight, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                float w = weight[i*N*K+n*K+k];
                output[i*N*C+j] += input[i*M*C+m*C+c]*w;
            }
        }
    }
}


__global__ void weighted_interpolate_backward(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                                              const float* gradOutput, const float* weight, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=threadIdx.x;j<N*C;j+=blockDim.x)
        {
            int n = j/C;
            int c = j%C;
            int nnSize = nnCount[i*N+n];
            for(int k=0;k<nnSize;k++)
            {
                int m = nnIndex[i*N*K+n*K+k];
                float w = weight[i*N*K+n*K+k];
                atomicAdd(&gradInput[i*M*C+m*C+c],gradOutput[i*N*C+j]*w);
            }
        }
    }
}



void MeanInterpolate_kernel_launcher_fast(int B, int N, int M, int C, int K, 
                            const int* nnIndex, const int* nnCount, const float* input, float* output, 
                            cudaStream_t stream)
{
      // cudaError_t err;
      // const int num_pair = b * n * n;
      // const int top_count = b * n;
      // cudaMemset(norm, 0, sizeof(float) * b * n);
      // cudaMemset(outputs, 0, sizeof(float) * b * n * c);
      // // aggregate_kernel<<<b, opt_n_threads(n)>>>(b, c, n, npoints, features, knn_graph, outputs, idxs);
      // aggregate_kernel<<<_GET_BLOCKS(num_pair), TOTAL_THREADS>>>(num_pair, b, n, c, features, xyz, outputs, norm, radius * radius, decay_radius * decay_radius, delta);
      // Normalization<<<_GET_BLOCKS(top_count), TOTAL_THREADS>>>(top_count, outputs, norm, b, n, c);
      
      // err = cudaGetLastError();
      // if (cudaSuccess != err) {
      //   fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      //   exit(-1);
      // }
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    mean_interpolate_forward<<<blocks, threads, 0, stream>>>(B, N, M, C, K, nnIndex, nnCount, input, output);
    // cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

void MeanInterpolateGrad_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, float* gradInput, cudaStream_t stream)
{
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    
    mean_interpolate_backward<<<blocks, threads, 0, stream>>>(B, N, M, C, K, nnIndex, nnCount, gradOutput, gradInput);
    // cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}


void WeightedInterpolate_kernel_launcher_fast(int B, int N, int M, int C, int K, 
                            const int* nnIndex, const int* nnCount, const float* input, const float* weight,
                            float* output, cudaStream_t stream)
{ 
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    weighted_interpolate_forward<<<blocks, threads, 0, stream>>>(B, N, M, C, K, nnIndex, nnCount, input, weight, output);
    // cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

void WeightedInterpolateGrad_kernel_launcher_fast(int B, int N, int M, int C, int K, const int* nnIndex, const int* nnCount,
                           const float* gradOutput, const float* weight, float* gradInput, cudaStream_t stream)
{
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    weighted_interpolate_backward<<<blocks, threads, 0, stream>>>(B, N, M, C, K, nnIndex, nnCount, gradOutput, weight, gradInput);
    // cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}