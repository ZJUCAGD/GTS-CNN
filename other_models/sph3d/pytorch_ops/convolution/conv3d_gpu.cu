#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda_utils.h"
#include "conv3d_gpu.h"


// nnIndex: B*M*K*2;
// nnCount: B*M;
// input:   B*N*C;
// filter:  filter_size*C*r;
// output:  B*M*(C*r)

__global__ void depthwise_conv3d_forward(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                                         const int* nnCount, const int* binIndex, const float* input,
                                         const float* filter, float* output)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                output[i*M*C*r+j] += input[i*N*C+n*C+cin]*filter[f*C*r+cout]/nnSize;
            }
        }
    }
}


__global__ void depthwise_input_backward(int B, int N, int M, int F, int C, int r, int K, const int* nnIndex,
                                         const int* nnCount, const int* binIndex, const float* input,
                                         const float* filter, const float* gradOutput, float* gradInput)
{
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                float derIn = gradOutput[i*M*C*r+j]*filter[f*C*r+cout]/nnSize;
                atomicAdd(&gradInput[i*N*C+n*C+cin],derIn);
            }
        }
    }
}


__global__ void depthwise_filter_backward(int B, int N, int M, int F, int C, int r, int K, const int* nnIndex,
                                          const int* nnCount, const int* binIndex, const float* input,
                                          const float* gradOutput, float* gradFilter, int sharedMemSize,
                                          int startIdx)
{
    extern __shared__ float gradPerBlock[]; // the gradient on each block
    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        gradPerBlock[i] = 0; // for 1D block
    }
    __syncthreads();

    int endIdx = sharedMemSize+startIdx;
    for(int i=blockIdx.x;i<B;i+=gridDim.x)
    {
        for(int j=blockIdx.y*blockDim.x+threadIdx.x;j<M*(C*r);j+=blockDim.x*gridDim.y)
        {
            int cout = j%(C*r); // output channel ID
            int cin = cout/r;   // input channel ID
            int m = j/(C*r);    // output point ID
            int nnSize = nnCount[i*M+m];

            for(int k=0;k<nnSize;k++)
            {
                int n = nnIndex[i*M*K+m*K+k];   // input point ID
                int f = binIndex[i*M*K+m*K+k];

                float derFilt = gradOutput[i*M*C*r+j]*input[i*N*C+n*C+cin]/nnSize;

                int currIdx = f*C*r+cout;
                if((currIdx>=startIdx) && (currIdx<endIdx)) // within the shared memory
                {
                    atomicAdd(&gradPerBlock[currIdx-startIdx],derFilt);
                }
            }
        }
    }
    __syncthreads();

    for (int i=threadIdx.x;i<sharedMemSize;i+=blockDim.x)
    {
        atomicAdd(&gradFilter[i+startIdx],gradPerBlock[i]); // for 1D block
    }
}





void DepthwiseConv3d_kernel_launcher_fast(int B, int N, int M, int C, int r, int K, const int* nnIndex,
                             const int* nnCount, const int* binIndex, const float* input,
                             const float* filter, float* output, cudaStream_t stream)
{
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    depthwise_conv3d_forward<<<blocks, threads, 0, stream>>>(B, N, M, C, r, K, nnIndex, nnCount, binIndex,
                                          input, filter, output);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

void DepthwiseConv3dGrad_kernel_launcher_fast(int B, int N, int M, int F, int C, int r, int K,
                                 const int* nnIndex, const int* nnCount, const int* binIndex,
                                 const float* input,  const float* filter, const float* gradOutput,
                                 float* gradInput, float* gradFilter, cudaStream_t stream)
{
    // titan xp has shared memory of 49152 bytes, each float value takes 4 bytes in the memory
    int maxSharedMemSize = int(49152/sizeof(float));
    // int maxSharedMemSize = 10000;
    cudaError_t err;
    dim3 blocks(32);  // blockIdx.x(col), blockIdx.y(row)
    dim3 threads(1024);
    depthwise_input_backward<<<blocks, threads, 0, stream>>>(B, N, M, F, C, r, K, nnIndex, nnCount, binIndex,
                                          input, filter, gradOutput, gradInput);

    int maxIter = (F*C*r)/maxSharedMemSize;
    int remainder = (F*C*r)%maxSharedMemSize;
    for(int iter=0;iter<maxIter;iter++)
    {
        depthwise_filter_backward<<<blocks, threads, sizeof(float)*maxSharedMemSize, stream>>>(B, N, M, F, C, r, K, nnIndex, nnCount,
                                                                              binIndex, input, gradOutput, gradFilter,
                                                                              maxSharedMemSize, maxSharedMemSize*iter);
    }
    if(remainder>0) // fill the remainder
    {
        depthwise_filter_backward<<<blocks, threads,sizeof(float)*remainder,stream>>>(B, N, M, F, C, r, K, nnIndex, nnCount,
                                                                       binIndex, input, gradOutput, gradFilter,
                                                                       remainder, maxSharedMemSize*maxIter);
    }
    err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
}

