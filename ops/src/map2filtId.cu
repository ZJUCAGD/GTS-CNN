#include "cuda_utils.h"
#include "map2filtId.h"



__global__ void map2filtId_kernel(const int bn, const int c,
                                 const float* xyz,
                                 const float* radius2,
                                 const int Nfilt,
                                 int* index,
                                 int* reindex,
                                 int* indexlen) {
  const int batch_index = blockIdx.x;
  xyz += batch_index * (bn/gridDim.x) *c;
  reindex += batch_index * (bn/gridDim.x);
  // idxs += batch_index * n * npoints;
  // dists += batch_index * n * npoints;
  // temp += batch_index * n * m;

  for (int i = threadIdx.x; i < bn/gridDim.x; i += blockDim.x) {
    const float *q = xyz + i * c;
    float tmp=0;
    for(int j=0;j<c;++j){
    	tmp+=q[j]*q[j];
    }
    for(int j=0;j<Nfilt;++j){
    	if(tmp<radius2[j]){
    		reindex[i]=j;
    		atomicAdd(indexlen+j+1,1); 
    		break;
    	}
    }
  }
}

__global__ void indexlen_cum(const int bn, const int Nfilt, int* index, int* reindex, int* indexlen){
	for(int i=1;i<=Nfilt;++i){
		indexlen[i]=indexlen[i]+indexlen[i-1];
	}
	int offset[10]={0,0,0,0,0,0,0,0,0,0};
	for(int i=0;i<bn;++i){
		int ballid=reindex[i];
		index[indexlen[ballid]+offset[ballid]]=i;
		++offset[ballid];
	}
}


__global__ void compute_reindex(const int bn, const int* index, int* reindex){
    // const int batch_index = blockIdx.x;
    
    const int offset = blockIdx.x * (bn/gridDim.x);
    
    for (int i = threadIdx.x; i < bn/gridDim.x; i += blockDim.x) {
      reindex[index[offset+i]]=offset+i;
  	}

}

void map2filtId_kernel_wrapper(const int bn, const int c, const float *xyz, const float *radius2, 
						const int Nfilt, int* index, int* reindex, int* indexlen){
	// printf("%d, %d\n", bn,c);
	int grid_dim=1;
	if(bn<32) grid_dim=1;
	else grid_dim=32;
	int thread_dim=opt_n_threads(bn/grid_dim);
  cudaError_t err;
  map2filtId_kernel<<<grid_dim, thread_dim>>>(bn, c, xyz, radius2, Nfilt, index, reindex, indexlen);
  indexlen_cum<<<1, 1>>>(bn, Nfilt, index, reindex, indexlen);
  compute_reindex<<<grid_dim, thread_dim>>>(bn, index, reindex);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
  // reindex 's val record the ball id each point loacate at.
  //2 runs to scan reindex to get index
  
  // for(int i=0;i<bn;++i){
  // 	indexlen[reindex[i]+1]++;
  // }
  // for(int i=1;i<=Nfilt;++i){
  // 	indexlen[i]=indexlen[i]+indexlen[i-1];
  // }

  // int offset[10]={0,0,0,0,0,0,0,0,0,0};
  // for(int i=0;i<bn;++i){
  // 	int ballid=reindex[i];
  // 	index[indexlen[ballid]+offset[ballid]]=i;
  // 	++offset[ballid];
  // }
  // // to get reindex   mid_row-->i---->mid_row
  // for(int i=0;i<bn;++i){
  // 	int mid_row=index[i];
  // 	reindex[mid_row]=i;
  // }
  // printf("kernel wrapper success!\n");
  
}