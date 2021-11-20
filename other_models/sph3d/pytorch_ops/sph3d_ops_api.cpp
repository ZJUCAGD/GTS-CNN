#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include "cuda_utils.h"
#include "unpooling/unpool3d_gpu.h"
#include "pooling/pool3d_gpu.h"
#include "nnquery/nnquery_gpu.h"
#include "convolution/conv3d_gpu.h"
#include "buildkernel/buildkernel_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	
	m.def("DepthwiseConv3d_wrapper", &DepthwiseConv3d_wrapper_fast, "DepthwiseConv3d_wrapper_fast");
    m.def("DepthwiseConv3dGrad_wrapper", &DepthwiseConv3dGrad_wrapper_fast, "DepthwiseConv3dGrad_wrapper_fast");

	m.def("SphericalKernel_wrapper", &SphericalKernel_wrapper_fast, "SphericalKernel_wrapper_fast");
  
    m.def("BuildSphereNeighbor_wrapper", &BuildSphereNeighbor_wrapper_fast, "BuildSphereNeighbor_wrapper_fast");
    m.def("BuildCubeNeighbor_wrapper", &BuildCubeNeighbor_wrapper_fast, "BuildCubeNeighbor_wrapper_fast");

    m.def("MaxPool3d_wrapper", &MaxPool3d_wrapper_fast, "MaxPool3d_wrapper_fast");
    m.def("MaxPool3dGrad_wrapper", &MaxPool3dGrad_wrapper_fast, "MaxPool3dGrad_wrapper_fast");
    m.def("AvgPool3d_wrapper", &AvgPool3d_wrapper_fast, "AvgPool3d_wrapper_fast");
    m.def("AvgPool3dGrad_wrapper", &AvgPool3dGrad_wrapper_fast, "AvgPool3dGrad_wrapper_fast");
    
    m.def("MeanInterpolate_wrapper", &MeanInterpolate_wrapper_fast, "MeanInterpolate_wrapper_fast");
    m.def("MeanInterpolateGrad_wrapper", &MeanInterpolateGrad_wrapper_fast, "MeanInterpolateGrad_wrapper_fast");
    m.def("WeightedInterpolate_wrapper", &WeightedInterpolate_wrapper_fast, "WeightedInterpolate_wrapper_fast");
    m.def("WeightedInterpolateGrad_wrapper", &WeightedInterpolateGrad_wrapper_fast, "WeightedInterpolateGrad_wrapper_fast");
}
