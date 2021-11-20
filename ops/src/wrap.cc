#include <vector>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"
#include "batch_knn.h"
#include "graph_pooling.h"
#include "group_points.h"
#include "kernel_correlation.h"
#include "aggregate.h"
#include "map2filtId.h"  // 2019.5.20

std::vector<at::Tensor> batch_knn_wrapper(at::Tensor query_tensor, at::Tensor reference_tensor, int npoints) {
  CHECK_INPUT(query_tensor);
  CHECK_INPUT_TYPE(query_tensor, at::ScalarType::Float);
  CHECK_INPUT(reference_tensor);
  CHECK_INPUT_TYPE(reference_tensor, at::ScalarType::Float);
  int b = query_tensor.size(0);
  int n = query_tensor.size(1);
  int c = query_tensor.size(2);
  int m = reference_tensor.size(1);
  // AT_ASSERT(m > npoints, "npoints must smaller than m")
  AT_ASSERTM(m > npoints, "npoints must smaller than m")
  // at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, n, npoints});
  // at::Tensor idxs_tensor = torch::CUDA(at::kInt).zeros({b, n, npoints});
  at::Tensor idxs_tensor = at::zeros({b, n, npoints}, torch::CUDA(at::kInt));
  at::Tensor dists_tensor = at::zeros({b, n, npoints}, torch::CUDA(at::kFloat));
  at::Tensor temp_tensor = at::zeros({b, n, m}, torch::CUDA(at::kFloat));
  const float *query = query_tensor.data<float>();
  const float *reference = reference_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();
  float *dists = dists_tensor.data<float>();
  float *temp = temp_tensor.data<float>();

  batch_knn_kernel_wrapper(b, n, c, m, npoints, query, reference, idxs, dists, temp);
  return {idxs_tensor, dists_tensor};
}

std::vector<at::Tensor> graph_max_pooling_wrapper(at::Tensor features_tensor, at::Tensor knn_graph_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  int b = features_tensor.size(0);
  int c = features_tensor.size(1);
  int n = features_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor outputs_tensor = at::zeros({b, c, n}, torch::CUDA(at::kFloat));
  at::Tensor idxs_tensor = at::zeros({b, c, n}, torch::CUDA(at::kInt));
  const float *features = features_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  float *outputs = outputs_tensor.data<float>();
  int *idxs = idxs_tensor.data<int>();

  graph_max_pooling_kernel_wrapper(b, c, n, npoints, features, knn_graph, outputs, idxs);
  return {outputs_tensor, idxs_tensor};
}

at::Tensor graph_max_pooling_grad_wrapper(at::Tensor grad_outputs_tensor, at::Tensor idxs_tensor) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  CHECK_INPUT(idxs_tensor);
  CHECK_INPUT_TYPE(idxs_tensor, at::ScalarType::Int);
  int b = grad_outputs_tensor.size(0);
  int c = grad_outputs_tensor.size(1);
  int n = grad_outputs_tensor.size(2);
  at::Tensor grad_inputs_tensor = at::zeros({b, c, n}, torch::CUDA(at::kFloat));
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  const int *idxs = idxs_tensor.data<int>();
  float *grad_inputs = grad_inputs_tensor.data<float>();

  graph_max_pooling_grad_kernel_wrapper(b, c, n, grad_outputs, idxs, grad_inputs);
  return grad_inputs_tensor;
}

at::Tensor graph_pooling_wrapper(at::Tensor features_tensor, at::Tensor knn_graph_tensor, at::Tensor weights_tensor) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = features_tensor.size(0);
  int c = features_tensor.size(1);
  int n = features_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor outputs_tensor = at::zeros({b, c, n}, torch::CUDA(at::kFloat));
  const float *features = features_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *outputs = outputs_tensor.data<float>();

  graph_pooling_kernel_wrapper(b, c, n, npoints, features, knn_graph, weights, outputs);
  return outputs_tensor;
}

at::Tensor graph_pooling_grad_wrapper(at::Tensor grad_outputs_tensor, at::Tensor knn_graph_tensor, at::Tensor weights_tensor) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  CHECK_INPUT(knn_graph_tensor);
  CHECK_INPUT_TYPE(knn_graph_tensor, at::ScalarType::Int);
  CHECK_INPUT(weights_tensor);
  CHECK_INPUT_TYPE(weights_tensor, at::ScalarType::Float);
  int b = grad_outputs_tensor.size(0);
  int c = grad_outputs_tensor.size(1);
  int n = grad_outputs_tensor.size(2);
  int npoints = knn_graph_tensor.size(2);
  at::Tensor grad_inputs_tensor = at::zeros({b, c, n}, torch::CUDA(at::kFloat));
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  const int *knn_graph = knn_graph_tensor.data<int>();
  const float *weights = weights_tensor.data<float>();
  float *grad_inputs = grad_inputs_tensor.data<float>();

  graph_pooling_grad_kernel_wrapper(b, c, n, npoints, grad_outputs, knn_graph, weights, grad_inputs);
  return grad_inputs_tensor;
}

at::Tensor group_points_wrapper(at::Tensor points_tensor, at::Tensor group_idxs_tensor) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  CHECK_INPUT(group_idxs_tensor);
  CHECK_INPUT_TYPE(group_idxs_tensor, at::ScalarType::Int);
  int b = points_tensor.size(0);
  int c = points_tensor.size(1);
  int n = points_tensor.size(2);
  int m = group_idxs_tensor.size(1);
  int k = group_idxs_tensor.size(2);
  at::Tensor out_tensor = at::zeros({b, c, m, k}, torch::CUDA(at::kFloat));
  const float *points = points_tensor.data<float>();
  const int *group_idxs = group_idxs_tensor.data<int>();
  float *out = out_tensor.data<float>();

  group_points_kernel_wrapper(b, c, n, m, k, points, group_idxs, out);
  return out_tensor;
}

at::Tensor group_points_grad_wrapper(at::Tensor grad_out_Tensor, at::Tensor group_idxs_tensor, int n) {
  CHECK_INPUT(grad_out_Tensor);
  CHECK_INPUT_TYPE(grad_out_Tensor, at::ScalarType::Float);
  CHECK_INPUT(group_idxs_tensor);
  CHECK_INPUT_TYPE(group_idxs_tensor, at::ScalarType::Int);
  int b = grad_out_Tensor.size(0);
  int c = grad_out_Tensor.size(1);
  int m = grad_out_Tensor.size(2);
  int k = grad_out_Tensor.size(3);
  at::Tensor grad_points_tensor = at::zeros({b, c, n}, torch::CUDA(at::kFloat));
  const float *grad_out = grad_out_Tensor.data<float>();
  const int *group_idxs = group_idxs_tensor.data<int>();
  float *grad_points = grad_points_tensor.data<float>();

  group_points_grad_kernel_wrapper(b, c, n, m, k, grad_out, group_idxs, grad_points);
  return grad_points_tensor;
}

at::Tensor kernel_correlation_wrapper(at::Tensor points_tensor, at::Tensor kernel_tensor, float sigma) {
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  CHECK_INPUT(kernel_tensor);
  CHECK_INPUT_TYPE(kernel_tensor, at::ScalarType::Float);
  int b = points_tensor.size(0);
  int n = points_tensor.size(1);
  int npoints = points_tensor.size(2);
  int l = kernel_tensor.size(0);
  int m = kernel_tensor.size(1);
  at::Tensor outputs_tensor = at::zeros({b, l, n}, torch::CUDA(at::kFloat));
  const float *points = points_tensor.data<float>();
  const float *kernel = kernel_tensor.data<float>();
  float *outputs = outputs_tensor.data<float>();

  kernel_correlation_kernel_wrapper(b, n, npoints, l, m, sigma, points, kernel, outputs);
  return outputs_tensor;
}

at::Tensor kernel_correlation_grad_wrapper(at::Tensor grad_outputs_tensor, at::Tensor points_tensor, at::Tensor kernel_tensor, float sigma) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  CHECK_INPUT(points_tensor);
  CHECK_INPUT_TYPE(points_tensor, at::ScalarType::Float);
  CHECK_INPUT(kernel_tensor);
  CHECK_INPUT_TYPE(kernel_tensor, at::ScalarType::Float);
  int b = points_tensor.size(0);
  int n = points_tensor.size(1);
  int npoints = points_tensor.size(2);
  int l = kernel_tensor.size(0);
  int m = kernel_tensor.size(1);
  at::Tensor grad_inputs_tensor = at::zeros({l, m, 3}, torch::CUDA(at::kFloat));
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  const float *points = points_tensor.data<float>();
  const float *kernel = kernel_tensor.data<float>();
  float *grad_inputs = grad_inputs_tensor.data<float>();

  kernel_correlation_grad_kernel_wrapper(b, n, npoints, l, m, sigma, grad_outputs, points, kernel, grad_inputs);
  return grad_inputs_tensor;
}


std::vector<at::Tensor> aggregate_wrapper(at::Tensor features_tensor, at::Tensor xyz_tensor, 
                                          float radius, float decay_radius, float delta) {
  CHECK_INPUT(features_tensor);
  CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT_TYPE(xyz_tensor, at::ScalarType::Float);
  int b = features_tensor.size(0);
  int n = features_tensor.size(1);
  int c = features_tensor.size(2) / 6;
  //int npoints = xyz_tensor.size(2);
  at::Tensor outputs_tensor = at::zeros({b, n, c}, torch::CUDA(at::kFloat));
  at::Tensor norm_tensor = at::zeros({b, n}, torch::CUDA(at::kFloat));
  const float *features = features_tensor.data<float>();
  const float *xyz = xyz_tensor.data<float>();
  float *outputs = outputs_tensor.data<float>();
  float *norm = norm_tensor.data<float>();

  aggregate_kernel_wrapper(b, n, c, features, xyz, outputs, norm, radius, decay_radius, delta);
  return {outputs_tensor, norm_tensor};
}

at::Tensor aggregate_grad_wrapper(at::Tensor grad_outputs_tensor,at::Tensor features_tensor, 
                                  at::Tensor xyz_tensor,// at::Tensor norm_tensor,
                                  float radius, float decay_radius, float delta) {
  CHECK_INPUT(grad_outputs_tensor);
  CHECK_INPUT_TYPE(grad_outputs_tensor, at::ScalarType::Float);
  // CHECK_INPUT(features_tensor);
  // CHECK_INPUT_TYPE(features_tensor, at::ScalarType::Float);
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT_TYPE(xyz_tensor, at::ScalarType::Float);
  // CHECK_INPUT(outputs_tensor);
  // CHECK_INPUT_TYPE(outputs_tensor, at::ScalarType::Float);
  // CHECK_INPUT(norm_tensor);
  // CHECK_INPUT_TYPE(norm_tensor, at::ScalarType::Float);
  int b = features_tensor.size(0);
  int n = features_tensor.size(1);
  int c = features_tensor.size(2) / 6;
  
  const float *grad_outputs = grad_outputs_tensor.data<float>();
  // const float *features = features_tensor.data<float>();
  const float *xyz = xyz_tensor.data<float>();
  // const float *outputs = outputs_tensor.data<float>();
  at::Tensor norm_tensor = at::zeros({b, n}, torch::CUDA(at::kFloat));
  float *norm = norm_tensor.data<float>();

  at::Tensor grad_inputs_tensor = at::zeros({b, n, c*6}, torch::CUDA(at::kFloat));
  float *grad_inputs = grad_inputs_tensor.data<float>();

  aggregate_grad_kernel_wrapper(b, n, c, xyz, grad_outputs, norm, grad_inputs, radius, decay_radius, delta);
  return grad_inputs_tensor;
}


std::vector<at::Tensor> map2filtId_wrapper(at::Tensor xyz_tensor, at::Tensor radius2_tensor) {
  CHECK_INPUT(xyz_tensor);
  CHECK_INPUT_TYPE(xyz_tensor, at::ScalarType::Float);
  CHECK_INPUT(radius2_tensor);
  CHECK_INPUT_TYPE(radius2_tensor, at::ScalarType::Float);
  int bn = xyz_tensor.size(0);
  int c = xyz_tensor.size(1);
  int Nfilt = radius2_tensor.size(0);
  // AT_ASSERT(m > npoints, "npoints must smaller than m")
  // AT_ASSERTM(m > npoints, "npoints must smaller than m")
  // at::Tensor idxs_tensor = at::zeros(torch::CUDA(at::kInt), {b, n, npoints});
  // at::Tensor idxs_tensor = torch::CUDA(at::kInt).zeros({b, n, npoints});
  at::Tensor index_tensor = at::zeros({bn}, torch::CUDA(at::kInt));
  at::Tensor reindex_tensor = at::zeros({bn}, torch::CUDA(at::kInt));
  at::Tensor indexlen_tensor = at::zeros({Nfilt+1}, torch::CUDA(at::kInt));

  const float *xyz = xyz_tensor.data<float>();
  const float *radius2 = radius2_tensor.data<float>();

  int *index = index_tensor.data<int>();
  int *reindex = reindex_tensor.data<int>();
  int *indexlen = indexlen_tensor.data<int>();
  // printf("into kernel wrapper...\n");
  map2filtId_kernel_wrapper(bn, c, xyz, radius2, Nfilt, index, reindex, indexlen);
  // printf("wrapper success!\n");
  return {index_tensor, reindex_tensor, indexlen_tensor};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batch_knn_wrapper", &batch_knn_wrapper, "batch knn");
    m.def("graph_max_pooling_wrapper", &graph_max_pooling_wrapper, "graph max pooling");
    m.def("graph_max_pooling_grad_wrapper", &graph_max_pooling_grad_wrapper, "graph max pooling grad");
    m.def("graph_pooling_wrapper", &graph_pooling_wrapper, "graph pooling");
    m.def("graph_pooling_grad_wrapper", &graph_pooling_grad_wrapper, "graph pooling grad");
    m.def("group_points_wrapper", &group_points_wrapper, "group points");
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper, "group points grad");
    m.def("kernel_correlation_wrapper", &kernel_correlation_wrapper, "kernel correlation");
    m.def("kernel_correlation_grad_wrapper", &kernel_correlation_grad_wrapper, "kernel correlation grad");
    m.def("aggregate_wrapper", &aggregate_wrapper, "aggregate"); // 
    m.def("aggregate_grad_wrapper", &aggregate_grad_wrapper, "aggregate grad");
    m.def("map2filtId_wrapper", &map2filtId_wrapper, "map2filtId");
}