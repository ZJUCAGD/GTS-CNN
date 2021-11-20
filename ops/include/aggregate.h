/* Geometric Convolution
 * Original author: Yijie Zhu
 * All Rights Reserved. 2019.
 */
#ifndef AGGREGATE_H
#define AGGREGATE_H

void aggregate_kernel_wrapper(const int b, const int n, const int c, const float *features, const float *xyz, 
							float *outputs, float *norm, 
							const float radius, const float decay_radius, const float delta);
void aggregate_grad_kernel_wrapper(const int b, const int n, const int c, const float *xyz,
							 	const float *grad_outputs, float *norm, float *grad_inputs, 
							 	const float radius, const float decay_radius, const float delta);
#endif