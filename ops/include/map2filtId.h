/* Geometric Convolution
 * Original author: Yijie Zhu
 * All Rights Reserved. 2019.
 */
#ifndef MAP2FILTID_H
#define MAP2FILTID_H

void map2filtId_kernel_wrapper(const int bn, const int c, const float *xyz, const float *radius2, 
						const int Nfilt, int* index, int* reindex, int* indxlen);
	
#endif