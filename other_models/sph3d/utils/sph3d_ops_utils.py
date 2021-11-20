import torch
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn as nn
from typing import Tuple

import sph3d_ops_cuda as sph3d_ops
import pointnet2_cuda as pointnet2

import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
ROOT_DIR=os.path.abspath(os.path.join(BASE_DIR, "../../../"))
sys.path.append(ROOT_DIR)
# print(sys.path)
from utils.kcnet_utils import Netpara, debugPrint

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance
        :param ctx:
        :param xyz: (B, N, 3) where N > npoint
        :param npoint: int, number of features in the sampled set
        :return:
             output: (B, npoint) tensor containing the set
        """
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        output = torch.cuda.IntTensor(B, npoint)
        temp = torch.cuda.FloatTensor(B, N).fill_(1e10)

        pointnet2.furthest_point_sampling_wrapper(B, N, npoint, xyz, temp, output)
        return output

    @staticmethod
    def backward(xyz, a=None):
        return None, None


furthest_point_sample = FurthestPointSampling.apply


class MeanInterpolate(Function):
    """

    """
    @staticmethod
    def forward(ctx, inputs, nn_index, nn_count):
        assert inputs.is_contiguous()
        assert nn_index.is_contiguous()
        assert nn_count.is_contiguous()
        ctx.save_for_backward(inputs, nn_index, nn_count)
        return sph3d_ops.MeanInterpolate_wrapper(inputs, nn_index, nn_count)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, nn_index, nn_count = ctx.saved_tensors
        grad_input = sph3d_ops.MeanInterpolateGrad_wrapper(inputs, grad_output, nn_index, nn_count)
        return grad_input, None, None

mean_interpolate = MeanInterpolate.apply

class WeightedInterpolate(Function):
    """

    """
    @staticmethod
    def forward(ctx, inputs, weight, nn_index, nn_count):
        assert inputs.is_contiguous()
        assert weight.is_contiguous()
        assert nn_index.is_contiguous()
        assert nn_count.is_contiguous()
        ctx.save_for_backward(inputs, weight, nn_index, nn_count)
        return sph3d_ops.WeightedInterpolate_wrapper(inputs, weight, nn_index, nn_count)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, weight, nn_index, nn_count = ctx.saved_tensors
        grad_input = sph3d_ops.WeightedInterpolateGrad_wrapper(inputs, grad_output, weight, nn_index, nn_count)
        return grad_input, None, None, None

weighted_interpolate = WeightedInterpolate.apply


class MaxPool3d(Function):
    """

    """
    @staticmethod
    def forward(ctx, inputs, nn_index, nn_count):
        assert inputs.is_contiguous()
        assert nn_index.is_contiguous()
        assert nn_count.is_contiguous()
        B, N, C = inputs.size()
        M = nn_count.size(1)
        output = torch.cuda.FloatTensor(B, M, C)
        max_index = torch.cuda.IntTensor(B, M, C)

        sph3d_ops.MaxPool3d_wrapper(inputs, nn_index, nn_count, output, max_index)
        ctx.save_for_backward(inputs, max_index)
        return output, max_index
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, max_index = ctx.saved_tensors
        grad_output_data=grad_output.data.contiguous() #zhu
        grad_input = sph3d_ops.MaxPool3dGrad_wrapper(inputs, grad_output_data, max_index)
        return grad_input, None, None

max_pool3d = MaxPool3d.apply

class AvgPool3d(Function):
    """

    """
    @staticmethod
    def forward(ctx, inputs, nn_index, nn_count):
        assert inputs.is_contiguous()
        assert nn_index.is_contiguous()
        assert nn_count.is_contiguous()
        ctx.save_for_backward(inputs, nn_index, nn_count)
        return sph3d_ops.AvgPool3d_wrapper(inputs, nn_index, nn_count)
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, nn_index, nn_count = ctx.saved_tensors
        grad_input = sph3d_ops.AvgPool3dGrad_wrapper(inputs, grad_output, nn_index, nn_count)
        return grad_input, None, None

avg_pool3d = AvgPool3d.apply


class BuildSphereNeighbor(Function):
    '''
    Input:
        database: (batch, npoint, 3+x) float32 array, database points
        query:    (batch, mpoint, 3) float32 array, query points
        radius:   float32, range search radius
        dilation_rate: float32, dilation rate of range search
        nnsample: int32, maximum number of neighbors to be sampled
    Output:
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist(optional): (batch, mpoint, nnsample) float32, sqrt distance array
    '''
    @staticmethod
    def forward(ctx, database, query, radius=0.1, dilation_rate=None, nnsample=100):
        assert database.is_contiguous()
        assert query.is_contiguous()
        database = database[:,:,0:3]
        query = query[:,:,0:3]

        if dilation_rate is not None:
            radius = dilation_rate * radius

        b,n,c=database.size()
        m=query.size(1)
        nnIndex = torch.cuda.IntTensor(b,m,nnsample)
        nnCount = torch.cuda.IntTensor(b,m)
        nnDist = torch.cuda.FloatTensor(b,m,nnsample)

        sph3d_ops.BuildSphereNeighbor_wrapper(database, query, radius, nnsample, nnIndex,nnCount,nnDist)
        return nnIndex,nnCount,nnDist
        
    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None

build_sphere_neighbor = BuildSphereNeighbor.apply

class BuildCubeNeighbor(Function):
    '''
    Input:
        database: (batch, npoint, 3) float32 array, database points
        query:    (batch, mpoint, 3) float32 array, query points
        length:   float32, cube search length
        dilation_rate: float32, dilation rate of cube search
        nnsample: int32, maximum number of neighbors to be sampled
        gridsize: int32 , cubical kernel size
    Output:
        nn_index: (batch, mpoint, nnsample, 2) int32 array, neighbor and filter bin indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
    '''
    @staticmethod
    def forward(ctx, database, query, length=0.1, dilation_rate=None, nnsample=100,gridsize=3):
        assert database.is_contiguous()
        assert query.is_contiguous()
        database = database[:, :, 0:3]
        query = query[:, :, 0:3]
        if dilation_rate is not None:
            length = dilation_rate * length
        b,n,c=database.size()
        m=query.size(1)
        nnIndex = torch.cuda.IntTensor(b,m,nnsample,2)
        nnCount = torch.cuda.IntTensor(b,m)
       
        sph3d_ops.BuildCubeNeighbor_wrapper(database, query, length, nnsample, gridsize,nnIndex, nnCount)
        return nnIndex, nnCount

    @staticmethod
    def backward(ctx, *grad_output):
        return None, None, None, None, None, None

build_cube_neighbor = BuildCubeNeighbor.apply



class DepthwiseConv3d(Function):
    '''
    Input:
        input:   (batch, npoint, in_channels) float32 array, input point features
        filter: (binsize, in_channels, channel_multiplier) float32 array, convolution filter
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        bin_index: (batch, mpoint, nnsample), filtet bins' indices
    Output:
        output: (batch, mpoint, out_channels) float32 array, output point features
    '''
    @staticmethod
    def forward(ctx, inputs, filters, nn_index, nn_count, bin_index):
        assert inputs.is_contiguous()
        assert filters.is_contiguous()
        assert nn_index.is_contiguous()
        assert nn_count.is_contiguous()
        assert bin_index.is_contiguous()
        ctx.save_for_backward(inputs, filters, nn_index, nn_count, bin_index)
        return sph3d_ops.DepthwiseConv3d_wrapper(inputs, filters, nn_index, nn_count, bin_index)
        
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):# -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs, filters, nn_index, nn_count, bin_index = ctx.saved_tensors
        
        b,n,c=inputs.size()
        f,_,m=filters.size()
        grad_input = torch.cuda.FloatTensor(b,n,c).fill_(0)
        grad_filter = torch.cuda.FloatTensor(f,c,m).fill_(0)
        
        sph3d_ops.DepthwiseConv3dGrad_wrapper(inputs, filters, grad_output, nn_index, nn_count, bin_index, grad_input, grad_filter)
        return grad_input, grad_filter, None, None, None

depthwise_conv3d = DepthwiseConv3d.apply


class SphericalKernel(Function):
    '''
    Input:
        database: (batch, npoint, 3+) float32 array, database points (x,y,z,...)
        query:    (batch, mpoint, 3+) float32 array, query points (x,y,z,...)
        nn_index: (batch, mpoint, nnsample) int32 array, neighbor indices
        nn_count: (batch, mpoint) int32 array, number of neighbors
        nn_dist: (batch, mpoint, nnsample) float32, sqrt distance array
        radius:  float32, range search radius
        kernel:   list of 3 int32, spherical kernel size
    Output:
        filt_index: (batch, mpoint, nnsample) int32 array, filter bin indices
    '''
    @staticmethod
    def forward(ctx, database, query, nn_index, nn_count, nn_dist, radius, kernel=[8,2,3]):
        assert database.is_contiguous()
        assert query.is_contiguous()

        n, p, q = kernel
        database = database[:, :, 0:3]  #(x,y,z)
        query = query[:, :, 0:3] #(x,y,z)

        return sph3d_ops.SphericalKernel_wrapper(database, query, nn_index, nn_count, nn_dist, radius, n, p, q)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return None, None, None, None, None, None, None

spherical_kernel = SphericalKernel.apply

if __name__ == '__main__':
    import models.sph3d.utils.sph3gcn_util as s3g_util
    B=32
    N=1024
    C=3
    TEST_NUM=1000000
    points = torch.Tensor(B, N, C).float().cuda().normal_()
    debugPrint(points.size())
    debugPrint(points)

    conv3d=s3g_util.SeparableConv3d(3,3,33,1).cuda()

    # ##################################
    # for i in range(TEST_NUM):
    #     jitter=torch.Tensor(B, N, C).float().cuda().normal_()/100
    #     points=points+jitter
    #     net = points-jitter
    #     intra_idx, intra_cnt, intra_dst, indices=s3g_util.build_graph(points,0.3,64,512,'FPS')
    #     pooled_points=s3g_util.pool3d(points, intra_idx, intra_cnt,
    #                                   method='max',
    #                                   scope='pool')
    #     filt_idx=spherical_kernel(points, points, intra_idx, intra_cnt, intra_dst, 0.3, [8,2,2])
    #     net=conv3d(net,intra_idx, intra_cnt, filt_idx)
    #     if(i%100==0):
    #         print(i)
    #         debugPrint(filt_idx.size())
    #         debugPrint(pooled_points.size())
    #         debugPrint(net.size())

    # ################################
    # for i in range(TEST_NUM):
    #     query = torch.Tensor(B, N//4, C).float().cuda().normal_()
    #     nnIndex,nnCount,nnDist=build_sphere_neighbor(points,query,0.3,None,64)
    #     if(i%100==0):
    #         print(i)
    #         debugPrint(nnIndex.size())
    #         # debugPrint(nnCount)
    #         debugPrint(nnDist.size())
    # #########################
    # for i in range(TEST_NUM):
    #     new_points=furthest_point_sample(points,500)
        
    #     if(i%100==0):
    #         print(i)
    #         debugPrint(new_points.size())
        
    # ####################
    