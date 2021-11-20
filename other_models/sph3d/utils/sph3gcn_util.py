import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
from utils.kcnet_utils import Fc, debugPrint
from sph3d_ops_utils import furthest_point_sample, weighted_interpolate, max_pool3d, avg_pool3d, build_sphere_neighbor,\
                    build_cube_neighbor, depthwise_conv3d, spherical_kernel
neighbor_fn = build_sphere_neighbor # default nn search method


def build_global_graph(xyz, query, radius):
    nn_uplimit = xyz.size(1)
    nn_idx, nn_cnt, nn_dst = neighbor_fn(xyz, query, radius, None, nn_uplimit)

    return nn_idx, nn_cnt, nn_dst


def build_graph(xyz, radius, nn_uplimit, num_sample, sample_method=None):
    intra_idx, intra_cnt, intra_dst = neighbor_fn(xyz, xyz, radius, None,nn_uplimit)

    if num_sample is not None:
        if sample_method == 'random':
            # sample_index = random_sample(num_sample, xyz)
            raise NotImplementedError
        elif sample_method == 'FPS':
            sample_index = furthest_point_sample(xyz,num_sample)
        elif sample_method == 'IDS':
            # prob = tf.divide(tf.reduce_sum(intra_dst, axis=-1), tf.cast(intra_cnt,dtype=tf.float32))
            # sample_index = inverse_density_sample(num_sample, prob)
            raise NotImplementedError
        else:
            raise ValueError('Unknown sampling method.')

        # batch_size = xyz.size(0)
        # batch_indices = torch.arange(batch_size).type_as(sample_index).view(-1, 1, 1).repeat(1, num_sample, 1) #b,num_sample,1
        # indices = torch.cat([batch_indices, sample_index.unsqueeze(2)], dim=2)
        indices = sample_index # (b,num_sample),int32
    else:
        indices = None

    return intra_idx, intra_cnt, intra_dst, indices

class SeparableConv3d(nn.Module):
    def __init__(self,
                input_channels,
                output_channels,
                kernel_size,
                depth_multiplier,
                scope=None,
                bn='BN', 
                bn_decay=None, 
                activation_fn="relu"):
        """ 3D separable convolution with non-linear operation.
        Args:
            inputs: 3-D tensor variable BxNxC
            num_out_channels: int
            kernel_size: int
            depth_multiplier: int
            scope: string
            nn_index: int32 array, neighbor indices
            nn_count: int32 array, number of neighbors
            filt_index: int32 array, filter bin indices
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            weight_decay: float
            activation_fn: function
            with_bn: bool, whether to use batch norm
            is_training: bool Tensor variable

        Returns:
          Variable tensor
        """
        super(SeparableConv3d, self).__init__()
        self.input_channels=input_channels
        self.output_channels=output_channels
        self.depthwise_kernel_shape = [kernel_size, input_channels, depth_multiplier]
        self.scope = scope
        
        self.depthwise_kernel = nn.Parameter(torch.Tensor(*self.depthwise_kernel_shape))
        # debugPrint(self.depthwise_kernel.size())
        
        self.FC=Fc(input_channels*depth_multiplier,[output_channels],input_dim=3, bn=bn, activation_fn=activation_fn) # out_c=32, b,n,in_c-->b,n,out_c
        self.reset_parameters()

    def forward(self, inputs, nn_index, nn_count, filt_index):
        outputs = depthwise_conv3d(inputs, self.depthwise_kernel, nn_index, nn_count, filt_index) # （b,n,in_c）-->(b,m,out_c)
        outputs = self.FC(outputs)
        outputs=outputs.contiguous()
        return outputs   #(b,m,out_c)

    def reset_parameters(self):
        self.depthwise_kernel.data.uniform_(-0.2, 0.2)


def pool3d(inputs, nn_index, nn_count, scope, method):
    """ 3D pooling.

    Args:
        inputs: 3-D tensor BxNxC
        nn_index: int32 array, neighbor and filter bin indices
        nn_count: int32 array, number of neighbors
        scope: string
        method: string, the pooling method

    Returns:
        Variable tensor
    """

    if method == 'max':
        outputs, max_index = max_pool3d(inputs, nn_index, nn_count)
    elif method == 'avg':
        outputs = avg_pool3d(inputs, nn_index, nn_count)
    else:
        raise ValueError("Unknow pooling method %s." % method)

    return outputs


      
###############################################
def build_graph_deconv(xyz, xyz_unpool, radius, nn_uplimit):
    intra_idx, intra_cnt, intra_dst = neighbor_fn(xyz, xyz, radius=radius,
                                                  nnsample=nn_uplimit)
    inter_idx, inter_cnt, inter_dst = neighbor_fn(xyz, xyz_unpool, radius=radius,
                                                  nnsample=nn_uplimit)

    return intra_idx, intra_cnt, intra_dst, inter_idx, inter_cnt, inter_dst


def unpool3d(inputs, nn_index, nn_count, nn_dist, scope, method):
    """ 3D unpooling

    Args:
        inputs: 3-D tensor BxNxC
        nn_index: int32 array, neighbor indices
        nn_count: int32 array, number of neighbors
        nn_dist: float32 array, neighbor (sqrt) distances for weight computation
        scope: string
        method: string, the unpooling method

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        if method == 'mean':
            outputs = tf_unpool3d.mean_interpolate(inputs, nn_index, nn_count)
        elif method == 'weighted':
            sum_nn_dist = tf.reduce_sum(nn_dist, axis=-1, keepdims=True)
            epsilon = 1e-7
            weight = tf.divide(nn_dist+epsilon, sum_nn_dist+epsilon)
            outputs = tf_unpool3d.weighted_interpolate(inputs, weight, nn_index, nn_count)
        else:
            raise ValueError("Unknow unpooling method %s." % method)

        return outputs

