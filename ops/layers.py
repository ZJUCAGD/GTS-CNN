import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch._jit_internal import weak_module, weak_script_method
from torch.autograd import Function
from torch.utils.cpp_extension import load

extension_ = load(name='extension_',
                  sources=[
                      'src/wrap.cc', 'src/batch_knn.cu',
                      'src/graph_pooling.cu', 'src/group_points.cu',
                      'src/kernel_correlation.cu', 'src/aggregate.cu',
                      'src/map2filtId.cu'
                  ],
                  extra_include_paths=[
                      '/usr/local/cuda/include',
                      os.path.join(os.getcwd(), '../ops', 'include')
                  ],
                  verbose=True)


class BatchKnn(Function):
    """


    :param query: (B, C, N)
           reference: (B, C, M)
           npoints: nearest points numbers
    :return: idxs: (B, N, npoints)
             dists: (B, N, npoints)
    """
    @staticmethod
    def forward(ctx, query, reference, npoints):
        query_t = query.transpose(1, 2).contiguous()
        reference_t = reference.transpose(1, 2).contiguous()
        idxs, dists = extension_.batch_knn_wrapper(query_t, reference_t,
                                                   npoints)

        return idxs, dists

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None, None


batch_knn = BatchKnn.apply


class GroupPoints(Function):
    """
    simpling points

    :param features: (B, C, N)
           group_idxs: (B, M, K)
    :return: out: (B, C, M, K)
    """
    @staticmethod
    def forward(ctx, features, group_idxs):
        out = extension_.group_points_wrapper(features, group_idxs)
        ctx.save_for_backward(features, group_idxs)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        features, group_idxs = ctx.saved_tensors
        n = features.size(2)
        grad_inputs = extension_.group_points_grad_wrapper(
            grad_out.data.contiguous(), group_idxs, n)

        return grad_inputs, None, None, None, None


group_points = GroupPoints.apply


class GraphPooling(Function):
    """


    :param features: (B, C, N)
           knn_graph: (B, N, npoints)
           weights: (B, N, npoints)
    :return: outputs: (B, C, N)
    """
    @staticmethod
    def forward(ctx, features, knn_graph, weights):
        outputs = extension_.graph_pooling_wrapper(features, knn_graph,
                                                   weights)
        ctx.save_for_backward(knn_graph, weights)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        knn_graph, weights = ctx.saved_tensors
        grad_inputs = \
            extension_.graph_pooling_grad_wrapper(grad_outputs.contiguous(), knn_graph, weights)
        return grad_inputs, None, None


graph_pooling = GraphPooling.apply


class GraphMaxPooling(Function):
    """


    :param features: (B, C, N)
           knn_graph: (B, N, npoints)
    :return: outputs: (B, C, N)
    """
    @staticmethod
    def forward(ctx, features, knn_graph):
        outputs, idxs = extension_.graph_max_pooling_wrapper(
            features, knn_graph)
        ctx.save_for_backward(idxs)
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        idxs = ctx.saved_tensors[0]
        grad_inputs = extension_.graph_max_pooling_grad_wrapper(
            grad_outputs.contiguous(), idxs)
        return grad_inputs, None


graph_max_pooling = GraphMaxPooling.apply


class KernelCorrelation(Function):
    """


    :param points: (B, N, npoints, 3)
           kernel: (L, m, 3)
           sigma: gauss kernel sigma
    :return: outputs: (B, L, N)
    """
    @staticmethod
    def forward(ctx, points, kernel, sigma):
        outputs = extension_.kernel_correlation_wrapper(points, kernel, sigma)
        ctx.save_for_backward(points, kernel)
        ctx.sigma = sigma
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        points, kernel = ctx.saved_tensors
        grad_inputs = extension_.kernel_correlation_grad_wrapper(
            grad_outputs.contiguous(), points, kernel, ctx.sigma)
        return None, grad_inputs, None


kernel_correlation = KernelCorrelation.apply


class LoaclGeometricStructure(nn.Module):
    """


    :param points: (B, 3, N)
           knn_graph: (B, N, npoints)
           sigma: gauss kernel sigma
    :return: out: (B, out_channels, N)
    """
    def __init__(self, out_channels, kernel_size, sigma):
        super(LoaclGeometricStructure, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.kernel = nn.Parameter(torch.Tensor(out_channels, kernel_size, 3))

        self.reset_parameters()

    def reset_parameters(self):
        self.kernel.data.uniform_(-0.2, 0.2)
        return

    def forward(self, points, knn_graph):
        assert (points.size(2) == knn_graph.size(1))
        assert (knn_graph.size(2) == self.kernel_size)
        x = group_points(points, knn_graph)  #   ---> (B,3,N,npoints)
        x = x - points.unsqueeze(3)
        x = x.transpose(1,
                        2).transpose(2,
                                     3).contiguous()  #--->(B, N, npoints, 3)
        outputs = kernel_correlation(x, self.kernel, self.sigma)
        return outputs


#############################################
class Aggregate(Function):
    '''
    inputs:
        feature: batch_size * num_points * num_channels     float32
        xyz: batch_size * num_points * 3                    float32
        radius:                                             float32
        decay_radius:                                       float32
        delta                                               int
    returns:
        output feature: batch_size * num_points * num_channels  float32
        norm feature: batch_size * num_points
    '''
    @staticmethod
    def forward(ctx, feat, xyz, radius, decay_radius, delta=0):
        feat = feat.contiguous()
        xyz = xyz.contiguous()
        outputs, norm_buffer = extension_.aggregate_wrapper(
            feat, xyz, radius, decay_radius, delta)
        ctx.save_for_backward(feat, xyz)
        ctx.radius = radius
        ctx.decay_radius = decay_radius
        ctx.delta = delta
        return outputs, norm_buffer

    @staticmethod
    def backward(ctx, *grad_outputs):
        feat, xyz = ctx.saved_tensors
        grad_feat = extension_.aggregate_grad_wrapper(
            grad_outputs[0].contiguous(), feat, xyz, ctx.radius,
            ctx.decay_radius, ctx.delta)
        return grad_feat, None, None, None, None


aggregate = Aggregate.apply


class Map2filtId(Function):
    '''
    inputs:

    '''
    @staticmethod
    def forward(ctx, xyz, radius):
        xyz = xyz.contiguous()
        radius2 = torch.pow(radius, 2)
        # print(radius2)
        index, reindx, indxlen = extension_.map2filtId_wrapper(xyz, radius2)
        return index, reindx, indxlen

    @staticmethod
    def backward(ctx, *grad_outputs):
        return None, None, None


map2f = Map2filtId.apply


class Fc(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels_list,
                 input_dim=3,
                 bn='BN',
                 activation_fn='relu'):
        """
        bn: False/str, type of Normalization layer, default='BN'(batch norm), others:'GN'(group norm)
        activation_fn: None/str, type of the activation function, defalut=='relu',others:'elu'
        """
        super(Fc, self).__init__()
        self.bn = bn
        self.bns = None
        self.output_channels_list = output_channels_list
        self.channels_list = [input_channels] + output_channels_list
        self.mlp = nn.ModuleList([
            nn.Linear(self.channels_list[idx - 1], c, bias=(self.bn == False))
            for idx, c in enumerate(self.channels_list) if idx >= 1
        ])
        if bn == 'BN':
            if input_dim < 4:
                self.bns = nn.ModuleList(
                    [nn.BatchNorm1d(c) for c in output_channels_list])
            else:
                self.bns = nn.ModuleList(
                    [nn.BatchNorm2d(c) for c in output_channels_list])
            self.bns.cuda()
        elif bn == 'GN':
            self.bns = nn.ModuleList(
                [nn.GroupNorm(8, c) for c in output_channels_list])
            self.bns.cuda()
        self.mlp.cuda()
        if activation_fn == 'relu':
            self.act = nn.functional.relu
        elif activation_fn == 'elu':
            self.act = nn.functional.elu

    def forward(self, inputs):
        if self.bn == False:
            for mlp_i in self.mlp:
                inputs = mlp_i(inputs)
                # inputs=self.act(inputs, inplace=True)
                # print('{} size = {}'.format(idx,inputs.size()))
        else:
            for idx, mlp_i in enumerate(self.mlp):
                inputs = mlp_i(inputs)
                inputs = inputs.transpose(1, -1)  #--->(b,c,n)
                inputs = self.bns[idx](inputs)
                # inputs=nn.functional.relu(inputs, inplace=True)
                inputs = self.act(inputs, inplace=True)
                inputs = inputs.transpose(1, -1)  #---->(b,n,c)
                # print('bn {} size = {}'.format(idx,inputs.size()))
        return inputs


class Perceptron(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 weight_decay=None,
                 bn=False,
                 activation_fn=None):
        super(Perceptron, self).__init__()
        self.input_channels = input_channels
        self.linear_layer = nn.Linear(input_channels,
                                      output_channels)  #,bias=False)

    def forward(self, inputs):
        # assert (inputs.size(2) == self.input_channels)
        outputs = self.linear_layer(inputs)
        return outputs


class Geoconv(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 bypass_num_outputs,
                 radius,
                 decay_radius,
                 bn=True,
                 bn_decay=None,
                 activation_fn="relu",
                 delta=False):
        ''' GeoCNN Geo-Conv
        Input:
            feat: (batch_size, num_point, input_channels) TF tensor
            points: (batch_size, num_point, 3) TF tensor
            num_outputs: the count of output channels
            bypass_num_outputs: the count of output channels of bypass
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        '''
        super(Geoconv, self).__init__()
        self.radius = radius
        self.decay_radius = decay_radius
        self.delta = delta
        # self.perceptron_feat = Perceptron(input_channels, output_channels)
        self.perceptron_feat = nn.Linear(input_channels, output_channels)
        self.perceptron_bypass = Fc(input_channels, [bypass_num_outputs * 6])
        # ag, _ = aggregate(mutual, xyz, radius, decay_radius, delta)  #(b,n,bypass*6)--->(b,n,bypass)
        self.bn1 = nn.Sequential(nn.BatchNorm1d(bypass_num_outputs),
                                 nn.ReLU(True))
        # self.perceptron_ag = Perceptron(bypass_num_outputs, output_channels)
        self.perceptron_ag = nn.Linear(bypass_num_outputs, output_channels)
        self.bn2 = nn.Sequential(nn.BatchNorm1d(output_channels),
                                 nn.ReLU(True))

    def forward(self, feat, xyz):
        self_feat = self.perceptron_feat(feat)  #(B,N,12)--->(B,N,64)
        mutual = self.perceptron_bypass(feat)  #(B,N,12)--->(B,N,32*6)
        ag, _ = aggregate(mutual, xyz, self.radius, self.decay_radius,
                          self.delta)  #(B,N,192)--->(B,N,32)
        ag = ag.transpose(1, 2)  #(B,N,32)--->(B,32,N)
        ag = self.bn1(ag)
        ag = ag.transpose(1, 2)  #(B,32,N)--->(B,N,32)
        ag = self.perceptron_ag(ag)  #(b,n,32)--->(b,n,64)
        outputs = self_feat + ag
        outputs = outputs.transpose(1, 2)
        outputs = self.bn2(outputs)
        outputs = outputs.transpose(1, 2)
        # print('outputs {}'.format(outputs.size()))
        return outputs  #(b,n,64)


class SphericalLinear(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 radius=None,
                 bias=True):
        ''' SphericalLinear module/layer
        Input:
            feat: (batch_size, num_point, input_channels) TF tensor
            points: (batch_size, num_point, 3) TF tensor
            input_channels: the count of input channels
            outputs_channels: the count of output channels
            radius: the split radius list of balls, each of which coressponds a linear layer. Default: [10.0]
            bias: linear layer need or not need bias vector
        '''
        super(SphericalLinear, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        if radius is None:
            self.radius = [100]  #None
            self.Nfilt = 1
        else:
            self.radius = radius
            self.Nfilt = len(radius)

        self.weight = nn.Parameter(
            torch.Tensor(self.Nfilt, output_channels,
                         input_channels))  # out=xA^T+b
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.Nfilt, output_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        std = 2.0 * math.sqrt(2.0 /
                              (self.input_channels + self.output_channels))
        a = math.sqrt(
            3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-a, a)
        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(self.input_channels)
            init.uniform_(self.bias, -bound, bound)

    def map2filtId(self, xyz):
        ''' 
        xyz: (b*n, 3) torch.tensor
        '''
        # radius2=[i*i for i in self.radius]
        n_points = xyz.size(0)
        xyz2filt_lists = [[] for i in range(self.Nfilt)]
        r_norm = torch.norm(xyz, dim=1).data  #(b*n,3)--->(b*n,)
        for i in range(n_points):
            for idx, r in enumerate(self.radius):
                if (r_norm[i] < r):
                    xyz2filt_lists[idx].append(i)
                    break

        indxlen = [0]
        for x in xyz2filt_lists:
            indxlen.append(indxlen[-1] + len(x))
        indx = [y for x in xyz2filt_lists for y in x]
        assert (len(indx) == n_points)
        reindx = [(val, idx) for idx, val in enumerate(indx)]
        reindx.sort()  #key=lambda x : x[0]
        reindx = [x[1] for x in reindx]

        return indx, reindx, indxlen

    @weak_script_method
    def forward(self, feat, xyz):
        # return F.linear(input, self.weight, self.bias)

        # if input.dim() == 2 and bias is not None:
        #     # fused op is marginally faster
        #     ret = torch.addmm(torch.jit._unwrap_optional(bias), input, weight.t())
        # else:
        #     output = input.matmul(weight.t())
        #     if bias is not None:
        #         output += torch.jit._unwrap_optional(bias)
        #     ret = output
        # return ret
        b, n, c = feat.size()
        feat = feat.contiguous().view(b * n, c)  #(b,n, c)--->(b*n, c)
        xyz = xyz.contiguous().view(b * n, -1)  #(b,n, 3)--->(b*n, 3)

        indx, reindx, indxlen = map2f(
            xyz, torch.as_tensor(self.radius, device=torch.device(
                'cuda')))  #[[0,2],[1,..],[3,..]]#--->[0,1,0,2,2,..,0]  N
        indxlen = indxlen.data
        indx = indx.long()
        reindx = reindx.long()
        # feat=feat.index_select(0,torch.as_tensor(indx,device=torch.device('cuda')))
        feat = feat.index_select(0, indx)
        new_feat = []
        for ballId in range(len(self.radius)):
            #pick sub-feat where xyz2filt=id
            if (indxlen[ballId] == indxlen[ballId + 1]):
                continue
            sub_feat = feat[indxlen[ballId]:indxlen[ballId + 1]]
            # print(sub_feat.size())
            # print(self.weight[ballId].size())
            # print(self.bias[ballId].size())
            # print('apply {}-th filter.'.format(ballId))
            new_feat.append(
                F.linear(sub_feat, self.weight[ballId], self.bias[ballId]))
            # new-feat[some pos]=new-sub-feat
        #---->new-feat
        new_feat = torch.cat(new_feat, dim=0)
        #convert to original order
        # new_feat=new_feat.index_select(0,torch.as_tensor(reindx,device=torch.device('cuda'))).view(b,n,-1)
        new_feat = new_feat.index_select(0, reindx).view(b, n, -1)

        return new_feat

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, ball_radius={}, Nfilt={}'.format(
            self.input_channels, self.output_channels, self.bias is not None,
            self.radius, self.Nfilt)


class SphericalGeoconv(torch.nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 bypass_num_outputs,
                 radius,
                 decay_radius,
                 bn=True,
                 bn_decay=None,
                 activation_fn="relu",
                 delta=False):
        ''' SphericalGeoCNN Geo-Conv
        Input:
            feat: (batch_size, num_point, input_channels) TF tensor
            points: (batch_size, num_point, 3) TF tensor
            num_outputs: the count of output channels
            bypass_num_outputs: the count of output channels of bypass
            radius: the inner radius of local ball of each point.
            decay radius: the outer radius of local ball of each point
        '''
        super(SphericalGeoconv, self).__init__()
        self.radius = radius
        self.decay_radius = decay_radius
        self.delta = delta
        # self.perceptron_feat = Perceptron(input_channels, output_channels)
        # self.perceptron_feat = nn.Linear(input_channels, output_channels)
        # self.perceptron_feat = SphericalLinear(input_channels, output_channels, radius=[0.3,0.6,1.2], bias=True)
        self.perceptron_feat = SphericalLinear(input_channels,
                                               output_channels,
                                               radius=[0.5, 1.2],
                                               bias=True)
        self.perceptron_bypass = Fc(input_channels, [bypass_num_outputs * 6],
                                    bn=False,
                                    activation_fn=None)
        # ag, _ = aggregate(mutual, xyz, radius, decay_radius, delta)  #(b,n,bypass*6)--->(b,n,bypass)
        self.bn1 = nn.Sequential(
            # nn.BatchNorm1d(bypass_num_outputs),
            nn.GroupNorm(8, bypass_num_outputs),
            nn.ReLU(True))
        # self.perceptron_ag = Perceptron(bypass_num_outputs, output_channels)
        self.perceptron_ag = nn.Linear(bypass_num_outputs, output_channels)
        self.bn2 = nn.Sequential(
            # nn.BatchNorm1d(output_channels),
            nn.GroupNorm(8, output_channels),
            nn.ReLU(True))

    def forward(self, feat, xyz):
        # self_feat = self.perceptron_feat(feat)
        self_feat = self.perceptron_feat(feat, xyz)  #(B,N,12)--->(B,N,64)
        mutual = self.perceptron_bypass(feat)  #(B,N,12)--->(B,N,32*6)
        ag, _ = aggregate(mutual, xyz, self.radius, self.decay_radius,
                          self.delta)  #(B,N,192)--->(B,N,32)
        ag = ag.transpose(1, 2)  #(B,N,32)--->(B,32,N)
        ag = self.bn1(ag)
        ag = ag.transpose(1, 2)  #(B,32,N)--->(B,N,32)
        ag = self.perceptron_ag(ag)  #(b,n,32)--->(b,n,64)
        outputs = self_feat + ag
        outputs = outputs.transpose(1, 2)
        outputs = self.bn2(outputs)
        outputs = outputs.transpose(1, 2)
        # print('outputs {}'.format(outputs.size()))
        return outputs  #(b,n,64)
