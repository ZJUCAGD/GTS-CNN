import os, sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from ops.layers import * (import LoaclGeometricStructure, batch_knn, graph_max_pooling)
from models.kcnet import GeoNet, parall_GeoNet, AdaptiveGeoNet, KCNetClassify #, AdaptiveKCNetClassify, KCNetSegment, AdaptiveKCNetSegment
from models.kcnet import knnPISphericalGeoNet, knnPD1SphericalGeoNet, knnSphericalGeoNet,PD1Net,PINet
from data.modelnet import ModelNetDataset, AdaptiveModelNetDataset, MeshModelNetDataset
from data.modelnet import MeshPD1ModelNetDataset, MeshPIModelNetDataset #, PD1ModelNetDataset, PIModelNetDataset
from data.shapenet import AdaptiveShapeNetDataset
from tensorboardX import SummaryWriter
from torch.autograd import gradcheck
from utils.misc import Netpara
gpu_0=torch.device('cuda:0')

# class Aggregate(Function):
#     '''
#     inputs:
#         feature: batch_size * num_points * num_channels     float32
#         xyz: batch_size * num_points * 3                    float32
#         radius:                                             float32
#         decay_radius:                                       float32
#         delta                                               int
#     returns:
#         output feature: batch_size * num_points * num_channels  float32
#         norm feature: batch_size * num_points
#     '''

#     @staticmethod
#     def forward(ctx, feat, xyz, radius, decay_radius, delta=0):
#         outputs, norm_buffer = extension_.aggregate_wrapper(feat, xyz, radius, decay_radius, delta)
#         ctx.save_for_backward(feat, xyz)
#         ctx.radius = radius
#         ctx.decay_radius = decay_radius
#         ctx.delta = delta
#         return outputs, norm_buffer

#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         feat, xyz = ctx.saved_tensors
#         grad_feat = extension_.aggregate_grad_wrapper(grad_outputs[0].contiguous(), 
#                                 feat, xyz, ctx.radius, ctx.decay_radius, ctx.delta)
#         return grad_feat, None, None, None, None

# aggregate = Aggregate.apply

# gradcheck takes a tuple of tensors as input, check if your gradient
# evaluated with these tensors are close enough to numerical
# approximations and returns True if they all verify this condition.

# input_agg = (torch.randn((1,10,18),dtype=torch.float, device= gpu_0,requires_grad=True), 
#       torch.randn((1,10,3),dtype=torch.float, device=gpu_0, requires_grad=False),
#       torch.tensor(0.05,device=gpu_0),
#       torch.tensor(0.15,device=gpu_0),
#       torch.tensor(0,device=gpu_0)
#       )
# test = gradcheck(aggregate, input_agg, eps=1e-4, atol=1e-4)
# print(test)


# class GraphMaxPooling(Function):
#     """
#     :param features: (B, C, N)
#            knn_graph: (B, N, npoints)
#     :return: outputs: (B, C, N)
#     """
#     @staticmethod
#     def forward(ctx, features, knn_graph):
#         outputs, idxs = extension_.graph_max_pooling_wrapper(features, knn_graph)
#         ctx.save_for_backward(idxs)
#         return outputs

#     @staticmethod
#     def backward(ctx, grad_outputs):
#         idxs = ctx.saved_tensors[0]
#         grad_inputs = extension_.graph_max_pooling_grad_wrapper(grad_outputs.contiguous(), idxs)
#         return grad_inputs, None


# graph_max_pooling = GraphMaxPooling.apply

# input_graph_max = (torch.randn((1,3,10),dtype=torch.float, device= gpu_0,requires_grad=True), 
#       torch.randint(0,10,(1,10,4), dtype=torch.int, device=gpu_0, requires_grad=False)
#       )
# print(input_graph_max)
# test_2 = gradcheck(graph_max_pooling, input_graph_max, eps=1e-4, atol=1e-4)
# print(test_2)


# x=torch.rand(256,2,dtype=torch.double,requires_grad=True)
# y=torch.randint(0,10,(256,),requires_grad=False)
# custom_op = nn.Linear(2,10).double()
# test_3 = torch.autograd.gradcheck(custom_op,(x,))
# print(test_3)


#######---------ModelNet 40------------######
# # origin
# trainset = AdaptiveModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy', train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
# testset = AdaptiveModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)

# # # mesh
# trainset = MeshModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM.npy', train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = MeshModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM.npy', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
# net = AdaptiveGeoNet(3,class_nums=trainset.class_nums)

# net = parall_GeoNet(3,class_nums=trainset.class_nums)
#net = knnSphericalGeoNet(3,class_nums=trainset.class_nums)

# # mesh+pd1
trainset = MeshPIModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM.npy', train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testset = MeshPIModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM.npy', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
net = knnPISphericalGeoNet(3,class_nums=trainset.class_nums)

# writer=SummaryWriter()
writer=None
for epcho in range(1, 101): #400
    net.fit(trainloader, epcho, writer)
    if(epcho%10==0):
	    net.score(testloader)
# net.score(testloader)
if writer is not None:
	writer.close()
