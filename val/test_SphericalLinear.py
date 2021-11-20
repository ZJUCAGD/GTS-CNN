import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ops.layers import SphericalLinear, SphericalGeoconv, map2f
# from ops.layers import LoaclGeometricStructure, batch_knn, graph_max_pooling
from utils.misc import Netpara

gpu_0 = torch.device('cuda:0')

# def map2filtId(xyz,radius=[0.3,0.6,1],Nfilt=3):
#     '''
#     xyz: (b*n, 3) torch.tensor
#     '''
#     radius2=[i*i for i in radius]
#     n_points=xyz.size(0)
#     xyz2filt_lists=[[] for _ in range(Nfilt)]
#     for i in range(n_points):
#         v=xyz[i]
#         for idx,r2 in enumerate(radius2):
#             if(np.dot(v,v)<r2):
#                 xyz2filt_lists[idx].append(i)
#                 break

#     indxlen=[0]
#     for x in xyz2filt_lists:
#         indxlen.append(indxlen[-1]+len(x))
#     indx = [y for x in xyz2filt_lists for y in x]
#     reindx = [(val,idx) for idx,val in enumerate(indx)]
#     reindx.sort()  #key=lambda x : x[0]
#     reindx=[x[1] for x in reindx]

#     return indx, reindx, indxlen
# class Aggregate(Function):

if __name__ == '__main__':
    input_channels = 3
    output_channels = 6
    # sl = SphericalLinear(input_channels, output_channels, radius=[0.3,0.6,1], bias=True)
    # print(sl)
    # Netpara(sl)
    # sgeo1 = SphericalGeoconv(64, 128, 64, 0.05, 0.15, bn=True)
    # print(sgeo1)
    # Netpara(sgeo1)
    # para = nn.Parameter(torch.Tensor(output_channels))
    # para = torch.randn(3, 4)
    # xyz2filt=np.array([0, 1, 2, 0, 1, 2])
    # indices = torch.tensor([0, 2])
    # feat=torch.rand(2,4,3)			# feat (2,4,3)
    # xyz=(feat-0.5)*2/np.sqrt(3)     # xyz (2,4,3) range in [1/sqrt(3)]
    # assert(feat.size(0)==2)
    # out=sl(feat,xyz)
    feat = torch.rand(4, 3)  # feat (4,3)
    xyz = (feat - 0.5) * 2 / np.sqrt(3)  # xyz (4,3) range in [1/sqrt(3)]
    xyz = xyz.cuda()
    radius = torch.as_tensor([0.3, 0.6, 1.2], device=gpu_0)
    index, reindex, indexlen = map2f(xyz, radius)
