#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""

import copy
import math
import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
from torch.autograd import Variable

import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.kcnet_utils import debugPrint
import models.ecc_model.ecc as ecc
import torch.nn.init as init

from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,
                                global_mean_pool)

# global writer
global_step = 0


def create_fnet(widths, nfeat, nfeato, orthoinit, llbias):
    """ Creates feature-generating network, a multi-layer perceptron.
    Parameters:
    widths: list of widths of hidden layers
    nfeat, nfeato: # input and output channels of the convolution
    orthoinit: whether to use orthogonal weight initialization
    llbias: whether to use bias in the last layer
    """
    fnet_modules = []
    for k in range(len(widths) - 1):
        fnet_modules.append(nn.Linear(widths[k], widths[k + 1]))
        if orthoinit: init.orthogonal_(fnet_modules[-1].weight, gain=init.calculate_gain('relu'))
        fnet_modules.append(nn.ReLU(True))
    fnet_modules.append(nn.Linear(widths[-1], nfeat * nfeato, bias=llbias))
    if orthoinit: init.orthogonal_(fnet_modules[-1].weight)
    return nn.Sequential(*fnet_modules)


class ECC(nn.Module):
    def __init__(self, config, nfeat, fnet_widths, fnet_orthoinit=True, fnet_llbias=True, edge_mem_limit=1e20, output_channels=40, device_id=3):
        super(ECC, self).__init__()
        self.device_id = device_id


        self.gconvs = []
        self.gpools = []
        self.pyramid_conf = []

        model_config = 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40'


        self.pyramid_conf.append((1.0, 2.0))


        nn1 = create_fnet(fnet_widths, 1, 24, fnet_orthoinit, fnet_llbias)
        self.conv1 = NNConv(1, 24, nn1, aggr='mean')


        nn2 = create_fnet(fnet_widths, 24, 48, fnet_orthoinit, fnet_llbias)
        self.conv2 = NNConv(24, 48, nn2, aggr='mean')

        nn3 = create_fnet(fnet_widths, 48, 48, fnet_orthoinit, fnet_llbias)
        self.conv3 = NNConv(48, 48, nn3, aggr='mean')

        nn4 = create_fnet(fnet_widths, 48, 48, fnet_orthoinit, fnet_llbias)
        self.conv4 = NNConv(48, 48, nn4, aggr='mean')

        nn5 = create_fnet(fnet_widths, 48, 96, fnet_orthoinit, fnet_llbias)
        self.conv5 = NNConv(48, 96, nn5, aggr='mean')

        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 40)









        for d, conf in enumerate(config.split(',')):
            conf = conf.strip().split('_')

            if conf[0] == 'f':  # Fully connected layer; args: output_feats
                self.add_module(str(d), nn.Linear(nfeat, int(conf[1])))
                nfeat = int(conf[1])
            elif conf[0] == 'b':  # Batch norm;            args: not_affine
                self.add_module(str(d), nn.BatchNorm1d(nfeat, eps=1e-5, affine=len(conf) == 1))
            elif conf[0] == 'r':  # ReLU;
                self.add_module(str(d), nn.ReLU(True))
            elif conf[0] == 'd':  # Dropout;                args: dropout_prob
                self.add_module(str(d), nn.Dropout(p=float(conf[1]), inplace=False))

            elif conf[0] == 'm' or conf[0] == 'a':  # Max and Avg poolong;    args: output_resolution, output_radius
                res, rad = float(conf[1]), float(conf[2])
                assert self.pyramid_conf[-1][0] < res, "Pooling should coarsen resolution."
                self.pyramid_conf.append((res, rad))

                gpool = ecc.GraphMaxPoolModule() if conf[0] == 'm' else ecc.GraphAvgPoolModule()
                self.gpools.append(gpool)
                self.add_module(str(d), gpool)

            elif conf[0] == 'i':  # Initial graph parameters;   args: initial_resolution, initial_radius
                res, rad = float(conf[1]), float(conf[2])
                assert len(self.pyramid_conf) == 0 or self.pyramid_conf[-1][
                    0] == res, "Graph cannot be coarsened directly"
                self.pyramid_conf.append((res, rad))

            elif conf[0] == 'c':  # Graph convolution;  args: output_feats
                nfeato = int(conf[1])
                assert len(self.pyramid_conf) > 0, "Convolution needs defined graph"

                fnet = create_fnet(fnet_widths, nfeat, nfeato, fnet_orthoinit, fnet_llbias)

                gconv = ecc.GraphConvModule(nfeat, nfeato, fnet, edge_mem_limit=edge_mem_limit)
                self.gconvs.append((gconv, len(self.pyramid_conf) - 1))
                self.add_module(str(d), gconv)
                nfeat = nfeato

            else:
                raise NotImplementedError('Unknown module: ' + conf[0])


        ##zhu
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def set_info(self, gc_infos, gp_infos, cuda):
        """ Provides pooling and convolution modules with graph structure information for the current batch.
        """
        gc_infos = gc_infos if isinstance(gc_infos,(list,tuple)) else [gc_infos]
        gp_infos = gp_infos if isinstance(gp_infos,(list,tuple)) else [gp_infos]

        for gc,i in self.gconvs:
            if cuda: gc_infos[i].cuda()
            gc.set_info(gc_infos[i])
        for i,gp in enumerate(self.gpools):
            if cuda: gp_infos[i].cuda()
            gp.set_info(gp_infos[i])

    def forward(self,F, graphs_info, cluster):
        """
        :param F: tensor shape (num_point, 1)
        :param edge_attr: list
        :param edge_index: tensor shape (num_edge, c_in)
        :param graphs_info: list of length 4, each is layer nnconv
        :param cluster: list
        :param batch:
        :return:
        """
        # graphs_info.edge_index
        # graphs_info.point2batch
        #
        edge_index = graphs_info[0].edge_index
        edge_attr = graphs_info[0].edge_index
        # F=torch.ones()
        F = self.conv1(F, edge_index, edge_attr)

        x, batch = max_pool_x(cluster[0], F, batch)





        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=transform)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_mean_pool(x, batch)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)




        for module in list(self._modules.values())[:-1]:
            input = module(input)
        return input



    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (F, targets, graphs_info, poolmaps_info) in enumerate(dataloader):  # zhuyijie
            assert len(graphs_info)==4
            # idxn, idxe, degs, degs_gpu, edgefeats = graphs_info[i].get_buffers()
            # self.set_info(graphs_info, poolmaps_info, 1)
            graphs_info, poolmaps_info = self.info_preprocess(graphs_info, poolmaps_info)
            F = F.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self.forward(F, graphs_info, poolmaps_info)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:  # batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8', batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  # zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def info_preprocess(self,graphs_info, poolmaps_info):
        # pass
        assert len(graphs_info) == 4
        for graph_info in graphs_info:
            # idxn, idxe, degs, degs_gpu, edgefeats = graphs_info.get_buffers()
            pass
        # return graphs_info, poolmaps_info
        return edge_index, point2batch,...