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
#from utils.kcnet_utils import debugPrint
import models.ecc_model.ecc as ecc
import torch.nn.init as init
import torchnet as tnt

# global writer
global_step = 0

def cloud_edge_feats(edgeattrs):
    """ Defines edge features for `GraphConvInfo` in the case of point clouds. Assembles edge feature tensor given point offsets as edge attributes.
    """

    columns = []
    offsets = np.asarray(edgeattrs['offset'])

    # todo: possible discretization, round to multiples of min(offsets[offsets>0]) ? Or k-means (slow?)?


    columns.append(offsets)


    p1 = np.linalg.norm(offsets, axis=1)
    p2 = np.arctan2(offsets[:, 1], offsets[:, 0])
    p3 = np.arccos(offsets[:, 2] / (p1 + 1e-6))
    columns.extend([p1[:, np.newaxis], p2[:, np.newaxis], p3[:, np.newaxis]])

    edgefeats = np.concatenate(columns, axis=1).astype(np.float32)


    edgefeats_clust, indices = ecc.unique_rows(edgefeats)
    return torch.from_numpy(edgefeats_clust), torch.from_numpy(indices)
edge_feat_func = cloud_edge_feats


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
        #model_config = 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40'
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

    def forward(self, input):
        #print(list(self._modules.values()))
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


        # batch_size = 3
        #
        # for batch_idx in range(len(trainset)//batch_size):
        #
        #     for k in range(batch_size):
        #         input = [trainset[batch_idx * batch_size + k] for k in range(batch_size)]
        #         inputs, targets, GIs, PIs = ecc.graph_info_collate_classification(input, edge_feat_func)
        #
        #     self.set_info(GIs, PIs, 1)
        #
        #     inputs, targets = inputs.cuda(), targets.cuda()
        #     inputs, targets = Variable(inputs), Variable(targets)
        #
        #     #inputs = inputs.cuda(self.device_id)
        #     #targets = targets.cuda(self.device_id)
        #     self.optimizer.zero_grad()
        #
        #     outputs = self.forward(inputs)
        #     losses = self.loss(outputs, targets)
        #     losses.backward()
        #     self.optimizer.step()
        #
        #     batch_loss += losses.item()
        #     epoch_loss += losses.item()
        #     batch_nums += 1
        #     if (batch_idx + 1) % 8 == 0:  # batch_size=16    16*8=128 samples
        #         print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
        #         global_step += 1
        #         # print('global_step={}'.format(global_step))
        #         if writer is not None:
        #             writer.add_scalar('scalar/batch_loss_every8', batch_loss / 8, global_step)
        #         batch_loss = 0.
        #
        # print('-----------epoch %d end train-----------' % epoch)
        # print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))
        #
        # return epoch_loss / batch_nums

        for batch_idx, (inputs, targets, GIs, PIs) in enumerate(dataloader):  # zhuyijie

            self.set_info(GIs, PIs, 1)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            #inputs = inputs.cuda(self.device_id)
            #targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self.forward(inputs)
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

        #with torch.no_grad():
        # for batch_idx, (inputs, targets) in enumerate(dataloader):


        for batch_idx, (inputs, targets, GIs, PIs) in enumerate(dataloader):  # zhuyijie
            # inputs = inputs.cuda(self.device_id)
            # targets = targets.cuda(self.device_id)
            self.set_info(GIs, PIs, 1)

            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)

            outputs = self.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets.data).sum().item()
            print(batch_idx)

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
