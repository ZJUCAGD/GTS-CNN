import argparse
import math
import h5py
import numpy as np
import socket
import importlib
import matplotlib.pyplot as plt
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
import provider
import math
import random
import models.pointCNN.utils.data_utils as data_utils
import time

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


from models.pointCNN.utils.model import RandPointCNN
from models.pointCNN.utils.util_funcs import knn_indices_func_gpu, knn_indices_func_cpu
from models.pointCNN.utils.util_layers import Dense

global_step = 0

NUM_CLASS = 40

AbbPointCNN = lambda a, b, c, d, e: RandPointCNN(a, b, 3, c, d, e, knn_indices_func_gpu) # e number of sample points


class Classifier(nn.Module):

    def __init__(self, device_id=3):
        super(Classifier, self).__init__()

        self.device_id = device_id

        # self.pcnn1 = AbbPointCNN(3, 32, 8, 1, -1)
        # self.pcnn2 = nn.Sequential(
        #     AbbPointCNN(32, 64, 8, 2, -1),
        #     AbbPointCNN(64, 96, 8, 4, -1),
        #     AbbPointCNN(96, 128, 12, 4, 120),
        #     AbbPointCNN(128, 160, 12, 6, 120)
        # )
        #
        # self.fcn = nn.Sequential(
        #     Dense(160, 128),
        #     Dense(128, 64, drop_rate=0.5),
        #     Dense(64, NUM_CLASS, with_bn=False, activation=None)
        # )


        # original paper
        # self.pcnn1 = AbbPointCNN(3, 48, 8, 1, -1)
        # self.pcnn2 = nn.Sequential(
        #     AbbPointCNN(48, 96, 12, 2, 384),
        #     AbbPointCNN(96, 192, 16, 2, 128),
        #     AbbPointCNN(192, 384, 16, 3, 128)
        # )
        #
        # self.fcn = nn.Sequential(
        #     Dense(384, 256),
        #     Dense(256, 128, drop_rate=0.5),
        #     Dense(128, NUM_CLASS, with_bn=False, activation=None)
        # )

        # add parameters to almost 1.5M
        self.pcnn1 = AbbPointCNN(3, 48, 8, 1, -1)
        self.pcnn2 = nn.Sequential(
            AbbPointCNN(48, 96, 12, 2, 512),
            AbbPointCNN(96, 192, 16, 2, 384),
            AbbPointCNN(192, 384, 16, 3, 128),
            AbbPointCNN(384, 512, 16, 3, 128)
        )

        self.fcn = nn.Sequential(
            Dense(512, 384),
            Dense(384, 256),
            Dense(256, 128, drop_rate=0.5),
            Dense(128, NUM_CLASS, with_bn=False, activation=None)
        )



        ##zhu
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, x):
        x = self.pcnn1(x)
        if False:
            print("Making graph...")
            k = make_dot(x[1])

            print("Viewing...")
            k.view()
            print("DONE")

            assert False
        x = self.pcnn2(x)[1]  # grab features

        logits = self.fcn(x)
        logits_mean = torch.mean(logits, dim=1)
        return logits_mean


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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  # zhuyijie
            inputs = inputs.permute(0, 2, 1)
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self((inputs, inputs))
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
                inputs = inputs.permute(0, 2, 1)
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                outputs = self((inputs, inputs))
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