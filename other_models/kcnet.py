import os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ops.layers import LoaclGeometricStructure, batch_knn, graph_max_pooling, group_points
from ops.layers import Fc, Perceptron, Geoconv, SphericalGeoconv,furthest_point_sample,three_nn,three_interpolate
from utils.misc import debugPrint
import json
import scipy.io as scio

# global writer
global_step=0

def gather_nd(input_tensor, indices):
    """
    input_tensor: (b,n,c), float32
    indices: (b,m), int

    """
    batch_size = input_tensor.size(0)
    # indices=indices.long()
    return torch.stack([torch.index_select(input_tensor[k],0,indices[k]) for k in range(batch_size)]) # keep dim as xyz


class KCNetClassify(nn.Module):

    def __init__(self, class_nums, device_id=0, initial_weights=True):
        super(KCNetClassify, self).__init__()

        self.class_nums = class_nums
        self.knn_points = 16
        self.device_id = device_id

        if initial_weights:
            self.initialize_weights()

        self.kc = LoaclGeometricStructure(32, 16, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(32 + 3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(192, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.classify = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points):
        knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        # knn_graph = adptive_knn(points,self.knn_points+1) # knn_points=16
        x = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x = torch.cat([points, x], dim=1)
        x = self.mlp1(x)
        y = graph_max_pooling(x, knn_graph)
        x = self.mlp2(x)
        x = torch.cat([x, y], dim=1)
        x = self.mlp3(x)
        x = F.max_pool1d(x, x.size(2), stride=1).squeeze(2)
        x = self.classify(x)
        return x

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every4',batch_loss / 4, global_step)
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                outputs = self(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total


class AdaptiveKCNetClassify(nn.Module):

    def __init__(self, class_nums, device_id=0, initial_weights=True):
        super(AdaptiveKCNetClassify, self).__init__()

        self.class_nums = class_nums
        self.knn_points = 16
        self.device_id = device_id

        if initial_weights:
            self.initialize_weights()

        self.kc = LoaclGeometricStructure(32, 16, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(32 + 3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(192, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.classify = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points, knn_graph):
        # knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        # knn_graph = adptive_knn(points,self.knn_points+1) # knn_points=16
        x = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x = torch.cat([points, x], dim=1)
        x = self.mlp1(x)
        y = graph_max_pooling(x, knn_graph)
        x = self.mlp2(x)
        x = torch.cat([x, y], dim=1)
        x = self.mlp3(x)
        x = F.max_pool1d(x, x.size(2), stride=1).squeeze(2)
        x = self.classify(x)
        return x

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

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer):
        # global writer
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, graphs, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            # graphs = graphs.cuda(self.device_id)   #zhuyijie
            graphs = graphs.to(torch.device('cuda:0'),dtype=torch.int)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs, graphs)   #zhuyijie
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every4',batch_loss / 4, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, graphs, targets) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                # graphs = graphs.cuda(self.device_id)   #zhuyijie
                graphs = graphs.to(torch.device('cuda:0'),dtype=torch.int)
                targets = targets.cuda(self.device_id)

                outputs = self(inputs, graphs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total


class KCNetSegment(nn.Module):

    def __init__(self, class_nums, category_nums, device_id=0, initial_weights=True):
        super(KCNetSegment, self).__init__()
        print('use KCNetSegment!!!')
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 18
        self.device_id = device_id
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        self.kc = LoaclGeometricStructure(16, 18, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3 + 16, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.mlp6 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mlp7 = nn.Sequential(
            nn.Conv1d(3 + 16 + 64 + 64 + 128 + 128 + 512 + 1024 + category_nums, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(256, class_nums, 1)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points, labels):
        knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        x1 = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x1 = torch.cat([points, x1], dim=1)
        x2 = self.mlp1(x1)
        x3 = self.mlp2(x2)
        x4 = self.mlp3(x3)
        x5 = graph_max_pooling(x4, knn_graph)
        x5 = self.mlp4(x5)
        x6 = self.mlp5(x5)
        x7 = graph_max_pooling(x6, knn_graph)
        x7 = self.mlp6(x7)
        x7 = F.max_pool1d(x7, x7.size(2), stride=1)
        x7 = x7.repeat([1, 1, knn_graph.size(1)])

        index = labels.unsqueeze(1).repeat([1, knn_graph.size(1)]).unsqueeze(1)
        one_hot = torch.zeros([knn_graph.size(0), self.category_nums, knn_graph.size(1)])
        one_hot = one_hot.cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, one_hot], dim=1)
        x = self.mlp7(x)

        return x

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

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer): #zhuyijie
        # global writer
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            labels = labels.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs, labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['ins'],ret['cls']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))

        return correct / total

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
            
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret


class AdaptiveKCNetSegment(nn.Module):

    def __init__(self, class_nums, category_nums, device_id=0, initial_weights=True):
        super(AdaptiveKCNetSegment, self).__init__()

        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 18
        self.device_id = device_id

        self.kc = LoaclGeometricStructure(16, 18, 0.005)
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3 + 16, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp2 = nn.Sequential(
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(True)
        )
        self.mlp3 = nn.Sequential(
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp4 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(True)
        )
        self.mlp5 = nn.Sequential(
            nn.Conv1d(128, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.mlp6 = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.mlp7 = nn.Sequential(
            nn.Conv1d(3 + 16 + 64 + 64 + 128 + 128 + 512 + 1024 + category_nums, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv1d(256, class_nums, 1)
        )

        if initial_weights:
            self.initialize_weights()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, points, knn_graph, labels):
        # knn_graph, _ = batch_knn(points, points.clone(), self.knn_points + 1)
        x1 = self.kc(points, knn_graph[:, :, 1:].contiguous())
        x1 = torch.cat([points, x1], dim=1)
        x2 = self.mlp1(x1)
        x3 = self.mlp2(x2)
        x4 = self.mlp3(x3)
        x5 = graph_max_pooling(x4, knn_graph)
        x5 = self.mlp4(x5)
        x6 = self.mlp5(x5)
        x7 = graph_max_pooling(x6, knn_graph)
        x7 = self.mlp6(x7)
        x7 = F.max_pool1d(x7, x7.size(2), stride=1)
        x7 = x7.repeat([1, 1, knn_graph.size(1)])

        index = labels.unsqueeze(1).repeat([1, knn_graph.size(1)]).unsqueeze(1)
        one_hot = torch.zeros([knn_graph.size(0), self.category_nums, knn_graph.size(1)])
        one_hot = one_hot.cuda(self.device_id)
        one_hot = one_hot.scatter_(1, index, 1)

        x = torch.cat([x1, x2, x3, x4, x5, x6, x7, one_hot], dim=1)
        x = self.mlp7(x)

        return x

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

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer): #zhuyijie
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, graphs, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id)
            graphs = graphs.to(torch.device('cuda:0'),dtype=torch.int)  ###zhuyijie
            targets = targets.cuda(self.device_id)
            labels = labels.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs, graphs, labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 4 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 4))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every4',batch_loss / 4, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            for batch_idx, (inputs, graphs, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                graphs = graphs.to(torch.device('cuda:0'),dtype=torch.int)  ###zhuyijie
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs, graphs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total


# 2019.10.30
class GeoNetSegment(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(GeoNetSegment,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 18
        self.device_id = device_id
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()

        # self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn='BN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        # self.FC3 = Fc(128,[256], bn='BN', activation_fn='relu')
        # self.FC4 = Fc(768,[2048], bn='BN', activation_fn='relu')

        self.geo1 = Geoconv(3, 64, 32, 0.05, 0.15, bn=True)
        self.geo2 = Geoconv(64, 64, 32, 0.1, 0.2, bn=True)
        self.geo3 = Geoconv(64, 64, 32, 0.15, 0.3, bn=True)

        self.geo4 = Geoconv(64, 128, 64, 0.2, 0.4, bn=True)
        self.geo5 = Geoconv(128, 1024, 128, 0.2, 0.4, bn=True)

        self.geo6 = Geoconv(1024+category_nums+64, 512, 32, 0.15, 0.3, bn=True)
        self.geo7 = Geoconv(512, 256, 32, 0.1, 0.2, bn=True)
        self.geo8 = Geoconv(256, 128, 32, 0.05, 0.1, bn=True)
        self.geo9 = Geoconv(128, 128, 32, 0.05, 0.1, bn=True)

        self.classify = nn.Sequential(
            # nn.Conv1d(3 + 16 + 64 + 64 + 128 + 128 + 512 + 1024 + category_nums, 512, 1),
            # nn.BatchNorm1d(512),
            # nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, class_nums, 1)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud,labels):  #B,C,N
        # b,n,npoints,c=x.size()
        point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud
        b,n,c=xyz.size()

        y=self.geo1(point_cloud,xyz)     #---->(b,n,64)
        y=self.geo2(y,xyz)     #---->(b,n,64)
        y=self.geo3(y,xyz)     #---->(b,n,64)
        point_feat = y

        y=self.geo4(y,xyz)     #---->(b,n,128)
        y=self.geo5(y,xyz)     #---->(b,n,1024)
        y=torch.max(y, 1, keepdim=True)[0] # (b,1,1024)

        # index = labels.unsqueeze(1).repeat([1, n]).unsqueeze(1)
        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)
        y=torch.cat([y,one_hot.unsqueeze(1)],dim=2) # -->(b,1,1024+1)
        
        y=torch.cat([point_feat, y.repeat([1,n,1])],dim=2) # -->(b,n,1024+1+64)

        y=self.geo6(y,xyz)     #---->(b,n,512)
        y=self.geo7(y,xyz)     #---->(b,n,256)
        y=self.geo8(y,xyz)     #---->(b,n,128)
        y=self.geo9(y,xyz)     #---->(b,n,128)

        y= self.classify(y.transpose(1,2)) # -->(b,2,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)

            self.optimizer.zero_grad()

            outputs = self(inputs,labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
                
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cup().numpy(),targets.cup().numpy(),shape_ious)
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {}, ins_miou = {}'.format(res['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))

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

    def compute_miou(self,predicted,targets, shape_ious):
        """
        predicted: numpy array, (b,n), int
        targets: numpy array, (b,n), int
        """
        batch_size=targets.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])

        cls_miou = np.mean(shape_ious.values())
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret

class TestGeoNetSegment(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(TestGeoNetSegment,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.name='TestGeoNetSegment'
        print(self.name)
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 18
        self.device_id = device_id
        
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()

        # self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn='BN', activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='BN', activation_fn='relu')
        self.FC4 = Fc(512,[1024], bn='BN', activation_fn='relu')

        self.geo1 = Geoconv(64, 128, 64, 0.1, 0.2, bn=True)
        # self.geo2 = Geoconv(64, 256, 64, 0.1, 0.2, bn=True)
        # self.geo3 = Geoconv(64, 64, 32, 0.15, 0.3, bn=True)

        self.geo3 = Geoconv(256, 512, 64, 0.2, 0.3, bn=True)
        # self.geo5 = Geoconv(128, 1024, 128, 0.2, 0.4, bn=True)

        # self.geo4 = Geoconv(1024+category_nums+64, 512, 64, 0.15, 0.3, bn=True)
        # self.geo5 = Geoconv(512, 256, 32, 0.05, 0.15, bn=True)
        # self.geo8 = Geoconv(256, 128, 32, 0.05, 0.1, bn=True)
        # self.geo9 = Geoconv(128, 128, 32, 0.05, 0.1, bn=True)

        self.classify = nn.Sequential(
            nn.Conv1d(3 + 128 + 512 + 1024 + category_nums, 512, 1,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, class_nums, 1,bias=True)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud,labels):  #B,C,N
        # b,n,npoints,c=x.size()
        point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud
        b,n,c=xyz.size()

        point_cloud=self.FC2(point_cloud)
        y1=self.geo1(point_cloud,xyz)     #---->(b,n,64)
        y2=self.FC3(y1)   #(b,n,256)

        y3=self.geo3(y2,xyz)     #---->(b,n,512)
        y4=self.FC4(y3)  #(B,N,1024)
        
        y=torch.max(y4, 1, keepdim=True)[0] # (b,1,1024)

        # index = labels.unsqueeze(1).repeat([1, n]).unsqueeze(1)
        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)
        y=torch.cat([xyz,y1,y3,y.repeat([1,n,1]), one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) # -->(b,n,3+128+512+1024+16)
        
        # y=torch.cat([point_feat, y.repeat([1,n,1])],dim=2) # -->(b,n,1024+16+64)

        # y=self.geo4(y,xyz)     #---->(b,n,512)
        # y=self.geo5(y,xyz)     #---->(b,n,256)
        # y=self.geo8(y,xyz)     #---->(b,n,128)
        # y=self.geo9(y,xyz)     #---->(b,n,128)

        y= self.classify(y.transpose(1,2)) # -->(b,50,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)

            self.optimizer.zero_grad()

            outputs = self(inputs,labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
                
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # if batch_idx>10:
                #     break
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))
        if is_save:
            with open('./{}_miou.txt'.format(self.name),'a+') as file:
                file.writelines(json.dumps(ret)+'\n')
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

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret


# 11.2
class TestKNNGeoNetSegment(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(TestKNNGeoNetSegment,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.name='TestKNNGeoNetSegment'
        print(self.name)
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 16
        self.device_id = device_id
        self.best_score = 0.0  
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[64], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[128,256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(256,[256,512], bn='GN', activation_fn='relu')

        self.geo1 = Geoconv(64, 128, 32, 0.1, 0.2, bn=True)
        self.geo2 = Geoconv(256, 256, 64, 0.15, 0.3, bn=True)


        self.classify = nn.Sequential(
            nn.Conv1d(3 + 128 + 64 + 128 + 256+ 512 + category_nums, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, class_nums, 1, bias=True)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)
    
    def forward(self, point_cloud,labels):  #B,C,N
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        # debugPrint(knn_graph.size())
        x = group_points(point_cloud, knn_graph[:16].contiguous())  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,128)
        x=torch.max(x,2,keepdim=False)[0] # b,n,128

        xyz=point_cloud.transpose(1,2)
        b,n,c=xyz.size()

        # point_cloud=self.FC2(point_cloud)
        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### end ARPE
        y0=y
        y1=self.geo1(y,xyz)     #---->(b,n,128)
        y2=self.FC3(y1)   #(b,n,256)

        y3=self.geo2(y2,xyz)     #---->(b,n,256)
        y4=self.FC4(y3)  #(B,N,512)
        
        y=torch.max(y4, 1, keepdim=True)[0] # (b,1,512)

        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)
        y=torch.cat([xyz,x,y0,y1,y3,y.repeat([1,n,1]), one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) #(b,n,3+128+64+128+256+512+16)

        y= self.classify(y.transpose(1,2)) # -->(b,50,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)

            self.optimizer.zero_grad()

            outputs = self(inputs,labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
            
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # if batch_idx>10:
                #     break
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))
        if is_save:
            with open('./{}_miou.txt'.format(self.name),'a+') as file:
                file.writelines(json.dumps(ret)+'\n')
            if self.best_score<ret['ins']:
                self.best_score=ret['ins']
                torch.save(self, './model_param/{}_best_weight_1103.ckpt'.format(self.name))
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

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret


# 11.3
class TestKNNPIGeoNetSegment(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(TestKNNPIGeoNetSegment,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.name='TestKNNPIGeoNetSegment'
        print(self.name)
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 16
        self.device_id = device_id
        self.best_score = 0.0  
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[64], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[128,256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(256,[256,512], bn='GN', activation_fn='relu')

        # self.geo1 = Fc(64,[128], bn='BN', activation_fn='relu')
        self.geo1 = Geoconv(64, 128, 32, 0.1, 0.2, bn=True)
        self.geo2 = Geoconv(256, 256, 64, 0.15, 0.3, bn=True)

        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,128,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.classify_pi=Fc(512,[256,128], bn='GN', activation_fn='relu')

        self.classify = nn.Sequential(
            nn.Conv1d(3 + 128 + 64 + 128 + 256+ 512 + 128 + category_nums, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, class_nums, 1, bias=True)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)
    
    def forward(self, point_cloud, pi, labels):  #(B,C,N), (B,50,50)
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        # debugPrint(knn_graph.size())
        x = group_points(point_cloud, knn_graph[:16].contiguous())  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,128)
        x=torch.max(x,2,keepdim=False)[0] # b,n,128

        xyz=point_cloud.transpose(1,2)
        b,n,c=xyz.size()

        # point_cloud=self.FC2(point_cloud)
        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### end ARPE
        y0=y
        y1=self.geo1(y,xyz)     #---->(b,n,128)
        # y1=self.geo1(y)     #---->(b,n,128)
        y2=self.FC3(y1)   #(b,n,256)

        y3=self.geo2(y2,xyz)     #---->(b,n,256)
        # y3=y2
        y4=self.FC4(y3)  #(B,N,512)
        
        y=torch.max(y4, 1, keepdim=True)[0] # (b,1,512)

        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)

        pi=self.pi_conv(pi.unsqueeze(1)).view(b,-1)
        pi=self.classify_pi(pi)  ##--->(b,128)
        
        y=torch.cat([xyz,x,y0,y1,y3,y.repeat([1,n,1]), pi.unsqueeze(1).repeat([1,n,1]),one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) #(b,n,3+128+64+128+256+512+128+16)

        y= self.classify(y.transpose(1,2)) # -->(b,50,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, _, pi, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)
            pi = pi.to(device=self.device_id,dtype=torch.float32)

            self.optimizer.zero_grad()

            outputs = self(inputs,pi,labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
            
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums
    
    def score(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, pi, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)
                pi = pi.to(device=self.device_id,dtype=torch.float32)

                outputs = self(inputs,pi,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # if batch_idx>10:
                #     break
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))
        is_save=False
        if is_save:
            with open('./{}_miou.txt'.format(self.name),'a+') as file:
                file.writelines(json.dumps(ret)+'\n')
            if self.best_score<ret['ins']:
                self.best_score=ret['ins']
                torch.save(self, './model_param/{}_best_weight_1103.ckpt'.format(self.name))
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

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret

# 11.4
class TestKNNPIGeoNetSegment_fps(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(TestKNNPIGeoNetSegment_fps,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.name='TestKNNPIGeoNetSegment_fps'
        print(self.name)
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 16
        self.device_id = device_id
        self.best_score = 0.0  
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[64], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[128,256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(256,[256,512], bn='GN', activation_fn='relu')

        # self.geo1 = Fc(64,[128], bn='BN', activation_fn='relu')
        self.geo1 = Geoconv(64, 128, 32, 0.1, 0.2, bn=True)
        self.geo2 = Geoconv(256, 256, 64, 0.15, 0.3, bn=True)
        
        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,128,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.classify_pi=Fc(512,[256,128], bn='GN', activation_fn='relu')

        self.classify = nn.Sequential(
            nn.Conv1d(3 + 128 + 64 + 128 + 256 + 512 + 128 + category_nums, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, class_nums, 1, bias=True)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)
    
    def forward(self, point_cloud, pi, labels):  #(B,C,N), (B,50,50)
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        # debugPrint(knn_graph.size())
        x = group_points(point_cloud, knn_graph[:16].contiguous())  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,128)
        x=torch.max(x,2,keepdim=False)[0] # b,n,128

        xyz=point_cloud.transpose(1,2).contiguous()
        b,n,c=xyz.size()

        # point_cloud=self.FC2(point_cloud)
        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### end ARPE
        y0=y
        y1=self.geo1(y,xyz)     #---->(b,n,128)

        sample_index = furthest_point_sample(xyz,1024).long()
        sample_xyz = gather_nd(xyz, sample_index)
        sample_y1 = gather_nd(y1, sample_index)
        # debugPrint(sample_y1)

        y2=self.FC3(sample_y1)   #(b,n,256)

        y3=self.geo2(y2,sample_xyz)     #---->(b,n,256)
        # y3=y2
        upsample=True
        if upsample:
            dist, idx = three_nn(xyz, sample_xyz)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_y3 = three_interpolate(y3.transpose(1,2).contiguous(), idx, weight).transpose(1,2) # (b,1024,c)--->(b,2048,c)
            # debugPrint(interpolated_y3.size())
        y4=self.FC4(y3)  #(B,N,512)
        
        y=torch.max(y4, 1, keepdim=True)[0] # (b,1,512)

        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)

        pi=self.pi_conv(pi.unsqueeze(1)).view(b,-1)
        pi=self.classify_pi(pi)  ##--->(b,128)
        
        y=torch.cat([xyz,x,y0,y1,interpolated_y3,y.repeat([1,n,1]), pi.unsqueeze(1).repeat([1,n,1]),one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) #(b,n,3+128+64+128+512+128+16)

        y= self.classify(y.transpose(1,2)) # -->(b,50,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, _, pi, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)
            pi = pi.to(device=self.device_id,dtype=torch.float32)

            self.optimizer.zero_grad()

            outputs = self(inputs,pi,labels)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
            # raise Exception
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums
    
    def score(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, _, pi, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)
                pi = pi.to(device=self.device_id,dtype=torch.float32)

                outputs = self(inputs,pi,labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # if batch_idx>10:
                #     break
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))
        # is_save=False
        if is_save:
            with open('./{}__upsample_miou.txt'.format(self.name),'a+') as file:
                file.writelines(json.dumps(ret)+'\n')
            if self.best_score<ret['ins']:
                self.best_score=ret['ins']
                torch.save(self, './model_param/{}_best_weight_1104_upsample.ckpt'.format(self.name))
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

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret

    def test(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        ret=[]
        with torch.no_grad():
            for batch_idx, (inputs, _, pi, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)
                pi = pi.to(device=self.device_id,dtype=torch.float32)

                outputs = self(inputs,pi,labels)
                _, predicted = torch.max(outputs.data, 1) # (b,2048)
                ret.append((inputs.cpu().numpy(),predicted.cpu().numpy(),labels.cpu().numpy()))
                if batch_idx>10:
                    debugPrint(ret)
                    break
        ret=zip(*ret)
        ret=[np.concatenate(items) for items in ret]
        debugPrint(ret)
        for i in ret:
            print(i.shape)
        if is_save:
            # with open('./{}__upsample_pred.txt'.format(self.name),'a+') as file:
            #     file.writelines(json.dumps(ret)+'\n')
            scio.savemat('./{}__upsample_pred.mat'.format(self.name), {'pos':ret[0],
                        'pred_seg':ret[1],'label':ret[2]})

        return ret

# 11.4
class TestKNNPISphereGeoNetSegment_fps(nn.Module):
    def __init__(self, input_channels, class_nums=50,category_nums=16, device_id=0, initial_weights=True):
        super(TestKNNPISphereGeoNetSegment_fps,self).__init__()
        # self.knn_points = 16 # {8,16,32}
        self.name='TestMeshPISphereGeoNetSegment_fps_2split_1120'
        print(self.name)
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.category_nums = category_nums
        self.knn_points = 16
        self.device_id = device_id
        self.best_score = 0.0  
        self.seg_classes = {
            'Airplane':     [0, 1, 2, 3],
            'Bag':          [4, 5],
            'Cap':          [6, 7],
            'Car':          [8, 9, 10, 11],
            'Chair':        [12, 13, 14, 15],
            'Earphone':     [16, 17, 18],
            'Guitar':       [19, 20, 21],
            'Knife':        [22, 23],
            'Lamp':         [24, 25, 26, 27],
            'Laptop':       [28, 29],
            'Motorbike':    [30, 31, 32, 33, 34, 35],
            'Mug':          [36, 37],
            'Pistol':       [38, 39, 40],
            'Rocket':       [41, 42, 43],
            'Skateboard':   [44, 45, 46],
            'Table':        [47, 48, 49]
        }
        self.seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        if initial_weights:
            self.initialize_weights()
        # self.FC0 = Fc(input_channels,[32],input_dim=3, bn='BN', activation_fn='relu')
        self.FC1 = Fc(input_channels,[32,64,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[64], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[128,256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(256,[256,512], bn='GN', activation_fn='relu')

        # self.geo1 = Fc(64,[128], bn='BN', activation_fn='relu')
        # self.geo1 = Geoconv(64, 128, 32, 0.1, 0.2, bn=True)
        # self.geo2 = Geoconv(256, 256, 64, 0.15, 0.3, bn=True)
        self.geo1 = SphericalGeoconv(64, 128, 32, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 256, 64, 0.15, 0.3, bn=True)

        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,128,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        self.classify_pi=Fc(512,[256,128], bn='GN', activation_fn='relu')

        self.classify = nn.Sequential(
            nn.Conv1d(3 + 128 + 64 + 128 + 256+ 512 + 128 + category_nums, 512, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(512, class_nums, 1, bias=True)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)
    
    def forward(self, point_cloud, pi, labels,knn_graph=None):  #(B,C,N), (B,50,50), long, (B,N,32+1)
        # debugPrint(knn_graph.size())
        if knn_graph is None:
            knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        
        x = group_points(point_cloud, knn_graph[:, :, :16].contiguous())  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,128)
        x=torch.max(x,2,keepdim=False)[0] # b,n,128

        xyz=point_cloud.transpose(1,2).contiguous()
        b,n,c=xyz.size()

        # point_cloud=self.FC2(point_cloud)
        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### end ARPE
        y0=y
        y1=self.geo1(y,xyz)     #---->(b,n,128)

        sample_index = furthest_point_sample(xyz,1024).long()
        sample_xyz = gather_nd(xyz, sample_index)
        sample_y1 = gather_nd(y1, sample_index)
        # debugPrint(sample_y1)

        y2=self.FC3(sample_y1)   #(b,n,256)

        y3=self.geo2(y2,sample_xyz)     #---->(b,n,256)
        # y3=y2
        upsample=True
        if upsample:
            dist, idx = three_nn(xyz, sample_xyz)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_y3 = three_interpolate(y3.transpose(1,2).contiguous(), idx, weight).transpose(1,2) # (b,1024,c)--->(b,2048,c)
            # debugPrint(interpolated_y3.size())
        y4=self.FC4(y3)  #(B,N,512)
        
        y=torch.max(y4, 1, keepdim=True)[0] # (b,1,512)

        one_hot = torch.zeros([b, self.category_nums],device=self.device_id).scatter_(1, labels.view(-1, 1), 1) # (b,c)

        pi=self.pi_conv(pi.unsqueeze(1)).view(b,-1)
        pi=self.classify_pi(pi)  ##--->(b,128)
        
        y=torch.cat([xyz,x,y0,y1,interpolated_y3,y.repeat([1,n,1]), pi.unsqueeze(1).repeat([1,n,1]),one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) #(b,n,3+128+64+256+512+128+16)
        # y=torch.cat([xyz,y0,x,y1,interpolated_y3,y.repeat([1,n,1]), pi.unsqueeze(1).repeat([1,n,1]),one_hot.unsqueeze(1).repeat([1,n,1])],dim=2) #(b,n,3+128+64+256+512+128+16)

        y= self.classify(y.transpose(1,2)) # -->(b,50,n)

        return y #(b,50,2048)

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

        for batch_idx, (inputs, knn_graph, pi, targets, labels) in enumerate(dataloader):
            inputs = inputs.cuda(self.device_id) # (b,3,2048)
            knn_graph = knn_graph.cuda(self.device_id) #(b,n,32+1)
            targets = targets.cuda(self.device_id) #(b,2018)
            labels = labels.cuda(self.device_id) #(b,)
            pi = pi.to(device=self.device_id,dtype=torch.float32)

            self.optimizer.zero_grad()

            outputs = self(inputs,pi,labels,knn_graph)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
            # raise Exception
        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums
    
    def score(self, dataloader, is_save=False):
        self.eval()
        correct = 0.
        total = 0
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        with torch.no_grad():
            for batch_idx, (inputs, knn_graph, pi, targets, labels) in enumerate(dataloader):
                inputs = inputs.cuda(self.device_id)
                knn_graph = knn_graph.cuda(self.device_id) #(b,n,32+1)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)
                pi = pi.to(device=self.device_id,dtype=torch.float32)

                outputs = self(inputs,pi,labels,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()
                self.compute_miou(predicted.cpu().numpy(),targets.cpu().numpy(),shape_ious)
                # if batch_idx>10:
                #     break
                # debugPrint(shape_ious['Airplane'])
        ret=self.get_miou(shape_ious) #{'cls':value,'ins':value}
        print('cls_miou = {:4f}, ins_miou = {:4f}'.format(ret['cls'],ret['ins']))
        print('Accuracy of the network: %.2f %%' % (100.0 * correct / total))
        # is_save=False
        if is_save:
            with open('./{}_upsample_miou_1120.txt'.format(self.name),'a+') as file:
                file.writelines(json.dumps(ret)+'\n')
            if self.best_score<ret['ins']:
                self.best_score=ret['ins']
                torch.save(self, './model_param/{}_best_weight_1120_upsample.ckpt'.format(self.name))
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

    def compute_miou(self,pred_label,true_label, shape_ious):
        """
        pred_label: numpy array, (b,n), int
        true_label: numpy array, (b,n), int
        """
        batch_size=true_label.shape[0]
        for bi in range(batch_size):
            segp = pred_label[bi, :]
            segl = true_label[bi, :]
            cat = self.seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
            for l in self.seg_classes[cat]:
                if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                    iou = 1.0
                else:
                    iou = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                part_ious[l - self.seg_classes[cat][0]] = iou
            shape_ious[cat].append(np.mean(part_ious))

    def get_miou(self,shape_ious):
        all_shape_ious = []
        
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat]) if len(shape_ious[cat])>0 else 0
        cls_miou = np.mean(list(shape_ious.values()))
        ins_miou = np.mean(all_shape_ious)

        ret = dict(shape_ious)
        ret['cls'] = cls_miou
        ret['ins'] = ins_miou
        return ret


class GeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(GeoNet,self).__init__()
        self.knn_points = 32 # {8,16,32}
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn='BN', activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='BN', activation_fn='relu')
        self.FC4 = Fc(768,[2048], bn='BN', activation_fn='relu')

        self.geo1 = Geoconv(64, 128, 64, 0.05, 0.15, bn=True)
        self.geo2 = Geoconv(256, 512, 64, 0.15, 0.3, bn=True)
        self.geo3 = Geoconv(896, 768, 64, 0.3, 0.6, bn=True)
        self.classify = nn.Sequential(
            nn.Linear(2048, 512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(512, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud):  #B,C,N
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph)  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(16,64,16,12))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        # x=x.transpose(2,3).contiguous().view(b*n,c,npoints)
        # x=nn.MaxPool1D(npoints)(x)  #--->(b*n,c,1)
        # x=x.squeeze()
        # x=x.view(b,n,c)
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)
        point_cloud=point_cloud.transpose(1,2)
        # xyz=xyz.transpose(1,2)
        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y= self.classify(y)

        return y

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
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


class AdaptiveGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(AdaptiveGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn=True, activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn=True, activation_fn='relu')
        self.FC3 = Fc(128,[256], bn=True, activation_fn='relu')
        self.FC4 = Fc(768,[2048], bn=True, activation_fn='relu')

        self.geo1 = Geoconv(64, 128, 64, 0.05, 0.15, bn=True)
        self.geo2 = Geoconv(256, 512, 64, 0.15, 0.3, bn=True)
        self.geo3 = Geoconv(896, 768, 64, 0.3, 0.6, bn=True)
        self.classify = nn.Sequential(
            nn.Linear(2048, 512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(512, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, point_cloud,knn_graph):  #B,C,N
        # print(point_cloud.size())
        # print(xyz.size())
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph[:, :, :16].contiguous())  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(16,64,16,12))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        # x=x.transpose(2,3).contiguous().view(b*n,c,npoints)
        # x=nn.MaxPool1D(npoints)(x)  #--->(b*n,c,1)
        # x=x.squeeze()
        # x=x.view(b,n,c)
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)
        point_cloud=point_cloud.transpose(1,2)
        # xyz=xyz.transpose(1,2)
        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y= self.classify(y)

        return y

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
        for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            self.optimizer.zero_grad()

            outputs = self(inputs,knn_graph)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                outputs = self(inputs,knn_graph)
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


class parall_GeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(parall_GeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        # self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn=True, activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn=True, activation_fn='relu')
        self.FC3 = Fc(384,[512], bn=True, activation_fn='relu')
        self.FC4 = Fc(1536,[1024], bn=True, activation_fn='relu')

        self.geo1_1 = Geoconv(64, 128, 64, 0.05, 0.15, bn=True)
        self.geo1_2 = Geoconv(64, 128, 64, 0.15, 0.3, bn=True)
        self.geo1_3 = Geoconv(64, 128, 64, 0.3, 0.6, bn=True)

        self.geo2_1 = Geoconv(512, 512, 128, 0.05, 0.15, bn=True)
        self.geo2_2 = Geoconv(512, 512, 128, 0.15, 0.3, bn=True)
        self.geo2_3 = Geoconv(512, 512, 128, 0.3, 0.6, bn=True)

        # self.geo3 = Geoconv(896, 768, 64, 0.3, 0.6, bn=True)
        self.classify = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(256, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, point_cloud):  #B,C,N
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        # x = group_points(point_cloud, knn_graph)  #   ---> (B,c,N,npoints)
        # x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # x=self.FC1(x)       #------->(B,N,npoints,384)
        # b,n,npoints,c=x.size()
        # x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)
        point_cloud=point_cloud.transpose(1,2)

        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y_1=self.geo1_1(y,xyz)     #---->(b,n,128)
        y_2=self.geo1_2(y,xyz)
        y_3=self.geo1_3(y,xyz)
        y=torch.cat([y_1,y_2,y_3],2)   #--->(b,n,384)
        y=self.FC3(y)      #----->(b,n,512)
        y_1=self.geo2_1(y,xyz)     #---->(b,n,512)
        y_2=self.geo2_2(y,xyz)     #---->(b,n,512)
        y_3=self.geo2_3(y,xyz)     #---->(b,n,512)
        y=torch.cat([y_1,y_2,y_3],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)
        y= self.classify(y)

        return y

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
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


## add SphericalLinear module/layer into the Geo_net
class SphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(SphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn='BN', activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='BN', activation_fn='relu')
        self.FC4 = Fc(768,[2048], bn='BN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(64, 128, 64, 0.05, 0.15, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        self.classify = nn.Sequential(
            nn.Linear(2048, 512,bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(512, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud):  #B,C,N
        # print(point_cloud.size())
        # print(xyz.size())
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph)  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(16,64,16,12))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        # x=x.transpose(2,3).contiguous().view(b*n,c,npoints)
        # x=nn.MaxPool1D(npoints)(x)  #--->(b*n,c,1)
        # x=x.squeeze()
        # x=x.view(b,n,c)
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)
        point_cloud=point_cloud.transpose(1,2)
        # xyz=xyz.transpose(1,2)
        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        return y

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            self.optimizer.zero_grad()

            outputs = self(inputs)#,knn_graph)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                outputs = self(inputs)#,knn_graph)
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


class TestKNNGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(TestKNNGeoNet,self).__init__()
        self.knn_points = 32
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='BN', activation_fn='relu')
        # self.FC1 = Arpe()
        self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        # self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        # self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='BN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')

        self.geo1 = Geoconv(64, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = Geoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        self.classify = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        debugPrint(self.knn_points)
        self.cuda(device_id)

    def forward(self, point_cloud, knn_graph):  #B,C,N
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        # x = group_points(point_cloud, knn_graph)  #   ---> (B,c,N,npoints)
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
 
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        return y

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

        # for batch_idx, (inputs, _, targets) in enumerate(dataloader):
        for batch_idx, (inputs, knn_graphs, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            knn_graphs = knn_graphs.to(self.device_id, dtype=torch.int)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs,knn_graphs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            for batch_idx, (inputs, knn_graphs, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                knn_graphs = knn_graphs.to(self.device_id, dtype=torch.int)
                targets = targets.cuda(self.device_id)

                outputs = self(inputs,knn_graphs)
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


class TestBallSplitGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(TestBallSplitGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC1 = Arpe()
        self.FC2 = Fc(input_channels,[64], bn='GN', activation_fn='relu')
        # self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        # self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(64, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        self.classify = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )


        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud):  #B,C,N
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph)  #   ---> (B,c,N,npoints)
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
 
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        # ### add ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        # y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        # y = y - point_cloud.unsqueeze(3)
        # y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        # y=y.transpose(1,3).contiguous().view(b,-1,3)
        # y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        # y=y.view(b,-1,n,32)
        # y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        # y=self.FC2_2(y)  #--->(b,n,64)
        # ### END add
        # debugPrint(point_cloud.size())
        point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud
        y=self.FC2(point_cloud)  #---->(b,n,64)
        
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        return y


        # point_cloud=point_cloud.transpose(1,2)
        # # xyz=xyz.transpose(1,2)
        # xyz=point_cloud
        # y=self.FC2(point_cloud)  #---->(b,n,64)
        # # assert(y.size()==(16,64,64))
        # y=self.geo1(y,xyz)     #---->(b,n,128)
        # y=self.FC3(y)      #----->(b,n,256)
        # y=self.geo2(y,xyz)     #---->(b,n,512)
        # y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        # y=self.FC4(y)      #----->(b,n,2048)
        # y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        # y= self.classify(y)

        # return y

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            self.optimizer.zero_grad()

            outputs = self(inputs)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):  #zhuyijie
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


class knnSphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnSphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC2 = Fc(input_channels,[64], bn=True, activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='GN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(128, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.2, 0.3, bn=True)
        # self.geo1 = Geoconv(128, 128, 64, 0.1, 0.2, bn=True)
        # self.geo2 = Geoconv(256, 512, 64, 0.2, 0.3, bn=True)
        #self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        self.classify = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(256, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud,knn_graph):  #B,C,N
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)
        
        ##add ARPE
        # x = x - point_cloud.unsqueeze(3)
        # x=torch.cat([point_cloud.unsqueeze(3),x],3) #---->(b,c,n,1+npoints)
        ##add
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(32,1024,17,3))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()

        # x=x.transpose(2,3).contiguous().view(b*n,c,npoints)
        # x=nn.MaxPool1D(npoints)(x)  #--->(b*n,c,1)
        # x=x.squeeze()
        # x=x.view(b,n,c)
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        ### add ARPE
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### add
        # point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud.transpose(1,2)

        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        #y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        return y

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
        for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            self.optimizer.zero_grad()

            outputs = self(inputs,knn_graph)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                outputs = self(inputs,knn_graph)
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


###ZHUYIJIE 2019.5.30
class knnPD1SphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnPD1SphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        # self.FC1 = Arpe()
        # self.FC2 = Fc(input_channels,[64], bn=True, activation_fn='relu')
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')
        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(128, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        self.classify = nn.Sequential(
            nn.Linear(1024+64, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(256, class_nums)
        )
        # self.classify = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, class_nums)
        # )
        self.pd1_conv=nn.Sequential(
            nn.Conv1d(1,out_channels=4,kernel_size=5,stride=2),#--->(b,4,48)
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv1d(4,out_channels=16,kernel_size=5,stride=2,bias=False),#--->(b,16,10)
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2) #--->(b,16,4)
            )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud,knn_graph, pd1):  #B,C,N
        """
        inputs:
            pd1: float32 tensor shape=(b,100)
        """
        pd1=pd1.unsqueeze(1)
        pd1=self.pd1_conv(pd1)


        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)
        
        ##add ARPE
        # x = x - point_cloud.unsqueeze(3)
        # x=torch.cat([point_cloud.unsqueeze(3),x],3) #---->(b,c,n,1+npoints)
        ##add
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(32,1024,17,3))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()

        # x=x.transpose(2,3).contiguous().view(b*n,c,npoints)
        # x=nn.MaxPool1D(npoints)(x)  #--->(b*n,c,1)
        # x=x.squeeze()
        # x=x.view(b,n,c)
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        ### add ARPE
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### add
        # point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud.transpose(1,2)
        
        # y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)

        pd1=pd1.view(b,-1)
        y=torch.cat([y,pd1],1)  #2019.5.30  -->(b,1024+64)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        return y

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
        for batch_idx, (inputs, knn_graph, pd1, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pd1 = pd1.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer.zero_grad()

            outputs = self(inputs,knn_graph,pd1)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, knn_graph, pd1, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pd1 = pd1.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self(inputs,knn_graph,pd1)
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

###ZHUYIJIE 2019.6.7
class knnPISphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnPISphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')

        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')

        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(128, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        # self.classify_1 = nn.Sequential(
        #     nn.Linear(1024, 256,bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(True)
        # )
        # self.classify_2=nn.Sequential(
        #     nn.Dropout(0.5),
        #     # nn.Linear(512, 128,bias=False),
        #     # nn.BatchNorm1d(128),
        #     # nn.ReLU(True),
        #     # nn.Dropout(0.5),
        #     # nn.Linear(128, class_nums)
        #     nn.Linear(256+64, class_nums)
        # )
        self.classify = nn.Sequential(

            # nn.Dropout(0.3), ##new add
            nn.Linear(1024+512, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            # nn.Linear(512, 128,bias=False),
            # nn.BatchNorm1d(128),
            # nn.ReLU(True),
            # nn.Dropout(0.5),
            # nn.Linear(128, class_nums)
            nn.Linear(256, class_nums)
        )

        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, point_cloud,knn_graph, pi):  #B,C,N
        """
        inputs:
            point_cloud: float32 tensor shape=(b, 3, n)
            knn_graph: int32 tensor shape=(b, n, 17)
            pi: float32 tensor shape=(b,50,50)
        """
        pi=pi.unsqueeze(1)
        pi=self.pi_conv(pi)


        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points)
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)
        
        ##add ARPE
        # x = x - point_cloud.unsqueeze(3)
        # x=torch.cat([point_cloud.unsqueeze(3),x],3) #---->(b,c,n,1+npoints)
        ##add
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        # assert(x.size()==(32,1024,17,3))
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        ### add ARPE
        knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ### end add ARPE
        # point_cloud=point_cloud.transpose(1,2)
        xyz=point_cloud.transpose(1,2)
        
        # y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)

        pi=pi.view(b,-1)
        y=torch.cat([y,pi],1)  #2019.6.7  -->(b,1024+128*2*2)
        y= self.classify(y)
        # print(self.geo1.perceptron_feat.weight[0])
        # y=self.classify_1(y)
        # y=torch.cat([y,pd1],1)##--->(b,256+64)
        # y=self.classify_2(y)

        return y

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer.zero_grad()

            outputs = self(inputs,knn_graph,pi)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
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
            for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self(inputs,knn_graph,pi)
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


###zhuyijie 2019.6.8
class pretrained_knnPISphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(pretrained_knnPISphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        self.best_score=0
        self.name='pretrained_knnPISphericalGeoNet_concate'
        print(self.name)
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,128],input_dim=4, bn='GN', activation_fn='relu')
        #ARPE
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')
        self.FC2_second = Fc(input_channels,[32,128], bn='GN', activation_fn='relu')

        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')
        self.FC4_second = Fc(512,[1024], bn='BN', activation_fn='relu')

        self.geo1 = SphericalGeoconv(128, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        self.classify_kc = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.classify_first = nn.Sequential(
            nn.Linear(128, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )
        self.classify_second = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )
        self.classify_pi=nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
        )
        self.classify = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )
        self.classify_new = nn.Sequential(
            # nn.Linear(256+256, class_nums)
            nn.Dropout(0.5),
            nn.Linear(256+256, class_nums)
        )

        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )

        self.criterion = nn.CrossEntropyLoss()
        # self.optimizer = optim.Adam(self.parameters(),lr=1e-5, weight_decay=1e-5)
        # self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        ####2019.6.7
        self.kcparams=[]
        for name, param in self.named_parameters():
            if 'pi' not in name:
                if 'classify' not in name:
                    self.kcparams.append(param)
        self.classify_param=[]
        for name, param in self.named_parameters():
            if 'classify' in name:
                self.classify_param.append(param)
        
        self.optimizer1 = optim.Adam([{'params':self.classify_pi.parameters()},
                                    {'params':self.classify.parameters()},
                                    {'params':self.pi_conv.parameters()}],
                                    weight_decay=1e-5)
        self.schedule1 = optim.lr_scheduler.StepLR(self.optimizer1, 10, 0.6)

        # self.optimizer2 = optim.Adam(self.kcparams, weight_decay=1e-5)
        self.optimizer2 = optim.Adam([{'params':self.kcparams},
                                       {'params':self.classify_kc.parameters()},
                                       {'params': self.classify.parameters()}],
                                        lr=1e-3,weight_decay=1e-5)
        self.schedule2 = optim.lr_scheduler.StepLR(self.optimizer2, 10, 0.6)

        self.optimizer3 = optim.Adam([{'params':self.kcparams, 'lr':1e-5},
                                    {'params':self.classify_param, 'lr':1e-3},
                                    {'params':self.pi_conv.parameters(),'lr':1e-5}],
                                     lr=1e-5, weight_decay=1e-5)
        self.schedule3 = optim.lr_scheduler.StepLR(self.optimizer3, 6, 0.6)

        ## 2.3
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        ##

        self.cuda(device_id)
    # @profile
    def forward(self, point_cloud,knn_graph, pi):  #B,C,N
        """
        inputs:
            point_cloud: float32 tensor shape=(b, 3, n)
            knn_graph: int32 tensor shape=(b, n, 17)
            pi: float32 tensor shape=(b,50,50)
        """
        y, _ = self.forward_kc(point_cloud, knn_graph)
        
        b=pi.size(0)
        pi=self.pi_conv(pi.unsqueeze(1))
        pi=self.classify_pi(pi.view(b,-1))  ##--->(b,256)
        
        # y=torch.div(torch.add(y,1,pi),0.5)    ##pretrained_knnPISphericalGeoNet
        # y=y.mul(pi)
        # y=torch.add(y,1,pi)
        y=torch.cat([y,pi],1)  #2019.6.7  -->(b,256+128*2*2)
        # y=torch.div(torch.add(pi,1,y),2)
        # y=torch.max(y,pi)
        # y=F.relu(y)

        # y= self.classify(y)
        y= self.classify_new(y)

        return y

    def forward_pi(self,pi):
        b,h,w=pi.size()
        pi=pi.unsqueeze(1)
        pi=self.pi_conv(pi)
        pi=pi.view(b,-1)
        # y=torch.cat([y,pi],1)  #2019.6.7  -->(b,1024+128*2*2)
        outputs=self.classify_pi(pi)
        outputs= self.classify(outputs)
        return outputs

    def forward_kc(self, point_cloud, knn_graph=None):
        if knn_graph is None:
            knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
            # x = group_points(point_cloud, knn_graph)
        
        x = group_points(point_cloud, knn_graph[:, :, :16].contiguous()) #---> (B,c,N,npoints)    
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ##end ARPE

        xyz=point_cloud.transpose(1,2)
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)
        y=self.classify_kc(y) #2019.6.7  --->(b,256)
        outputs= self.classify(y)
        return y, outputs

    def forward_first(self, point_cloud, knn_graph=None):
        if knn_graph is None:
            knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
            # x = group_points(point_cloud, knn_graph)
        
        x = group_points(point_cloud, knn_graph[:, :, :16].contiguous()) #---> (B,c,N,npoints)    
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)

        # y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.FC4(y)      #----->(b,n,1024)
        y=x
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,128)
        y=self.classify_first(y) #2019.6.7  --->(b,256)
        
        return y

    def forward_second(self, point_cloud, knn_graph=None):
        if knn_graph is None:
            knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
            # x = group_points(point_cloud, knn_graph)

        b,c,n=point_cloud.size()
        # debugPrint(point_cloud.size())

        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,128)
        ##end ARPE


        xyz=point_cloud.transpose(1,2)
        
        # y = self.FC2_second(xyz)  # 2.4, 3--->32---->128

        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        
        y=self.FC4_second(y)      #(x,x,512)----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)
        y=self.classify_second(y) #2019.6.7  --->(b,256)
        return y

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)


    def fit_pi(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule1 is not None:
            self.schedule1.step()

        # print('----------epoch %d start train pi----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (pi, targets) in enumerate(dataloader):  #zhuyijie
            # inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pi = pi.to(device=self.device_id,dtype=torch.float32)
            self.optimizer1.zero_grad()

            # outputs = self(inputs,knn_graph,pi)
            outputs= self.forward_pi(pi)
            ##
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer1.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                # print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        # print('-----------epoch %d end train pi-----------' % epoch)
        print('train pi epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

    def fit_kc(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule2 is not None:
            self.schedule2.step()

        print('----------epoch %d start train kc----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (inputs, knn_graph, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
            self.optimizer2.zero_grad()

            _, outputs= self.forward_kc(inputs, knn_graph)
            ###

            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer2.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train kc-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

    def fit_second(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train second----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (inputs, knn_graph, _, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
            self.optimizer.zero_grad()

            outputs= self.forward_second(inputs, knn_graph)
            ###

            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train second-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

    def fit(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        for param_group in self.optimizer3.param_groups:
             print("current learning rate={}".format(param_group['lr']))
        if self.schedule3 is not None:
            self.schedule3.step()

        print('----------epoch %d start train----------' % epoch)

        for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
            pi = pi.to(device=self.device_id,dtype=torch.float32)
        # inputs, knn_graph, pi, targets = dataloader.next()
        # print(targets)
        # batch_idx = -1
        # while inputs is not None:
        #     batch_idx += 1
            self.optimizer3.zero_grad()

            outputs = self(inputs,knn_graph,pi)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer3.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.
            # inputs, knn_graph, pi, targets = dataloader.next()

        print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score_pi(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (pi, targets) in enumerate(dataloader):  #zhuyijie
                # inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pi = pi.to(device=self.device_id,dtype=torch.float32)
                outputs = self.forward_pi(pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the PI network: %.2f %%' % (100.0 * correct / total))

        return correct / total

    def score_kc(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (inputs, knn_graph, _, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
               
                _, outputs = self.forward_kc(inputs,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        # print('Accuracy of the KC network: %.2f %%' % (100.0 * correct / total))
        score = 100.0 * correct / total             
        print('Accuracy of the KC network: %.2f %%' % score)

        if score>self.best_score:
            self.best_score=score
        print('------- The best score is: %.2f %%' % self.best_score)

        return correct / total

    def score_second(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (inputs, knn_graph, _, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
               
                outputs = self.forward_second(inputs,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        score = 100.0 * correct / total             
        print('Accuracy of the KC network: %.2f %%' % score)

        if score>self.best_score:
            self.best_score=score
        print('------- The best score is: %.2f %%' % self.best_score)

        return correct / total
    # @profile
    def score(self, dataloader,is_save=False):
        self.eval()
        correct = 0.
        total = 0
        for param_group in self.optimizer3.param_groups:
             print("current learning rate={}".format(param_group['lr']))
        with torch.no_grad():
            for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
                pi = pi.to(device=self.device_id,dtype=torch.float32)
            # inputs, knn_graph, pi, targets = dataloader.next()
            # iteration = 0
            # while inputs is not None:
            #     iteration += 1
                # 训练代码             
                outputs = self(inputs,knn_graph,pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                # inputs, knn_graph, pi, targets = dataloader.next()

        score=correct / total
        print('Accuracy of the total network: %.2f %%' % (100.0 * score))        
        if is_save:
            if self.best_score<score:
                self.best_score=score
                torch.save(self, './model_param/{}_best_weight_1120.ckpt'.format(self.name))

        return score

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


###zhuyijie 2019.11.14
class first_two_knnPISphericalGeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(first_two_knnPISphericalGeoNet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        self.best_score=0
        self.name='first_two_knnPISphericalGeoNet'
        print(self.name)
        if initial_weights:
            self.initialize_weights()

        self.FC1 = Fc(input_channels,[32,64,128],input_dim=4, bn='GN', activation_fn='relu')
        #ARPE
        self.FC2_1 = Fc(input_channels,[32], bn='GN', activation_fn='relu')
        self.FC2_2 = Fc(32,[128], bn='GN', activation_fn='relu')

        self.FC3 = Fc(128,[256], bn='GN', activation_fn='relu')
        self.FC4 = Fc(640,[1024], bn='BN', activation_fn='relu')

        # self.geo1 = SphericalGeoconv(128, 128, 64, 0.1, 0.2, bn=True)
        # self.geo2 = SphericalGeoconv(256, 512, 64, 0.15, 0.3, bn=True)
        self.geo1 = Geoconv(128, 128, 64, 0.1, 0.2, bn=True)
        self.geo2 = Geoconv(256, 512, 64, 0.15, 0.3, bn=True)
        # self.geo3 = SphericalGeoconv(896, 768, 64, 0.3, 0.6, bn=True)
        
        self.classify_kc = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, class_nums)
        )
        # self.classify_pi=nn.Sequential(
        #     # nn.Dropout(0.5),
        #     nn.Linear(512, 256, bias=False),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(True)
        # )
        # self.classify = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(256, class_nums)
        # )

        ##### 50*50
        # self.pi_conv=nn.Sequential(
        #     nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
        #     nn.BatchNorm2d(16),
        #     # nn.GroupNorm(4, 8),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
        #     nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
        #     nn.BatchNorm2d(64),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        #     nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
        #     nn.BatchNorm2d(128),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU()
        # )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        self.cuda(device_id)

    def forward_kc(self, point_cloud, knn_graph=None):
        knn_graph=None
        if knn_graph is None:
            knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        # debugPrint(knn_graph.size())
        x = group_points(point_cloud, knn_graph[:, :, :16].contiguous()) #---> (B,c,N,npoints)    
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
        x=self.FC1(x)       #------->(B,N,npoints,384)
        b,n,npoints,c=x.size()
        x=torch.max(x, 2, keepdim=False)[0]  #--->(b,n,384)
        ### begin ARPE
        # knn_graph, _ = batch_knn(point_cloud, point_cloud.clone(), self.knn_points*2)
        y = group_points(point_cloud, knn_graph[:, :, 1:].contiguous())
        y = y - point_cloud.unsqueeze(3)
        y=torch.cat([point_cloud.unsqueeze(3),y],3) #---->(b,c,n,1+npoints)
        y=y.transpose(1,3).contiguous().view(b,-1,3)
        y=self.FC2_1(y)  #--->(b,(1+npoints)*n,32)
        y=y.view(b,-1,n,32)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,n,32)
        y=self.FC2_2(y)  #--->(b,n,64)
        ##end ARPE

        xyz=point_cloud.transpose(1,2)
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y=self.classify_kc(y) #2019.11 --->(b,40)
        # outputs= self.classify(y)
        return y

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit_kc(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        print('----------epoch %d start train kc----------' % epoch)

        for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
            inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
            self.optimizer.zero_grad()

            outputs= self.forward_kc(inputs, knn_graph)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0:
                print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        print('-----------epoch %d end train kc-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

    def score_kc(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (inputs, knn_graph, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
               
                outputs = self.forward_kc(inputs,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the KC network: %.2f %%' % (100.0 * correct / total))

        return correct / total
    # @profile
    """
    def score(self, dataloader,is_save=False):
        self.eval()
        correct = 0.
        total = 0
        for param_group in self.optimizer3.param_groups:
             print("current learning rate={}".format(param_group['lr']))
        with torch.no_grad():
            for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(device=self.device_id,dtype=torch.int)
                pi = pi.to(device=self.device_id,dtype=torch.float32)
            # inputs, knn_graph, pi, targets = dataloader.next()
            # iteration = 0
            # while inputs is not None:
            #     iteration += 1
                # 训练代码             
                outputs = self(inputs,knn_graph,pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                # inputs, knn_graph, pi, targets = dataloader.next()

        score=correct / total
        print('Accuracy of the total network: %.2f %%' % (100.0 * score))        
        if is_save:
            if self.best_score<score:
                self.best_score=score
                torch.save(self, './model_param/{}_best_weight_64.ckpt'.format(self.name))

        return score
    """

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


###ZHUYIJIE 2019.5.30
class PD1Net(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(PD1Net,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()
       
        self.classify = nn.Sequential(
            nn.Dropout(0.5),    
            nn.Linear(800, class_nums)
        )
        self.pd1_conv=nn.Sequential(
            nn.Conv1d(1,out_channels=4,kernel_size=2,stride=2),#--->(b,4,200)
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv1d(4,out_channels=8,kernel_size=2,stride=2,bias=False),#--->(b,8,100)
            # nn.BatchNorm1d(16),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv1d(8,out_channels=16,kernel_size=2,stride=2,bias=False),#--->(b,16,50)
            # nn.BatchNorm1d(16),
            nn.GroupNorm(4, 16),
            nn.ReLU()
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        )

        # self.pd1_conv=nn.Sequential(
        #     nn.Conv1d(1,out_channels=4,kernel_size=2,stride=2),#--->(b,4,50)
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv1d(4,out_channels=16,kernel_size=2,stride=2,bias=False),#--->(b,16,25)
        #     # nn.BatchNorm1d(16),
        #     nn.GroupNorm(4, 16),
        #     nn.ReLU()
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        # )

        # self.pd1_conv=nn.Sequential(
        #     nn.Conv1d(1,out_channels=4,kernel_size=5,stride=2),#--->(b,4,48)
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv1d(4,out_channels=16,kernel_size=5,stride=2,bias=False),#--->(b,16,10)
        #     # nn.BatchNorm1d(16),
        #     nn.GroupNorm(4, 16),
        #     nn.ReLU()
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,4)
        # )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)

    def forward(self, pd1):  #B,C,N
        """
        inputs:
            pd1: float32 tensor shape=(b,100)
        """
        b,c=pd1.size()
        pd1=pd1.unsqueeze(1)
        pd1=self.pd1_conv(pd1)
        pd1=pd1.view(b,-1)
        y= self.classify(pd1)
        return y

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        # print('----------epoch %d start train----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (pd1, targets) in enumerate(dataloader):  #zhuyijie
            # inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pd1 = pd1.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer.zero_grad()

            outputs = self(pd1)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                # print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        # print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (pd1, targets) in enumerate(dataloader):  #zhuyijie
                # inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pd1 = pd1.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self(pd1)
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


###ZHUYIJIE 2019.5.30
class PINet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(PINet,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()
       
        self.classify = nn.Sequential(
            # nn.Linear(1024, 256,bias=False),
            # nn.BatchNorm1d(256),
            # nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, class_nums)
            # nn.Dropout(0.5),    
            # nn.Linear(128*4, class_nums)
        )

        ##### 20*20
        # self.pi_conv=nn.Sequential(
        #     nn.Conv2d(1,out_channels=4,kernel_size=2,stride=1,bias=False),#--->(b,4,19,19)
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,9,9)
        #     nn.BatchNorm2d(16),
        #     # nn.GroupNorm(4, 8),
        #     nn.ReLU(),
        #     nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,4,4)
        #     nn.BatchNorm2d(64),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        #     nn.Conv2d(64,out_channels=128,kernel_size=2,stride=2,bias=False),#--->(b,128,2,2)
        #     nn.BatchNorm2d(128),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU()
        # )

        # ##### 50*50 good
        # self.pi_conv=nn.Sequential(
        #     nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,24,24)
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
        #     nn.BatchNorm2d(16),
        #     # nn.GroupNorm(4, 8),
        #     nn.ReLU(),
        #     # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
        #     nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
        #     nn.BatchNorm2d(64),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        #     nn.Conv2d(64,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
        #     nn.BatchNorm2d(256),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1), #--->(b,256,1,1)
        #     # nn.Conv2d(128,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
        #     # nn.BatchNorm2d(256),
        #     # # nn.GroupNorm(4, 16),
        #     # nn.ReLU()
        # )

        ##### 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,23,23)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,11,11)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            # nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool2d(2, stride=1), #--->(b,256,1,1)
            # nn.Conv2d(128,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            # nn.BatchNorm2d(256),
            # # nn.GroupNorm(4, 16),
            # nn.ReLU()
        )

        # ##### 100*100
        # self.pi_conv=nn.Sequential(
        #     nn.Conv2d(1,out_channels=4,kernel_size=5,stride=2,bias=False),#--->(b,4,48,48)
        #     nn.BatchNorm2d(4),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,23,23)
        #     nn.BatchNorm2d(16),
        #     # nn.GroupNorm(4, 8),
        #     nn.ReLU(),
        #     nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
        #     nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
        #     nn.BatchNorm2d(64),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
        #     nn.Conv2d(64,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
        #     nn.BatchNorm2d(256),
        #     # nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, stride=1), #--->(b,256,1,1)
        #     # nn.Conv2d(128,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
        #     # nn.BatchNorm2d(256),
        #     # # nn.GroupNorm(4, 16),
        #     # nn.ReLU()
        # )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        self.cuda(device_id)
        

    def forward(self, pi):  #B,C,H,W
        """
        inputs:
            
        """
        pi=pi.unsqueeze(1)
        b,c,h,w=pi.size()
        # pd1=pd1.unsqueeze(1)
        pi=self.pi_conv(pi)

        pi=pi.view(b,-1)
        y= self.classify(pi)
        return y

    def loss(self, outputs, targets):
        return self.criterion(outputs, targets)

    def fit(self, dataloader, epoch, writer=None):
        global global_step
        self.train()
        batch_loss = 0.
        epoch_loss = 0.
        batch_nums = 0
        if self.schedule is not None:
            self.schedule.step()

        # print('----------epoch %d start train----------' % epoch)

        # for batch_idx, (inputs, targets) in enumerate(dataloader):
        for batch_idx, (pi, targets) in enumerate(dataloader):  #zhuyijie
            # inputs = inputs.cuda(self.device_id)
            targets = targets.cuda(self.device_id)
            # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer.zero_grad()

            outputs = self(pi)
            losses = self.loss(outputs, targets)
            losses.backward()
            self.optimizer.step()

            batch_loss += losses.item()
            epoch_loss += losses.item()
            batch_nums += 1
            if (batch_idx + 1) % 8 == 0: #batch_size=16    16*8=128 samples
                # print('[%d, %5d] loss %.3f' % (epoch, batch_idx, batch_loss / 8))
                global_step += 1
                # print('global_step={}'.format(global_step))
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

        # print('-----------epoch %d end train-----------' % epoch)
        print('epoch %d loss %.3f' % (epoch, epoch_loss / batch_nums))

        return epoch_loss / batch_nums

    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0

        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (pi, targets) in enumerate(dataloader):  #zhuyijie
                # inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                # knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self(pi)
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
