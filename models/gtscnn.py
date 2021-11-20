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
from ops.layers import Fc, Perceptron, Geoconv, SphericalGeoconv
from utils.misc import Netpara

# global writer
global_step=0

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
        for batch_idx, (inputs, _, targets) in enumerate(dataloader):
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
            for batch_idx, (inputs, _, targets) in enumerate(dataloader):
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
            for batch_idx, (inputs, _, targets, labels) in enumerate(dataloader): #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                labels = labels.cuda(self.device_id)

                outputs = self(inputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0) * targets.size(1)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total


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


class GeoNet(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(GeoNet,self).__init__()
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


class AdaptiveGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(AdaptiveGTSCNN,self).__init__()
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



class parall_GTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(parall_GTSCNN,self).__init__()
        self.knn_points = 16
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        if initial_weights:
            self.initialize_weights()

        # self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn=True, activation_fn='relu')
        self.FC2 = Fc(input_channels,[64], bn='BN', activation_fn='relu')
        self.FC3 = Fc(384,[512], bn='BN', activation_fn='relu')
        self.FC4 = Fc(1536,[1024], bn='BN', activation_fn='relu')

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
class SphericalGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(SphericalGTSCNN,self).__init__()
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
            nn.Linear(512, class_nums)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 20, 0.5)

        self.cuda(device_id)

    def forward(self, point_cloud,knn_graph):  #B,C,N
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


class knnSphericalGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnSphericalGTSCNN,self).__init__()
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
        
        # y=self.FC2(point_cloud)  #---->(b,n,64)
        # assert(y.size()==(16,64,64))
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
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



# 2019.5.30
class knnPD1SphericalGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnPD1SphericalGTSCNN,self).__init__()
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
            nn.Linear(1024+400, 256,bias=False),
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

        self.pd1_conv=nn.Sequential(
            nn.Conv1d(1,out_channels=4,kernel_size=2,stride=2),#--->(b,4,50)
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv1d(4,out_channels=16,kernel_size=2,stride=2,bias=False),#--->(b,16,25)
            # nn.BatchNorm1d(16),
            nn.GroupNorm(4, 16),
            nn.ReLU()
            # nn.MaxPool1d(3, stride=2) #--->(b,16,4)
        )
        # self.pd1_conv=nn.Sequential(
        #     nn.Conv1d(1,out_channels=4,kernel_size=5,stride=2),#--->(b,4,48)
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=2), #--->(b,4,23)
        #     nn.Conv1d(4,out_channels=16,kernel_size=5,stride=2,bias=False),#--->(b,16,10)
        #     # nn.BatchNorm1d(16),
        #     nn.GroupNorm(4, 16),
        #     nn.ReLU(),
        #     nn.MaxPool1d(3, stride=2) #--->(b,16,4)
        # )

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
        # y=self.classify_1(y)
        # y=torch.cat([y,pd1],1)##--->(b,256+64)
        # y=self.classify_2(y)

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


class knnPISphericalGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(knnPISphericalGTSCNN,self).__init__()
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
        
        self.classify_kc = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
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

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(),lr=1e-5, weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)

        ####2019.6.7
        self.kcparams=[]
        for name, param in self.named_parameters():
            if 'pi' not in name:
                self.kcparams.append(param)
        self.optimizer1 = optim.Adam([{'params':self.classify_pi.parameters()},
                                    {'params':self.classify.parameters()},
                                    {'params':self.pi_conv.parameters()}],
                                    weight_decay=1e-5)
        self.schedule1 = optim.lr_scheduler.StepLR(self.optimizer1, 10, 0.6)

        self.optimizer2 = optim.Adam(self.kcparams, weight_decay=1e-5)
        self.schedule2 = optim.lr_scheduler.StepLR(self.optimizer2, 10, 0.6)

        # self.optimizer3 = optim.Adam(self.parameters(), lr=1e-5, weight_decay=1e-5)
        # self.schedule3 = optim.lr_scheduler.StepLR(self.optimizer3, 10, 0.6)

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
        y=self.FC4(y)      #----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)

        pi=pi.view(b,-1)
        
        # y=torch.cat([y,pi],1)  #2019.6.7  -->(b,1024+128*2*2)
        # y= self.classify(y)
        y=self.classify_kc(y) #2019.6.7  --->(b,256)
        pi=self.classify_pi(pi)  ##--->(b,256)
        y=torch.div(torch.add(y,1,pi),2)
        y= self.classify(y)


        # print(self.geo1.perceptron_feat.weight[0])
        # y=self.classify_1(y)
        # y=torch.cat([y,pd1],1)##--->(b,256+64)
        # y=self.classify_2(y)

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

    def forward_kc(self, point_cloud, knn_graph):
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)    
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
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
        ##end ARPE
        xyz=point_cloud.transpose(1,2)
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y=self.classify_kc(y)
        outputs= self.classify(y)
        return outputs

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
            pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
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
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            # pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer2.zero_grad()

            # outputs = self(inputs,knn_graph,pi)
            outputs= self.forward_kc(inputs, knn_graph)
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
                pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self.forward_pi(pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

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
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                # pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self.forward_kc(inputs,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total

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


class pretrained_knnPISphericalGTSCNN(nn.Module):
    def __init__(self, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(pretrained_knnPISphericalGTSCNN,self).__init__()
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
        
        self.classify_kc = nn.Sequential(
            nn.Linear(1024, 256,bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True)
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

        self.optimizer2 = optim.Adam(self.kcparams, weight_decay=1e-5)
        self.schedule2 = optim.lr_scheduler.StepLR(self.optimizer2, 10, 0.6)

        self.optimizer3 = optim.Adam([{'params':self.kcparams, 'lr':1e-5},
                                    {'params':self.classify_param, 'lr':1e-3},
                                    {'params':self.pi_conv.parameters(),'lr':1e-5}],
                                     lr=1e-5, weight_decay=1e-5)
        self.schedule3 = optim.lr_scheduler.StepLR(self.optimizer3, 6, 0.6)

        self.cuda(device_id)
    # @profile
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
        y=self.FC4(y)      #----->(b,n,1024)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,1024)

        pi=pi.view(b,-1)
        
        # y=torch.cat([y,pi],1)  #2019.6.7  -->(b,1024+128*2*2)
        # y= self.classify(y)
        y=self.classify_kc(y) #2019.6.7  --->(b,256)
        pi=self.classify_pi(pi)  ##--->(b,256)
        # y=torch.div(torch.add(y,1,pi),2)    ##pretrained_knnPISphericalGTSCNN
        y=torch.div(torch.add(pi,5,y),6)
        # y=y.mul(pi)
        # y=torch.add(y,1,pi)
        y= self.classify(y)


        # print(self.geo1.perceptron_feat.weight[0])
        # y=self.classify_1(y)
        # y=torch.cat([y,pd1],1)##--->(b,256+64)
        # y=self.classify_2(y)

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

    def forward_kc(self, point_cloud, knn_graph):
        x = group_points(point_cloud, knn_graph[:, :, 1:].contiguous()) #---> (B,c,N,npoints)    
        x=x.permute(0,2,3,1) #---->(B,N,npoints,c)
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
        ##end ARPE
        xyz=point_cloud.transpose(1,2)
        y=self.geo1(y,xyz)     #---->(b,n,128)
        y=self.FC3(y)      #----->(b,n,256)
        y=self.geo2(y,xyz)     #---->(b,n,512)
        y=torch.cat([y,x],2) #---->(b,n,896)
        # y=self.geo3(y,xyz)     #---->(b,n,768)
        y=self.FC4(y)      #----->(b,n,2048)
        y=torch.max(y, 1, keepdim=False)[0]  #--->(b,2048)
        y=self.classify_kc(y)
        outputs= self.classify(y)
        return outputs

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
            pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
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
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            # pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
            self.optimizer2.zero_grad()

            # outputs = self(inputs,knn_graph,pi)
            outputs= self.forward_kc(inputs, knn_graph)
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
            knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
            pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
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
                pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self.forward_pi(pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

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
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                # pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
                outputs = self.forward_kc(inputs,knn_graph)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        print('Accuracy of the network on the test images: %.2f %%' % (100.0 * correct / total))

        return correct / total
    # @profile
    def score(self, dataloader):
        self.eval()
        correct = 0.
        total = 0
        for param_group in self.optimizer3.param_groups:
             print("current learning rate={}".format(param_group['lr']))
        with torch.no_grad():
            # for batch_idx, (inputs, targets) in enumerate(dataloader):
            for batch_idx, (inputs, knn_graph, pi, targets) in enumerate(dataloader):  #zhuyijie
                inputs = inputs.cuda(self.device_id)
                targets = targets.cuda(self.device_id)
                knn_graph = knn_graph.to(torch.device('cuda:0'),dtype=torch.int)
                pi = pi.to(torch.device('cuda:0'),dtype=torch.float32)
            # inputs, knn_graph, pi, targets = dataloader.next()
            # iteration = 0
            # while inputs is not None:
            #     iteration += 1            
                outputs = self(inputs,knn_graph,pi)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                # inputs, knn_graph, pi, targets = dataloader.next()

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
            nn.Linear(512, class_nums)
        )
        self.pd1_conv=nn.Sequential(
            nn.Conv1d(1,out_channels=4,kernel_size=2,stride=2),#--->(b,4,128)
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv1d(4,out_channels=8,kernel_size=2,stride=2,bias=False),#--->(b,8,64)
            # nn.BatchNorm1d(8),
            nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv1d(8,out_channels=16,kernel_size=2,stride=2,bias=False),#--->(b,16,32)
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

    def forward(self, pd1):
        """
        inputs:
            pd1: float32 tensor shape=(b,100) / (b,256), (B,C,N).
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
        ## 16*16
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=2,stride=1,bias=False),#--->(b,4,15,15)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,7,7)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,3,3)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=2,stride=1,bias=False),#--->(b,128,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )
        '''
        ## 20*20
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=2,stride=1,bias=False),#--->(b,4,19,19)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,9,9)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,4,4)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=128,kernel_size=2,stride=2,bias=False),#--->(b,128,2,2)
            nn.BatchNorm2d(128),
            # nn.GroupNorm(4, 16),
            nn.ReLU()
        )

        ## 50*50
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=4,stride=2,bias=False),#--->(b,4,24,24)
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
            nn.Conv2d(64,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            nn.BatchNorm2d(256),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1), #--->(b,256,1,1)
            # nn.Conv2d(128,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            # nn.BatchNorm2d(256),
            # # nn.GroupNorm(4, 16),
            # nn.ReLU()
        )
  
        ## 100*100
        self.pi_conv=nn.Sequential(
            nn.Conv2d(1,out_channels=4,kernel_size=5,stride=2,bias=False),#--->(b,4,48,48)
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2), #--->(b,4,23)
            nn.Conv2d(4,out_channels=16,kernel_size=3,stride=2,bias=False),#--->(b,16,23,23)
            nn.BatchNorm2d(16),
            # nn.GroupNorm(4, 8),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2), #--->(b,16,11,11)
            nn.Conv2d(16,out_channels=64,kernel_size=3,stride=2,bias=False),#--->(b,64,5,5)
            nn.BatchNorm2d(64),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            # nn.MaxPool1d(3, stride=2) #--->(b,16,12)
            nn.Conv2d(64,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            nn.BatchNorm2d(256),
            # nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1), #--->(b,256,1,1)
            # nn.Conv2d(128,out_channels=256,kernel_size=3,stride=2,bias=False),#--->(b,256,2,2)
            # nn.BatchNorm2d(256),
            # # nn.GroupNorm(4, 16),
            # nn.ReLU()
        )
        '''

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        self.cuda(device_id)
        

    def forward(self, pi):  
        """
        inputs: shape (B,C,H,W)
            
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
            for batch_idx, (pi, targets) in enumerate(dataloader):
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
