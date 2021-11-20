import torch
import torch.nn as nn
import torch.optim as optim
from apex import amp
# import tf_util as tf_util
from .convolution_elements import ConvElements
from .convolution_layer import ConvLayer
from .pooling import PoolingLayer
from utils.kcnet_utils import debugPrint
import numpy as np

global_step=0

class PCNN(nn.Module):
    """
    This is PCNN(Point Convolutional Neural Networks by Extension Operators) NETWORK
    """
    def __init__(self, conf, input_channels, class_nums=1, device_id=0, initial_weights=True):
        super(PCNN, self).__init__()
        self.conf = conf
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id

        self.with_bn = self.conf.get_bool("with_bn") # default: True
        self.is_rotations = self.conf["with_rotations"] # default: False
        self.pool_sizes_sigma = self.conf.get_list('pool_sizes_sigma') #[[1024,0.03125],[256,0.0625],[64,0.125],[1,1.0]]
        self.spacing = self.conf.get_float('kernel_spacing') # 2.0
        self.kernel_sigma_factor=self.conf.get_float('kernel_sigma_factor') # 1.0
        self.blocks = self.conf.get_list('blocks_out_channels') #[[64],[256],[1024]]
        self.is_interpolation = self.conf.get_bool('interpolation') # False
        # if self.is_rotations:
        #     self.rotate_layer=self
        input_channel = self.input_channels + 1
        
        self.convlayers = nn.ModuleList([])
        for block_index, block in enumerate(self.blocks):
            for out_index,out_channel in enumerate(block): 
                convlayer = ConvLayer(input_channel,
                                    out_channel, 
                                    27,
                                    '{0}_block_{1}'.format(block_index,out_index),
                                    self.is_interpolation,
                                    )
                input_channel = out_channel
                self.convlayers.append(convlayer)
            # self.poolingLayers.append(0)

        self.classify = nn.Sequential(
            nn.Linear(1024, self.conf.get_int('fc1.size'),bias=False),
            nn.BatchNorm1d(self.conf.get_int('fc1.size')),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.conf.get_int('fc1.size'), self.conf.get_int('fc2.size'),bias=False),
            nn.BatchNorm1d(self.conf.get_int('fc2.size')),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(self.conf.get_int('fc2.size'), class_nums)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        # if initial_weights: 
        #     self.initialize_weights()

        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        self.cuda(device_id)
        

    def forward(self, pc):  # B,C,N--->#B,N,C=3
        # if self.is_rotations: # False
        #     pc=self.rotation(pc)
        pc=pc.transpose(1,2)
        batch_size, num_point,num_channel = pc.size()
        network = torch.cat((pc, torch.ones(batch_size,num_point,1,device=self.device_id, dtype=torch.float32)),dim=2) # (b,n,4)
        input_channel = num_channel+1
        index=0
        for block_index, block in enumerate(self.blocks):
            # e.g, block_index=0, block=[64]
            block_elm = ConvElements(pc,
                                    1.0/np.sqrt(1.0*num_point),
                                    self.spacing,
                                    self.kernel_sigma_factor)
            for out_index,out_channel in enumerate(block):
                network = self.convlayers[index](block_elm, network)
                input_channel = out_channel
                index+=1
            pc, network = PoolingLayer(block_elm, 
                                       out_channel, 
                                       out_channel,
                                       int(self.pool_sizes_sigma[block_index + 1][0])).get_layer(
                                       network,
                                       is_subsampling=self.conf.get_bool('subsampling'),# False
                                       use_fps= True)

        network = network.view(batch_size, -1)
        # debugPrint(network.size())
        network = self.classify(network)
        return network

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
            
            # losses.backward()
            with amp.scale_loss(losses, self.optimizer) as scaled_loss:
                scaled_loss.backward()

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
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def rotation(self, pc):
        """
        pc: point clouds, shape (b,n,3) 
        return:
            rotated point clouds with shape (b,n,3)
        """
        batch_size, num_point, num_channel = pc.size()
        cov = self.cov(pc)
        _, axis = torch.symeig(cov)
        axis = torch.where(torch.det(axis) < 0, torch.matmul(axis,
            tf.tensor([[[0, 1], [1, 0]]], dtype=torch.float32).repeat(axis.size(0), 1, 1)), axis)

        # indicies = [[[b, 0, 0], [b, 2, 0], [b, 0, 2], [b, 2, 2]] for b in list(range(batch_size))] # (b,4,3)
        indices=torch.tensor([0,6,2,8]).repeat(batch_size,1)
        updates = axis.view(batch_size, -1) # (b,4)
        updates = torch.matmul(tf.constant([[[0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0]]], dtype=torch.float32).repeat(batch_size, 1, 1), 
                                updates.unsqueeze(-1)).view(batch_size, -1) # (b,4,)

        alignment_transform = torch.zeros(batch_size,9).scatter_(dim=0,index=indicies, src=updates).view(batch_size,3,3)+torch.diag([0.0, 1.0, 0.0]).unsqueeze(0) #--->(b,3,3)
        mean_points = torch.mean(pc, dim=1, keepdims=True)
        pc = torch.matmul(pc - mean_points, alignment_transform) + mean_points

        return pc

    def cov(self, x):
        x = torch.index_select(x,[0,2],dim=2).transpose(1,2) #(x,y,z)-->(x,z)  shape(b,n,2)
        mean_x = torch.mean(x, dim=1, keepdims=True) # (b,n,2)-->(b,1,2)
        mx = torch.matmul(mean_x.transpose(1,2), mean_x) # (b,2,2)
        vx = torch.einsum('bij,bik->bjk', x, x) # (b,2,2)
        num = float(x.size(1))
        cov_xx = 1. / num * (vx - (1. / num) * mx)
        return cov_xx  # (n,2,2)