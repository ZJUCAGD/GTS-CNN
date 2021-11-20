import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from .kdtree import make_cKDTree

from utils.kcnet_utils import Netpara, debugPrint

# global writer
global_step=0
        
       
class KDNet(nn.Module):
    def __init__(self, num_classes = 40, input_channels=3, depth=11, device_id=0, initial_weights=True):
        super(KDNet, self).__init__()
        self.device_id = device_id
        self.DEPTH=depth
        self.num_classes=num_classes
        ## max_pooling
        # self.conv1 = nn.Conv1d(3, 8 * 3, 1, 1)
        # self.conv2 = nn.Conv1d(8, 32 * 3, 1, 1)
        # self.conv3 = nn.Conv1d(32, 64 * 3, 1, 1)
        # self.conv4 = nn.Conv1d(64, 64 * 3, 1, 1)
        # self.conv5 = nn.Conv1d(64, 64 * 3, 1, 1)
        # self.conv6 = nn.Conv1d(64, 128 * 3, 1, 1)
        # self.conv7 = nn.Conv1d(128, 256 * 3, 1, 1)
        # self.conv8 = nn.Conv1d(256, 512 * 3, 1, 1)
        # self.conv9 = nn.Conv1d(512, 512 * 3, 1, 1)
        # self.conv10 = nn.Conv1d(512, 512 * 3, 1, 1)
        # self.conv11 = nn.Conv1d(512, 1024 * 3, 1, 1)
        # self.fc = nn.Linear(1024, num_classes)

        # 3--8--32--64--64--64--128--256--512--512--512--1024
        # 1024-512-256-128-64-32-16-8-4-2

        # self.conv1 = nn.Conv1d(3, 8 * 3, 1, 1)
        # self.conv2 = nn.Conv1d(8*2, 32 * 3, 1, 1)
        # self.conv3 = nn.Conv1d(32*2, 64 * 3, 1, 1)
        # self.conv4 = nn.Conv1d(64*2, 64 * 3, 1, 1)
        # self.conv5 = nn.Conv1d(64*2, 64 * 3, 1, 1)
        # self.conv6 = nn.Conv1d(64*2, 128 * 3, 1, 1)
        # self.conv7 = nn.Conv1d(128*2, 256 * 3, 1, 1)
        # self.conv8 = nn.Conv1d(256*2, 512 * 3, 1, 1)
        # self.conv9 = nn.Conv1d(512*2, 512 * 3, 1, 1)
        # self.conv10 = nn.Conv1d(512*2, 512 * 3, 1, 1)
        # self.conv11 = nn.Conv1d(512*2, 1024 * 3, 1, 1)

        # 3--32--64--64--128--128--256--256--512--512--128--num_classes
        #self.fc1 = nn.Linear(3, 32)
        self.conv1 = ConvBlock(3, 32 * 3, 1, 1)
        self.conv2 = ConvBlock(32*2, 64 * 3, 1, 1)
        self.conv3 = ConvBlock(64*2, 64 * 3, 1, 1)
        self.conv4 = ConvBlock(64*2, 128 * 3, 1, 1)
        self.conv5 = ConvBlock(128*2, 128 * 3, 1, 1)
        self.conv6 = ConvBlock(128*2, 256 * 3, 1, 1)
        self.conv7 = ConvBlock(256*2, 256 * 3, 1, 1)
        self.conv8 = ConvBlock(256*2, 512 * 3, 1, 1)
        self.conv9 = ConvBlock(512*2, 512 * 3, 1, 1)
        self.conv10 = ConvBlock(512*2, 128 * 3, 1, 1)
        # self.conv11 = nn.Conv1d(128*2, self.num_classes * 3, 1, 1)
        self.fc = nn.Linear(128*2, num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        self.cuda(self.device_id)

    def forward(self, x, c):
        def kdconv(x, dim, featdim, sel, conv):
            """
            inputs:
                x: (B, C, N)
                dim: int, num of point of input
                featdim: int, num of channel in one of three dir of conv output
                sel: cutdim_v at a depth, (B, 2^(MAX_DEPTH-depth))
                conv: conv op, input_channel=C, output_channel=featdim*3
            output:
                x: (B, featdim*2, N/2)
            """
            x = conv(x) # (b,featdim*3, dim)
            batchsize=x.size(0)
            x = x.view(-1, featdim, 3 * dim)
            x = x.transpose(0,1).contiguous().view(featdim, 3 * dim * batchsize)
            tmp=torch.arange(0, dim) * 3
            # debugPrint(sel.size())
            sel = Variable(sel + tmp.long()).view(-1,1)
            offset = Variable((torch.arange(0,batchsize) * dim * 3).repeat(dim,1).transpose(1,0).contiguous().long().view(-1,1))
            sel = sel+offset

            if x.is_cuda:
                sel = sel.cuda()
            # sel = sel.squeeze()
            # debugPrint(sel.size())
            x = torch.index_select(x, dim=1, index=sel.squeeze())

            x = x.view(featdim, batchsize, dim)
            x = x.transpose(1,0).transpose(2,1).contiguous()
            x = x.view(-1, dim//2, featdim * 2).transpose(2,1).contiguous()

            # x = x.view(-1, featdim, dim / 2, 2)
            # x = torch.squeeze(torch.max(x, dim=-1, keepdim=True)[0], 3)
            # debugPrint(x.size())
            return x
        # debugPrint(x.size())
        # debugPrint(len(c))
        # debugPrint(c[0].size())
        # debugPrint(c[1].size())
        # debugPrint(c[-1].size())

        # x1 = kdconv(x, 1024, 8, c[-1], self.conv1)
        # x2 = kdconv(x1, 1024, 32, c[-2], self.conv2)
        # x3 = kdconv(x2, 512, 64, c[-3], self.conv3)
        # x4 = kdconv(x3, 256, 64, c[-4], self.conv4)
        # x5 = kdconv(x4, 128, 64, c[-5], self.conv5)
        # x6 = kdconv(x5, 64, 128, c[-6], self.conv6)
        # x7 = kdconv(x6, 32, 256, c[-7], self.conv7)
        # x8 = kdconv(x7, 16, 512, c[-8], self.conv8)
        # x9 = kdconv(x8, 8, 512, c[-9], self.conv9)
        # x10 = kdconv(x9, 4, 512, c[-10], self.conv10)
        # x11 = kdconv(x10, 2, 1024, c[-11], self.conv11)
        # x11 = x11.view(-1, 1024)

        # init_num_pts=x.size()[-1]
        #
        # x = torch.transpose(x,dim0=2,dim1=1).contiguous().view(-1,3) ##[N*1024,3]
        # x = F.relu(self.fc1(x)) ##[N*1024,32]
        # x = torch.transpose(x.view(-1,init_num_pts,32),dim0=2,dim1=1) ##[N,32,1024]



        x1 = kdconv(x, 1024, 32, c[0], self.conv1)
        # debugPrint(x1.size())
        # raise Exception("抛出一个异常")
        x2 = kdconv(x1, 512, 64, c[1], self.conv2)
        x3 = kdconv(x2, 256, 64, c[2], self.conv3)
        x4 = kdconv(x3, 128, 128, c[3], self.conv4)
        x5 = kdconv(x4, 64, 128, c[4], self.conv5)
        x6 = kdconv(x5, 32, 256, c[5], self.conv6)
        x7 = kdconv(x6, 16, 256, c[6], self.conv7)
        x8 = kdconv(x7, 8, 512, c[7], self.conv8)
        x9 = kdconv(x8, 4, 512, c[8], self.conv9)
        x10 = kdconv(x9, 2, 128, c[9], self.conv10)
        # debugPrint(x10.size())
        # x11 = kdconv(x10, 1, self.num_classes, c[10], self.conv11)
        # debugPrint(x10.size())
        x11 = x10.view(-1, 128*2)

        x = self.fc(x11)
        # out = F.log_softmax()
        return x

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
        for batch_idx, (points_v, cutdim_v, targets) in enumerate(dataloader):  #zhuyijie
            points_v = points_v.cuda(self.device_id) # (b,n,c)
            # cutdim_v = cutdim_v.cuda(self.device_id) list of depth array with (b,2)
            targets = targets.cuda(self.device_id)

            self.optimizer.zero_grad()

            outputs = self(points_v, cutdim_v)
            # outputs = self(inputs)
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
            for batch_idx, (points_v, cutdim_v, targets) in enumerate(dataloader):  #zhuyijie
                points_v = points_v.cuda(self.device_id)
                # cutdim_v = cutdim_v.cuda(self.device_id)
                targets = targets.cuda(self.device_id)

                # outputs = self(inputs)
                outputs = self(points_v, cutdim_v)

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



#https://github.com/wassname/kdnet.pytorch/blob/master/kdnet.py
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, dropout=0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU()
#         self.drp = nn.Dropout(dropout)
        
        # init
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.conv.weight, gain=gain)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
#         x = self.drp(x)
        return x

class KDNet_Batch(nn.Module):
    def __init__(self, depth=11, k = 16, input_levels=11, channels2=4, features=8, max_pool=False):
        """
        Uses a for loop, for simpler and slower logic.
        
        depth: Desired models depth, should be <=input_levesl
        input_levels: levels in the input kdtree
        k: output dimensions
        channels:  input channels
        features: base number of features for first convnet
        """
        super().__init__()
        self.channels2 = channels2
        self.depth = depth
        self.input_levels = input_levels
        self.max_pool = max_pool
        self.initial_leaf_dim = 2**(input_levels - depth)
        
        current_channels = self.channels2//2*self.initial_leaf_dim
        self.mult = 1 if self.max_pool else 2
        
        channels = (2**(np.arange(1,input_levels+1)//2)*features).tolist()
        print(channels)
        
        self.convs = torch.nn.ModuleList()
        for i in range(depth):
            out_channels = channels[i]
            self.convs.append(ConvBlock(current_channels * self.mult, out_channels * self.channels2,1,1))
            current_channels = out_channels

        hidden_dims = current_channels * self.mult
        print(hidden_dims)
        self.fc = nn.Linear(hidden_dims, k)
        
    def forward(self, x, cutdims):
        def kdconv(x, cutdim, block):
            # This version is just does each sample seperate, then joins the batch
            batchsize = x.size(0)
            channels = self.channels2
            batch, feat_dims, old_leaves = x.size()
            old_leaf_dims = feat_dims // channels
            
            # featdim = channels * points_in_leaf (since they are all unorder we group them as the non spatial dim)
            x = x.view(batchsize, channels * old_leaf_dims, old_leaves)
            x = block(x)
            leaf_dims = x.size(1)//channels
            leaves = x.size(-1)
            # It comes out as (-1, leaf_dims* channels, leaves) we will group the channels with the leaves then select the ones we want
            x = x.view(-1, leaf_dims, channels, leaves)
            x = x.view(batchsize, leaf_dims, channels * leaves)

            # Do each batch separately for now to avoid errors
            xs = []
            for i in range(batchsize):
                sel = Variable(cutdim[i] + (torch.arange(0, leaves) * channels).long())
                if x.is_cuda:
                    sel = sel.cuda()
                xi = torch.index_select(x[i], dim=-1, index=sel)
                
                # Reshape back to real dimensions
                xi = xi.view(leaf_dims, leaves)

                # Reduce amount of leaves for next level
                if self.max_pool:
                    xi = xi.view(leaf_dims, leaves // 2, 2)
                    xi = torch.squeeze(torch.max(xi, dim=-1, keepdim=True)[0], 3)
                else:
                    xi = xi.view(leaf_dims*2, leaves // 2)
                xs.append(xi)
            x = torch.stack(xs, 0)
            return x
        
        if len(x.shape)==4:
            # From (batch, channels, leaf_dim, leaves) to  (batch, channels * leaf_dim, leaves) 
            # We treat the channels and leaf_dim as one since they are both non spatial/non-ordered dimensions
            x = x.view(batch_size, -1, x.size(-1))
        
        
        # input shape should be (batch, channels, leaf_points, leaves)
        for i in range(self.depth):
            dim = 2**(self.depth-i)
            x = kdconv(x, cutdims[self.depth-i-1],  self.convs[i])
        
        x = x.view(-1, self.fc.in_features)
        out = self.fc(x)
        return out

    
class KDNet_Batch2(nn.Module):
    def __init__(self, depth=11, k = 16, input_levels=11, channels=4, features=None):
        """
        A slightly more complex version.
        
        depth: Desired models depth, should be <=input_levesl
        input_levels: levels in the input kdtree
        k: output dimensions
        channels:  input channels
        features: base number of features for first convnet
        """
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.input_levels = input_levels
        if features is None:
            features = 2**(input_levels-depth)
        
        current_channels = self.channels2//2
        self.convs = torch.nn.ModuleList()
        channels = (2**(np.arange(1,input_levels+1)//2)*features).tolist()
        print(channels)
        for i in range(depth):
            out_channels = channels[i]
            self.convs.append(ConvBlock(current_channels*2, out_channels * self.channels,1,1))
            current_channels = out_channels

        self.fc = nn.Linear(current_channels*2**(self.input_levels-self.depth+1), k)
        nn.init.constant_(self.fc.weight, 0.001)
        nn.init.constant_(self.fc.bias, 0)
        
    def forward(self, x, cutdims):
        def kdconv(x, dim, featdim, cutdim, block):
            x = block(x)
            batchsize = x.size(0)  
            
            # Reshape to (featuredim, -1) so we can select
            x = x.view(batchsize, featdim, self.channels2 * dim)
            x = x.transpose(1,0).contiguous()
            x = x.view(featdim, self.channels2 * dim * batchsize)

            # We want to select the cut dimension, but index_select can only take in a 1d array
            # so we have some reshaping to do.
            
            # Offset cutdim so we can use it to select on a flattened array
            cutdim_offset = (torch.arange(0, dim) * self.channels2).repeat(batchsize, 1).long()
            sel = Variable(cutdim + cutdim_offset) 
            sel = sel.view(-1, 1)
            # Work out offsets for cutdims
            offset = Variable((torch.arange(0, batchsize) * dim * self.channels2))
            offset = offset.repeat(dim, 1).long().transpose(1, 0).contiguous().view(-1,1)
            
            sel2 = sel+offset
            sel2 = sel2.squeeze()
            if x.is_cuda:
                sel2 = sel2.cuda()     

            x = torch.index_select(x, dim = 1, index = sel2)
            # Reshape back
            x = x.view(featdim, batchsize, dim)
            x = x.transpose(1,0)    # (batchsize, featdim, dim)
            x = x.transpose(2,1).contiguous()    # (batchsize, dim, featdim)
            
            # move some half of the dimensions to the features
            x = x.view(-1, dim//2, featdim * 2)  # (-1, dim//2, featdim*2)   
            x = x.transpose(2,1).contiguous()    # (-1, featdim*2, dim//2)   
            return x
        
        for i in range(self.depth):
            outdims = self.convs[i].conv.out_channels//self.channels2
            dim = 2**(self.input_levels-i)
            x = kdconv(x, dim, outdims, cutdims[-i-1],  self.convs[i])
        
        x = x.view(-1, outdims*2**(self.input_levels-self.depth+1))
        out = self.fc(x)
        return out