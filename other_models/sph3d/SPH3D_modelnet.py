import os, sys
import gc
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "utils"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils.kcnet_utils import Netpara, debugPrint, Fc
import sph3d_ops_utils
import sph3gcn_util as s3g_util

global_step = 0

def gather_nd(input_tensor, indices):
    """
    input_tensor: (b,n,c), float32
    indices: (b,m), int

    """
    batch_size = input_tensor.size(0)
    # indices=indices.long()
    return torch.stack([torch.index_select(input_tensor[k],0,indices[k]) for k in range(batch_size)]) # keep dim as xyz

def normalize_xyz(points):
    points -= points.mean(dim=1,keepdim=True)
    scale = torch.pow(points,2).sum(dim=-1,keepdim=True).max(dim=1,keepdim=True)[0]
    scale = torch.sqrt(scale)
    points /= scale

    return points


# def _separable_conv3d_block(net, list_channels, bin_size, nn_index, nn_count, filt_idx,
#                             name, depth_multiplier=None, weight_decay=None, reuse=None,
#                             with_bn=True, with_bias=True, is_training=None):
#     for l, num_out_channels in enumerate(list_channels):
#         scope = name + '_' + str(l+1) # number from 1, not 0
#         net = s3g_util.separable_conv3d(net, num_out_channels, bin_size,
#                                         depth_multiplier[l], scope, nn_index,
#                                         nn_count, filt_idx, weight_decay=weight_decay,
#                                         with_bn=with_bn, with_bias=with_bias,
#                                         reuse=reuse, is_training=is_training)
#     return net

class SPH3D(nn.Module):
    def __init__(self, input_channels, class_nums=1, config=None, device_id=0, initial_weights=True):
        super(SPH3D, self).__init__()
        self.input_channels = input_channels
        self.class_nums = class_nums
        self.device_id = device_id
        self.config = config
        self.global_radius = 100.0 # global_radius(>=2.0) should connect all points to each point in the cloud

        # self.FC1 = Fc(input_channels,[64,128,384],input_dim=4, bn='BN', activation_fn='relu')
        self.FC1= Fc(input_channels,[self.config.mlp],input_dim=3, bn='BN', activation_fn='relu') # out_c=32, b,n,in_c-->b,n,out_c
        
        self.SeparableConv3d_block = nn.ModuleList()
        c_in=32
        # if self.config.use_raw:
        #     c_in+=3
        for l in range(len(self.config.radius)):# [0.1, 0.2, 0.4]
            if self.config.use_raw:
                c_in+=3
            prefix_name = 'conv'+str(l+1)
            for k, c_out in enumerate(self.config.channels[l]): #[64,64]
                scope = prefix_name + '_' + str(k+1) # number from 1, not 0
                self.SeparableConv3d_block.append(s3g_util.SeparableConv3d(c_in,c_out,
                                                    self.config.binSize, self.config.multiplier[l][k], scope)
                                                 )
                c_in = c_out

        self.SeparableConv3d=s3g_util.SeparableConv3d(c_in, self.config.global_channels, 17, config.global_multiplier, #global_c=512,global_multiplier=2
                                                    'global_conv')

        self.classify = nn.Sequential(
                            # nn.Linear(2048, 512,bias=False),
                            nn.Linear(64+128+128+512, 512, bias=False),
                            nn.BatchNorm1d(512),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            nn.Linear(512, 256,bias=False),
                            nn.BatchNorm1d(256),
                            nn.ReLU(True),
                            nn.Dropout(0.5),
                            nn.Linear(256, class_nums)
                        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), weight_decay=1e-5)
        self.schedule = optim.lr_scheduler.StepLR(self.optimizer, 10, 0.6)
        
        # if initial_weights:
        #     self.initialize_weights()
        self.cuda(device_id)

    def forward(self, point_cloud):  #B,C,N
        point_cloud=point_cloud.transpose(1,2).contiguous()
        batch_size,num_point,c=point_cloud.size()
        if self.config.normalize:
            point_cloud = normalize_xyz(point_cloud)
        xyz=point_cloud
        query = xyz.mean(dim=1, keepdim=True)  # the global viewing point, b,1,c

        net = self.FC1(xyz) # (b,n,32)
        global_feat=[]
        index=0
        for l in range(len(self.config.radius)): 
            if self.config.use_raw:
                net=torch.cat([net,xyz],2) #---->(b,n,32+3)
            # the neighbor information is the same within xyz_pose_1 and xyz_pose_2.
            # Therefore, we compute it with xyz_pose_1, and apply it to xyz_pose_2 as well
            intra_idx, intra_cnt, intra_dst, indices = s3g_util.build_graph(xyz, self.config.radius[l], self.config.nn_uplimit[l],
                                                      self.config.num_sample[l], sample_method=self.config.sample)
            
            filt_idx = sph3d_ops_utils.spherical_kernel(xyz, xyz, intra_idx, intra_cnt, intra_dst,
                                                  self.config.radius[l], self.config.kernel)
            for _ in self.config.channels[l]: #[64,64]
                net = self.SeparableConv3d_block[index](net, intra_idx, intra_cnt,filt_idx)
                index+=1

            if self.config.num_sample[l]>1:
                indices=indices.long()
                # ==================================gather_nd====================================
                xyz = gather_nd(xyz, indices)
                intra_idx = gather_nd(intra_idx, indices)
                intra_cnt = gather_nd(intra_cnt, indices)
                intra_dst = gather_nd(intra_dst, indices)
                # =====================================END=======================================

                net = s3g_util.pool3d(net, intra_idx, intra_cnt,
                                      method=self.config.pool_method, # max
                                      scope='pool'+str(l+1)) # (b,m,c_out)

            global_maxpool = net.max(dim=1, keepdim=True)[0] # (b,1,c_out)
            global_feat.append(global_maxpool)
            
        # =============================global feature extraction in the final layer==================
        nn_idx, nn_cnt, nn_dst = s3g_util.build_global_graph(xyz, query, self.global_radius)
        filt_idx = sph3d_ops_utils.spherical_kernel(xyz, query, nn_idx, nn_cnt, nn_dst,
                                             self.global_radius, [8,2,1])
        
        net = self.SeparableConv3d(net,nn_idx, nn_cnt, filt_idx) # (b,1,c_out)
                                       
        global_feat.append(net)
        y = torch.cat(global_feat,dim=2).view(batch_size,-1)
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
                if writer is not None:
                    writer.add_scalar('scalar/batch_loss_every8',batch_loss / 8, global_step)
                batch_loss = 0.

            gc.collect()
            torch.cuda.empty_cache()
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
                gc.collect()
                torch.cuda.empty_cache()
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
                if m.bias is not None:
                    m.bias.data.zero_()
