#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import importlib
import time
import random
import numpy as np
from pyhocon import ConfigFactory
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from data.modelnet import AdaptiveModelNetDataset
from utils import pointcloud_utils as put
from utils.misc import Netpara, debugPrint, setup_seed, worker_init_fn

device_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
torch.cuda.set_device(device_id)

#######---------ModelNet 40------------######
trainset = AdaptiveModelNetDataset(
    'modelnet/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy', train=True)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=32,
                                          shuffle=True,
                                          num_workers=4,
                                          worker_init_fn=worker_init_fn)
testset = AdaptiveModelNetDataset(
    'modelnet/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', train=False)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=32,
                                         shuffle=True,
                                         num_workers=4,
                                         worker_init_fn=worker_init_fn)
setup_seed(1024)

net_name = 'pointnet_cls'

if net_name == 'pointnet_cls':
    from other_models.pointnet.POINTNET import PointNetCls
    net = PointNetCls(k=40, feature_transform=False, device_id=device_id)
elif net_name == '':
    from other_models.pointnet2.PointNet2_MsgCls import Pointnet2MSG
    net = Pointnet2MSG(input_channels=0,
                       num_classes=40,
                       use_xyz=True,
                       device_id=device_id)
elif net_name == 'sonet':
    from other_models.sonet.classifier import Model
    from other_models.sonet.options import Options
    from data.modelnet import ModelNet_Shrec_Loader
    opt = Options().parse()
    # opt.surface_normal = False
    trainset = ModelNet_Shrec_Loader(opt.dataroot, 'train', opt)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opt.batch_size,
                                              shuffle=True,
                                              num_workers=opt.nThreads,
                                              worker_init_fn=worker_init_fn)

    testset = ModelNet_Shrec_Loader(opt.dataroot, 'test', opt)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=opt.batch_size,
                                             shuffle=True,
                                             num_workers=opt.nThreads,
                                             worker_init_fn=worker_init_fn)
    net = Model(opt)
    Netpara(net.encoder)
    Netpara(net.classifier)
elif net_name == 'sph3d':
    from other_models.sph3d.SPH3D_modelnet import SPH3D
    net_config = importlib.import_module('models.sph3d.modelnet_config')
    net = SPH3D(input_channels=3,
                class_nums=40,
                config=net_config,
                device_id=device_id)
elif net_name == 'ECC':
    import functools
    from data.modelnet import ECC_ModelNetDataset
    import other_models.ecc_model.ecc as ecc
    from other_models.ecc_model.ECC import ECC
    import logging
    '''
    CUDA_VISIBLE_DEVICES=0 python main.py \
    --dataset modelnet40 --test_nth_epoch 25 --lr 0.1 --lr_steps '[30,60,90]' --epochs 100 --batch_size 64 --batch_parts 4 \
    --model_config 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40'  \
    --fnet_llbias 0 --fnet_widths '[16,32]' --pc_augm_scale 1.2 --pc_augm_mirror_prob 0.2 --pc_augm_input_dropout 0.1 \
    --nworkers 3 --edgecompaction 1 --edge_mem_limit 1000 --odir results/modelnet40  
    '''
    model_config = 'i_1_2, c_24,b,r, c_48,b,r, m_2.5_7.5, c_48,b,r, c_48,b,r, m_7.5_22.5, c_96,b,r, m_1e10_1e10, f_64,b,r,d_0.2,f_40'
    net = ECC(model_config, 1, [(3) + (3)] + [16, 32], 1, 1, 1000, device_id=0)

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
        columns.extend(
            [p1[:, np.newaxis], p2[:, np.newaxis], p3[:, np.newaxis]])
        edgefeats = np.concatenate(columns, axis=1).astype(np.float32)
        edgefeats_clust, indices = ecc.unique_rows(edgefeats)
        logging.debug('Edge features: %d -> %d unique edges, %d dims',
                      edgefeats.shape[0], edgefeats_clust.shape[0],
                      edgefeats_clust.shape[1])
        return torch.from_numpy(edgefeats_clust), torch.from_numpy(indices)

    edge_feat_func = cloud_edge_feats
    collate_func = functools.partial(
        ecc.graph_info_collate_classification,
        edge_func=functools.partial(edge_feat_func))

    trainset = ECC_ModelNetDataset(
        'modelnet/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy',
        pyramid_conf=net.pyramid_conf,
        train=True)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              collate_fn=collate_func,
                                              batch_size=4,
                                              shuffle=True,
                                              num_workers=4,
                                              worker_init_fn=worker_init_fn)
    testset = ECC_ModelNetDataset(
        'modelnet/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy',
        pyramid_conf=net.pyramid_conf,
        train=False)
    testloader = torch.utils.data.DataLoader(testset,
                                             collate_fn=collate_func,
                                             batch_size=4,
                                             shuffle=True,
                                             num_workers=4,
                                             worker_init_fn=worker_init_fn)

elif net_name == 'pointConv':
    from other_models.pointconv.POINTCONV import PointConvDensityClsSsg as PointConvClsSsg
    net = PointConvClsSsg(input_channels=3,
                          num_classes=40,
                          device_id=device_id)
elif net_name == 'pointCNN':
    from other_models.pointCNN.pointCNN import Classifier
    net = Classifier(device_id=device_id)
elif net_name == 'pcnn':
    from other_models.pcnn.pcnn import PCNN
    from apex import amp
    conf = ConfigFactory.parse_file(
        'other_models/pcnn/confs/var_lesspoints.conf')
    net = PCNN(conf=conf.get_config('network'),
               input_channels=3,
               class_nums=40,
               device_id=device_id)  #trainset.class_nums
    net, net.optimizer = amp.initialize(net, net.optimizer,
                                        opt_level="O1")  # O is not number zero
    # with amp.scale_loss(loss, optimizer) as scaled_loss:
    # 	scaled_loss.backward()
elif net_name == 'rscnn':
    from other_models.rscnn.RSCNN import RSCNN_SSN
    net = RSCNN_SSN(num_classes=40,
                    input_channels=0,
                    relation_prior=1,
                    use_xyz=True)
elif net_name == 'dgcnn':
    from other_models.dgcnn.DGCNN import DGCNN
    net = DGCNN(output_channels = 40, device_id=device_id)
elif net_name == 'kdnet':
    from other_models.kdnet.kdnet import KDNet
    from data.modelnet import KDNet_ModelNetDataset
    depth=10
    trainset = KDNet_ModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy', depth=depth, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
    testset = KDNet_ModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', depth=depth, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
    setup_seed(1024)
    net = KDNet(input_channels=3, num_classes=40, depth=depth, device_id=device_id)
else:
    raise ValueError("not a implemented point net!")
    
Netpara(net)
# writer=SummaryWriter()
writer = None
load_weight = False
train_sperate = False
tic = time.time()

for epcho in range(1, 101):
    net.fit(trainloader, epcho, writer)
    if (epcho % 10 == 0):
        net.score(testloader)

# net.score(testloader)
if writer is not None:
    writer.close()
toc = time.time()
print("%.3f ms has passed" % ((toc - tic) * 1000))
print("Done!!!")