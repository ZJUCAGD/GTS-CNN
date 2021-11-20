import os, sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from utils.kcnet_utils import * #import LoaclGeometricStructure, batch_knn, graph_max_pooling
from models.gtscnn import GeoNet, parall_GeoNet, AdaptiveGeoNet, KCNetClassify#, AdaptiveKCNetClassify, KCNetSegment, AdaptiveKCNetSegment
from models.gtscnn import pretrained_knnPISphericalGeoNet,knnPISphericalGeoNet, knnPD1SphericalGeoNet, knnSphericalGeoNet,PD1Net,PINet
from data.modelnet import ModelNetDataset, AdaptiveModelNetDataset, MeshModelNetDataset
from data.modelnet import DataPrefetcher, PD1ModelNetDataset, PIModelNetDataset, MeshPD1ModelNetDataset, MeshPIModelNetDataset
from data.shapenet import AdaptiveShapeNetDataset
from tensorboardX import SummaryWriter    #zhuyijie
from torch.autograd import gradcheck
from utils.misc import Netpara
gpu_0=torch.device('cuda:0')

#######---------ModelNet 40------------######
# # origin
# trainset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
# testset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

# # # mesh
# trainset = MeshModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
#                                'modelnet40/modelNet40_train_16nn_GM.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = MeshModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
#                                'modelnet40/modelNet40_test_16nn_GM.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
# mesh+pd100/pi
trainset = MeshPIModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
							   'modelnet40/modelNet40_train_16nn_GM.npy'), train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
testset = MeshPIModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
							   'modelnet40/modelNet40_test_16nn_GM.npy'), train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)

# ## pd/pi
# trainset_pi = PIModelNetDataset(train=True)
# trainloader_pi = torch.utils.data.DataLoader(trainset_pi, batch_size=32, shuffle=True, num_workers=4)
# testset_pi = PIModelNetDataset(train=False)
# testloader_pi = torch.utils.data.DataLoader(testset_pi, batch_size=32, shuffle=True, num_workers=4)

# net = AdaptiveGeoNet(3,class_nums=trainset.class_nums)
# net = parall_GeoNet(3,class_nums=trainset.class_nums)
# net = knnSphericalGeoNet(3,class_nums=trainset.class_nums)
# net = knnPD1SphericalGeoNet(3,class_nums=trainset.class_nums)
net = pretrained_knnPISphericalGeoNet(3,class_nums=trainset.class_nums)
# net = PINet(3,class_nums=trainset.class_nums)
# Netpara(net)

# writer=SummaryWriter() #zhuyijie
writer=None
load_weight=True
train_sperate=True


if load_weight:
	print("start load pretrained model!")
	net.load_state_dict(torch.load('./model_param/knnPISphericalGeoNet_second_stage.ckpt'))
	print("end with loaded pretrained model!")
	print("scoring test data-----")
	# test_prefetcher=DataPrefetcher(testloader)
	net.score(testloader)
	## 3.third train all
	for epcho in range(1, 61): 
		# train_prefetcher=DataPrefetcher(trainloader)
		net.fit(trainloader, epcho, writer)
		if(epcho%10==0):
			# test_prefetcher=DataPrefetcher(testloader)
			net.score(testloader)
	torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_adapt_stage_add_div6_6_10.ckpt')
else:
	if train_sperate:
		## 1.first train pi 
		for epcho in range(1, 51):
			net.fit_pi(trainloader_pi, epcho, writer)
			if(epcho%10==0):
				net.score_pi(testloader_pi)
		torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_first_stage_6_8.ckpt')
		## 2.second train kc
		for epcho in range(1, 101):
			net.fit_kc(trainloader, epcho, writer)
			if(epcho%10==0):
				net.score_kc(testloader)

		torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_second_stage_6_8.ckpt')
		## 3.third train all
		for epcho in range(1, 61): 
			net.fit(trainloader, epcho, writer)
			if(epcho%10==0):
				net.score(testloader)

		torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_third_stage_6_8.ckpt')
	else:
		for epcho in range(1, 101): #400
			net.fit(trainloader, epcho, writer)
			if(epcho%10==0):
				net.score(testloader)
# net.score(testloader)
if writer is not None:
	writer.close()
##zhuyijie
## Save and load the entire model.
# torch.save(net, 'adaptive_train_cls_model_4_18.ckpt')
# torch.save(net, 'shapenet_orig_train_cls_model_4_25.ckpt')
# torch.save(net, 'meshpd100sphericalgeonet_2geoconv_GN_2spherical_FC2_32_128_relu_ALLGN_classify1024400_256_modelnet40_train_cls_model_6_1.ckpt')

## model = torch.load('model.ckpt')
# # Save and load only the model parameters (recommended).
# torch.save(net.state_dict(), 'adaptive_train_cls_params_4_18.ckpt')
# # resnet.load_state_dict(torch.load('params.ckpt'))
print("Done!!!")





#############

import os, sys
import time
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# print(BASE_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
# from utils.kcnet_utils import * #import LoaclGeometricStructure, batch_knn, graph_max_pooling
from models.kcnet import GeoNet, TestKNNGeoNet, TestBallSplitGeoNet, SphericalGeoNet, parall_GeoNet, AdaptiveGeoNet#, KCNetClassify, AdaptiveKCNetClassify, KCNetSegment, AdaptiveKCNetSegment
from models.kcnet import first_two_knnPISphericalGeoNet,pretrained_knnPISphericalGeoNet,knnPISphericalGeoNet, knnPD1SphericalGeoNet, knnSphericalGeoNet,PD1Net,PINet
from data.modelnet import ModelNetDataset, AdaptiveModelNetDataset, MeshModelNetDataset
from data.modelnet import DataPrefetcher, TestMeshModelNetDataset, TestKNNModelNetDataset, PIModelNetDataset, MeshPD1ModelNetDataset, MeshPIModelNetDataset#PD1ModelNetDataset,
from data.shapenet import AdaptiveShapeNetDataset
from tensorboardX import SummaryWriter    #zhuyijie
from torch.autograd import gradcheck
from utils.kcnet_utils import Netpara, debugPrint, setup_seed, worker_init_fn
device_id=1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
torch.cuda.set_device(device_id)

#######---------ModelNet 40------------######
# # origin
# trainset = AdaptiveModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy', train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
# testset = AdaptiveModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)

# trainset = TestKNNModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy',
# 							['../modelnet/data/modelNet40_train_16nn_GM_adaptive_32knn_sparse_1.npy',
# 							'../modelnet/data/modelNet40_train_16nn_GM_adaptive_32knn_sparse_2.npy',
# 							'../modelnet/data/modelNet40_train_16nn_GM_adaptive_32knn_sparse_3.npy',
# 							'../modelnet/data/modelNet40_train_16nn_GM_adaptive_32knn_sparse_4.npy'],
# 								train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
# testset = TestKNNModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', 
# 							'../modelnet/data/modelNet40_test_16nn_GM_adaptive_32knn_sparse.npy',
# 								train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
##11.1
trainset = TestMeshModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM.npy',train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
testset = TestMeshModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM.npy', train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)


# # # mesh
# trainset = MeshModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
#                                'modelnet40/modelNet40_train_16nn_GM.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = MeshModelNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
#                                'modelnet40/modelNet40_test_16nn_GM.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4)
# # mesh+pd100/pi #10.12
# trainset = MeshPIModelNetDataset('../modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy', train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
# testset = MeshPIModelNetDataset('../modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)

# # ## pd/pi
trainset_pi = PIModelNetDataset('../modelnet/data/PD_pi/pi_00005_logistic_50_train.npy',train=True)
trainloader_pi = torch.utils.data.DataLoader(trainset_pi, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
testset_pi = PIModelNetDataset('../modelnet/data/PD_pi/pi_00005_logistic_50_test.npy',train=False)
testloader_pi = torch.utils.data.DataLoader(testset_pi, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
setup_seed(1024)

# net = GeoNet(3,class_nums=trainset.class_nums,device_id=device_id)
# net = TestKNNGeoNet(3,class_nums=trainset.class_nums,device_id=device_id) # 11.1
# net = SphericalGeoNet(3,class_nums=trainset.class_nums,device_id=device_id) # 10.16
# net = TestBallSplitGeoNet(3,class_nums=trainset.class_nums,device_id=device_id) # 10.16
# net = AdaptiveGeoNet(3,class_nums=trainset.class_nums)
# net = parall_GeoNet(3,class_nums=trainset.class_nums)
# net = knnSphericalGeoNet(3,class_nums=trainset.class_nums)
# net = knnPD1SphericalGeoNet(3,class_nums=trainset.class_nums)
net = pretrained_knnPISphericalGeoNet(3,class_nums=trainset.class_nums,device_id=device_id) #10.12
# net = first_two_knnPISphericalGeoNet(3,class_nums=trainset.class_nums,device_id=device_id) #11.14
# net = PINet(3,class_nums=trainset.class_nums)
Netpara(net)
print("begining............................")
# writer=SummaryWriter()
writer=None
load_weight=False
train_sperate=True

tic=time.time()
if load_weight:
	print("start load pretrained model!")
	net.load_state_dict(torch.load('./model_param/knnPISphericalGeoNet_second_stage.ckpt'))
	print("end with loaded pretrained model!")
	print("scoring test data-----")
	test_prefetcher=DataPrefetcher(testloader)
	net.score(test_prefetcher)
	## 3.third train all
	for epcho in range(1, 61): 
		train_prefetcher=DataPrefetcher(trainloader)
		net.fit(train_prefetcher, epcho, writer)
		if(epcho%10==0):
			test_prefetcher=DataPrefetcher(testloader)
			net.score(test_prefetcher)
	torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_adapt_stage_add_div6_relu_6_10.ckpt')
else:
	if train_sperate:
		## train first branch
		for epcho in range(1, 101):
			net.fit_kc(trainloader, epcho, writer)
			# if(epcho%10==0):
			net.score_kc(testloader)
		# ## 1.first train pi 
		# for epcho in range(1, 51):
		# 	net.fit_pi(trainloader_pi, epcho, writer)
		# 	if(epcho%10==0):
		# 		net.score_pi(testloader_pi)
		# # torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_first_stage_6_8.ckpt')
		# ## 2.second train kc
		# for epcho in range(1, 101):
		# 	net.fit_kc(trainloader, epcho, writer)
		# 	if(epcho%10==0):
		# 		net.score_kc(testloader)
		# # torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_second_stage_concate_1120.ckpt')
		# ## 3.third train all
		# for epcho in range(1, 61): 
		# 	net.fit(trainloader, epcho, writer)
		# 	if(epcho%10==0):
		# 		net.score(testloader,is_save=True)

		# torch.save(net.state_dict(), './model_param/knnPISphericalGeoNet_third_stage_concate_1120.ckpt')
	else:
		for epcho in range(1, 101): #400
			if((epcho-1)%10==0):
				net.score_kc(testloader)
			net.fit_kc(trainloader, epcho, writer)
		net.score_kc(testloader)
			
# net.score(testloader)
if writer is not None:
	writer.close()
toc=time.time()
print("%.3f ms has passed"% ((toc-tic)*1000))
print("Done!!!")
