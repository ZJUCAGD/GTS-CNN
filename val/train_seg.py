import os, sys
import time
import torch
import numpy as np
# from utils.kcnet_utils import * #import LoaclGeometricStructure, batch_knn, graph_max_pooling
from models.gtscnn import TestKNNPISphereGeoNetSegment_fps,TestKNNPIGeoNetSegment_fps,TestKNNPIGeoNetSegment,GeoNetSegment,TestGeoNetSegment, KCNetSegment, TestKNNGeoNetSegment,AdaptiveKCNetSegment
# from models.kcnet import pretrained_knnPISphericalGeoNet,knnPISphericalGeoNet, knnPD1SphericalGeoNet, knnSphericalGeoNet,PD1Net,PINet
from data.shapenet import AdaptiveShapeNetDataset,KNNPIShapeNetDataset
from data.shapenet import DataPrefetcher
from tensorboardX import SummaryWriter
from utils.misc import Netpara, debugPrint, setup_seed, worker_init_fn

device_id=1
os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
torch.cuda.set_device(device_id)

# Adaptive
# trainset = AdaptiveShapeNetDataset('../shapenet/data/shapenet_train_18nn_GM_adaptive_knn_sparse.npy', train=True)	
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
# testset = AdaptiveShapeNetDataset('../shapenet/data/shapenet_test_18nn_GM_adaptive_knn_sparse.npy', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)

# mesh+pi @11.3
trainset = KNNPIShapeNetDataset('shapenet/shapenet_train_18nn_GM_adaptive_knn_sparse.npy', 
								'shapenet/PD_pi/pi_00005_logistic_50_2048_a10b05_nospecs_train.npy', 
								train=True)	
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)
testset = KNNPIShapeNetDataset('shapenet/shapenet_test_18nn_GM_adaptive_knn_sparse.npy', 
								'shapenet/PD_pi/pi_00005_logistic_50_2048_a10b05_nospecs_test.npy', 
								train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=4,worker_init_fn=worker_init_fn)

# net = GeoNet(3,class_nums=trainset.class_nums,device_id=device_id)
# net = GeoNetSegment(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id)
# net = TestGeoNetSegment(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id)
# net=KCNetSegment(class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id)
# net = TestKNNGeoNetSegment(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id)
# net = TestKNNPIGeoNetSegment(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id) #11.3
# net = TestKNNPIGeoNetSegment_fps(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id) #11.4
net = TestKNNPISphereGeoNetSegment_fps(3,class_nums=trainset.class_nums, category_nums=trainset.category_nums,device_id=device_id) #11.4
Netpara(net)
writer=SummaryWriter()
# writer=False
load_weight=False
train_sperate=False

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
			if((epcho-1)%10==0):
				net.score(testloader,is_save=True)
			net.fit(trainloader, epcho, writer)
			# if(epcho%10==0):
			# 	net.score(testloader)
		net.score(testloader,is_save=True)
if writer is not None:
	writer.close()
toc=time.time()
print("%.3f ms has passed"% ((toc-tic)*1000))
print("Done!!!")