import torch
from models.gtscnn import KCNetClassify, AdaptiveKCNetClassify, KCNetSegment, AdaptiveKCNetSegment
from data.modelnet import ModelNetDataset, AdaptiveModelNetDataset, MeshModelNetDataset
from data.shapenet import AdaptiveShapeNetDataset
from tensorboardX import SummaryWriter

# trainset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# testset = ModelNetDataset('/opt/modelnet40_normal_resampled', train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)

#######----------ModelNet 10-----------########
# trainset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet10_train_16nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# testset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet10_test_16nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
# net = KCNetClassify(trainset.class_nums)


# trainset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet10_train_16nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# testset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet10_test_16nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
# net = AdaptiveKCNetClassify(trainset.class_nums)


#######---------ModelNet 40------------######
# origin
# trainset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# testset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
# net = KCNetClassify(trainset.class_nums)

# Adaptive
# trainset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
# testset = AdaptiveModelNetDataset(('/home/zhu/Documents/paper_code/KCNet/'
# 	'modelnet/data/modelNet40_test_16nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=2)
# net = AdaptiveKCNetClassify(trainset.class_nums)

# Mesh
trainset = MeshModelNetDataset(('data/modelnet/modelNet40_train_16nn_GM.npy'), train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
testset = MeshModelNetDataset(('data/modelnet/modelNet40_test_16nn_GM.npy'), train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=2)
net = AdaptiveKCNetClassify(trainset.class_nums)

#######---------shapeNet 50------------######
# origin
# trainset = AdaptiveShapeNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
# 	'shapenet/data/shapenet_train_18nn_GM_adaptive_knn_sparse.npy'), train=True)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = AdaptiveShapeNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
# 	'shapenet/data/shapenet_test_18nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)
# net = KCNetSegment(trainset.class_nums, trainset.category_nums)

# Adaptive
# trainset = AdaptiveShapeNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
# 	'shapenet/data/shapenet_train_18nn_GM_adaptive_knn_sparse.npy'), train=True)	
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# testset = AdaptiveShapeNetDataset(('/home/zhu/Documents/code/mytest/KCNet/'
# 	'shapenet/data/shapenet_test_18nn_GM_adaptive_knn_sparse.npy'), train=False)
# testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)
# net = AdaptiveKCNetSegment(trainset.class_nums, trainset.category_nums)

writer=SummaryWriter()
for epcho in range(1, 151):
    net.fit(trainloader, epcho, writer)
    if(epcho%20==0):
	    net.score(testloader)
net.score(testloader)
writer.close()
print("Done!!!")
