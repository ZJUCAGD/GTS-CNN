import os
import random
import numpy as np
import torch
import torch.nn as nn

def Netpara(net: nn.Module):
    '''
        detect the paramaters info of the pytorch net(torch.nn.Module type)
        zhuyijie @2019.5.19
    '''
    k=0
    for name, param in net.named_parameters():
      if param.requires_grad:
          print(name)
          l = 1
          print("该层的结构：" + str(list(param.size())))
          for j in param.size():
              l *= j
          print("该层参数和：" + str(l))
          k = k + l
    print("总参数数量和：" + str(k))

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id):
    """
    After creating the workers, each worker has an independent seed that
     is initialized to the current random sedd+the id of the worker
    :param worker_id
    :return
    """
    np.random.seed(1024+worker_id)

#############################################################
# print the running time fo each line inside of the function
# add decorate @profile to the tested function, then in the shell, run:
# [CUDA_LAUNCH_BLOCKING=1] kernprof  -lv  line_profiler_test.py
#############################################################
# with torch.autograd.profiler.profile(use_cuda=True) as prof:
#     y=x**2
#     y=y.mean()
#     y.backward()
# print(prof)
#############################################################

## Save and load the entire model.
# torch.save(net, 'cls_model.ckpt')
# model = torch.load('model.ckpt')

# # Save and load only the model parameters (recommended).
# torch.save(net.state_dict(), 'cls_model_params.ckpt')
# resnet.load_state_dict(torch.load('cls_params.ckpt'))


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def augment_fn(batch_xyz, batch_label, augment_ratio=0.5):
    bsize, num_point, _ = batch_xyz.shape

    # shuffle the orders of samples in a batch
    idx = np.arange(bsize)
    np.random.shuffle(idx)
    batch_xyz = batch_xyz[idx,:,:]
    batch_label = batch_label[idx]

    # shuffle the point orders of each sample
    batch_xyz = data_util.shuffle_points(batch_xyz)

    # perform augmentation on the first np.int32(augment_ratio*bsize) samples
    augSize = np.int32(augment_ratio * bsize)
    augment_xyz = batch_xyz[0:augSize, :, :]

    augment_xyz = data_util.rotate_point_cloud(augment_xyz)
    augment_xyz = data_util.rotate_perturbation_point_cloud(augment_xyz)
    augment_xyz = data_util.random_scale_point_cloud(augment_xyz)
    augment_xyz = data_util.shift_point_cloud(augment_xyz)

    batch_xyz[0:augSize, :, :] = augment_xyz

    return batch_xyz, batch_label