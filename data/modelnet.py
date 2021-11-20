import math
import os
import copy
import signal
import multiprocessing as mp
import numpy as np
import math
import random
import scipy.sparse as sp
import torch
import torch.utils.data as data
from utils.augmentation import *


def multiprocess_fn(pool, fn, input_list, opts=[]):
    """ multiprocessing util tool
    """
    results = [
        pool.apply_async(fn, args=(x, ) + tuple(opts)) for x in input_list
    ]
    results = [p.get() for p in results]
    return results


class ModelNetDataset(data.Dataset):
    def __init__(self, root, point_nums=2048, train=True, argumentation=True):
        self.root = root
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        file = open(os.path.join(root, 'modelnet10_shape_names.txt'), 'r')
        self.shape_list = [str.rstrip() for str in file.readlines()
                           ]  # [bathtub bed chair desk
        # dresser monitor night_stand sofa table toilet]
        file.close()
        self.class_nums = len(self.shape_list)

        if train:
            file = open(os.path.join(root, 'modelnet10_train.txt'), 'r')
        else:
            file = open(os.path.join(root, 'modelnet10_test.txt'), 'r')
        for line in file.readlines():
            line = line.rstrip()
            name = line[0:-5]
            label = self.shape_list.index(name)
            self.dataset.append((os.path.join(os.path.join(root, name),
                                              line + '.txt'), label))
        file.close()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_path, label = self.dataset[index]
        data = np.loadtxt(file_path,
                          dtype=np.float32,
                          delimiter=',',
                          usecols=(0, 1, 2))
        data = data[
            np.random.choice(data.shape[0], self.point_nums, replace=False), :]

        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(np.float32))

        return pc, label


class AdaptiveModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        totalData = np.load(
            self.datafile, allow_pickle=True).tolist()  # array(dict) ---> dict
        # self.class_nums = len(totalData['label_dict'])
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        #dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        for i in range(b):
            self.dataset.append((totalData['data'][i], totalData['graph'][i],
                                 totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]
        data, graph, label = self.dataset[index]
        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(
            np.float32))  # --->(3, point_nums)
        graph = graph.tocoo()
        rows = graph.row
        cols = graph.col
        graph = [[row] for row in range(self.point_nums)]
        for i, row in enumerate(rows):
            graph[row].append(cols[i])
        graph = [
            row_graph + [row] * (17 - len(row_graph))
            for row, row_graph in enumerate(graph)
        ]
        graph = torch.from_numpy(np.array(graph))  # (point_nums, 17)
        label = torch.tensor(label).long()
        return pc, graph, label


class MeshModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        totalData = np.load(
            self.datafile, allow_pickle=True,
            encoding='latin1').tolist()  # array(dict) -> dict
        # self.class_nums = len(totalData['label_dict'])
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        print(totalData['data'].shape)
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        #dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        if ('train_' in self.datafile):
            mesh_graph_file = os.env('KCNET_ROOT') + 'modelnet40/train_graph_csr.npy'
        else:
            mesh_graph_file = os.env('KCNET_ROOT') + 'modelnet40/test_graph_csr.npy'
        graphs = np.load(mesh_graph_file, allow_pickle=True)
        for i in range(b):
            self.dataset.append(
                (totalData['data'][i], graphs[i], totalData['graph'][i]['M'],
                 totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]
        data, graph, knn_graph, label = self.dataset[index]
        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(np.float32))  # ->(3, point_nums)
        graph = graph.tocoo()
        rows = graph.row
        cols = graph.col
        graph = [[row] for row in range(self.point_nums)]
        for i, row in enumerate(rows):
            graph[row].append(cols[i])
        vertex_2ring = copy.deepcopy(graph)
        for i in range(self.point_nums):
            for source in graph[i][1:]:
                for target in graph[source][1:]:
                    if (target not in vertex_2ring[i]):
                        vertex_2ring[i].append(target)
        for row in range(self.point_nums):  # for vertex_2ring
            res = 17 - len(vertex_2ring[row])
            if (res <= 0):
                vertex_2ring[row] = vertex_2ring[row][:17]
            else:
                i = 0
                for node in knn_graph.getrow(row).nonzero()[1]:
                    if (node not in vertex_2ring[row]):
                        vertex_2ring[row].append(node)
                        i += 1
                    if (i >= res):
                        break

        # graph = [row_graph + [row]*(17-len(row_graph))  for row, row_graph in enumerate(graph)]
        vertex_2ring = torch.from_numpy(
            np.array(vertex_2ring))  # (point_nums, 17)
        label = torch.tensor(label).long()
        return pc, vertex_2ring, label


# 2019.5.30
class MeshPD1ModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []
        totalData = np.load(
            self.datafile, allow_pickle=True,
            encoding='latin1').tolist()  # array(dict) ---> dict
        # self.class_nums = len(totalData['label_dict'])
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        print(totalData['data'].shape)
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        #dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        if ('train_' in self.datafile):
            mesh_graph_file = os.env(
                'KCNET_ROOT') + 'modelnet40/train_graph_csr.npy'
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_train.npy'
        else:
            mesh_graph_file = os.env(
                'KCNET_ROOT') + 'modelnet40/test_graph_csr.npy'
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_test.npy'

        graphs = np.load(mesh_graph_file, allow_pickle=True)
        pd1s = np.load(pd1_file, allow_pickle=True)
        for i in range(b):
            self.dataset.append(
                (totalData['data'][i], graphs[i], totalData['graph'][i]['M'],
                 pd1s[i], totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]
        data, graph, knn_graph, pb1, label = self.dataset[index]
        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(
            np.float32))  # --->(3, point_nums)
        graph = graph.tocoo()
        rows = graph.row
        cols = graph.col
        graph = [[row] for row in range(self.point_nums)]
        for i, row in enumerate(rows):
            graph[row].append(cols[i])
        vertex_2ring = copy.deepcopy(graph)
        for i in range(self.point_nums):
            for source in graph[i][1:]:
                for target in graph[source][1:]:
                    if (target not in vertex_2ring[i]):
                        vertex_2ring[i].append(target)
        for row in range(self.point_nums):  # for vertex_2ring
            res = 17 - len(vertex_2ring[row])
            if (res <= 0):
                vertex_2ring[row] = vertex_2ring[row][:17]
            else:
                i = 0
                for node in knn_graph.getrow(row).nonzero()[1]:
                    if (node not in vertex_2ring[row]):
                        vertex_2ring[row].append(node)
                        i += 1
                    if (i >= res):
                        break

        # graph = [row_graph + [row]*(17-len(row_graph))  for row, row_graph in enumerate(graph)]
        vertex_2ring = torch.from_numpy(
            np.array(vertex_2ring))  # (point_nums, 17)
        label = torch.tensor(label).long()
        return pc, vertex_2ring, pb1[:100], label


class PD1ModelNetDataset(data.Dataset):
    def __init__(self, point_nums=1024, train=True, argumentation=True):
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.class_nums = 40
        self.dataset = []
        if self.train:
            # pd1_file='/home/zhu/Documents/code/mytest/KCNet/modelnet40/PB400_train.npy'
            pd_file = os.env(
                'KCNET_ROOT') + 'modelnet40/PD_harr/pd_harr_256_train.npy'
        else:
            pd_file = os.env(
                'KCNET_ROOT') + 'modelnet40/PD_harr/pd_harr_256_test.npy'
        if self.train:
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_train.npy'
        else:
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_test.npy'
        pd1s = np.load(pd1_file, allow_pickle=True)
        pds = np.load(pd_file, allow_pickle=True)
        b, c = pds.shape
        for i in range(b):
            self.dataset.append((pds[i][:256], pd1s[i][100]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pb1, label = self.dataset[index]
        pb1 = np.reshape(pb1, (16, 16))
        label = torch.tensor(label).long()
        return pb1, label


class PIModelNetDataset(data.Dataset):
    def __init__(self, point_nums=1024, train=True, argumentation=True):
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.class_nums = 40
        self.dataset = []
        if self.train:
            pi_file = os.env('KCNET_ROOT') + 'modelnet40/PD_pi/pi_00005_logistic_50_train.npy'
        else:
            pi_file = os.env('KCNET_ROOT') + 'modelnet40/PD_pi/pi_00005_logistic_50_test.npy'
        if self.train:
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_train.npy'
        else:
            pd1_file = os.env('KCNET_ROOT') + 'modelnet40/PB100_test.npy'
        # graphs = np.load(mesh_graph_file, allow_pickle=True)
        pd1s = np.load(pd1_file, allow_pickle=True)
        # graphs = np.load(mesh_graph_file, allow_pickle=True)
        pis = np.load(pi_file, allow_pickle=True)
        b, h, w = pis.shape
        for i in range(b):
            self.dataset.append((pis[i], pd1s[i][100]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pi, label = self.dataset[index]
        label = torch.tensor(label).long()
        return pi, label


# 2019.6.7
class MeshPIModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        totalData = np.load(
            self.datafile, allow_pickle=True,
            encoding='latin1').tolist()  # array(dict) ---> dict
        # self.class_nums = len(totalData['label_dict'])
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        print(totalData['data'].shape)
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        #dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        if ('train_' in self.datafile):
            mesh_graph_file = os.env('KCNET_ROOT') + 'modelnet40/train_graph_csr.npy'
            pi_file = os.env('KCNET_ROOT') + 'modelnet40/PD_pi/pi_00005_logistic_50_train.npy'
        else:
            mesh_graph_file = os.env('KCNET_ROOT') + 'modelnet40/test_graph_csr.npy'
            pi_file = os.env('KCNET_ROOT') + 'modelnet40/PD_pi/pi_00005_logistic_50_test.npy'
        mesh_graphs = np.load(mesh_graph_file, allow_pickle=True)
        pis = np.load(pi_file, allow_pickle=True)

        def init_worker():
            signal.signal(signal.SIGINT, signal.SIG_IGN)

        processes = 8
        if processes:
            pool = mp.Pool(processes=processes, initializer=init_worker)
        else:
            pool = None

        diagrams = list(zip(mesh_graphs, totalData['graph']))
        vertex_2rings = multiprocess_fn(pool, self.graph2ring,
                                        diagrams)  #, [opt1,opt2])

        for i in range(b):
            # vertex_2ring = self.graph2ring(mesh_graphs[i], totalData['graph'][i]['M'])
            # self.dataset.append((totalData['data'][i],mesh_graphs[i],totalData['graph'][i]['M'],pis[i],totalData['label'][i]))
            self.dataset.append((totalData['data'][i], vertex_2rings[i],
                                 pis[i], totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]

        # data, graph, knn_graph, pi, label=self.dataset[index]
        data, vertex_2ring, pi, label = self.dataset[index]
        if self.train and self.argumentation:
            #data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.transpose().astype(
            np.float32))  # --->(3, point_nums)
        # graph = graph.tocoo()
        # rows = graph.row
        # cols = graph.col
        # graph = [[row] for row in range(self.point_nums)]
        # for i,row in enumerate(rows):
        #     graph[row].append(cols[i])
        # vertex_2ring=copy.deepcopy(graph)
        # for i in range(self.point_nums):
        #     for source in graph[i][1:]:
        #         for target in graph[source][1:]:
        #             if(target not in vertex_2ring[i]):
        #                 vertex_2ring[i].append(target)
        # for row in range(self.point_nums): # for vertex_2ring
        #     res=17-len(vertex_2ring[row])
        #     if(res<=0):
        #         vertex_2ring[row] = vertex_2ring[row][:17]
        #     else:
        #         i=0
        #         for node in knn_graph.getrow(row).nonzero()[1]:
        #             if(node not in vertex_2ring[row]):
        #                 vertex_2ring[row].append(node)
        #                 i+=1
        #             if(i>=res):
        #                 break
        # vertex_2ring = self.graph2ring(graph, knn_graph)

        # graph = [row_graph + [row]*(17-len(row_graph))  for row, row_graph in enumerate(graph)]
        vertex_2ring = torch.from_numpy(
            np.array(vertex_2ring))  # (point_nums, 17)
        label = torch.tensor(label).long()
        return pc, vertex_2ring, pi, label

    def graph2ring(self, inputs):
        graph, knn_graph = inputs
        knn_graph = knn_graph['M']
        graph = graph.tocoo()
        rows = graph.row
        cols = graph.col
        graph = [[row] for row in range(self.point_nums)]
        for i, row in enumerate(rows):
            graph[row].append(cols[i])
        vertex_2ring = copy.deepcopy(graph)
        for i in range(self.point_nums):
            for source in graph[i][1:]:
                for target in graph[source][1:]:
                    if (target not in vertex_2ring[i]):
                        vertex_2ring[i].append(target)
        for row in range(self.point_nums):  # for vertex_2ring
            res = 17 - len(vertex_2ring[row])
            if (res <= 0):
                vertex_2ring[row] = vertex_2ring[row][:17]
            else:
                i = 0
                for node in knn_graph.getrow(row).nonzero()[1]:
                    if (node not in vertex_2ring[row]):
                        vertex_2ring[row].append(node)
                        i += 1
                    if (i >= res):
                        break
        return vertex_2ring


## Nvidia 分布式框架 Apex
## https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_pc, self.next_vertex_2ring, self.next_pi, self.next_label = next(
                self.loader)
        except StopIteration:
            self.next_pc = None
            self.next_vertex_2ring = None
            self.next_pi = None
            self.next_label = None
            return
        with torch.cuda.stream(self.stream):
            self.next_pc = self.next_pc.cuda(non_blocking=True)
            self.next_vertex_2ring = self.next_vertex_2ring.to(
                torch.device('cuda:0'), non_blocking=True, dtype=torch.int)
            self.next_pi = self.next_pi.to(torch.device('cuda:0'),
                                           non_blocking=True,
                                           dtype=torch.float32)
            self.next_label = self.next_label.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        pc = self.next_pc
        vertex_2ring = self.next_vertex_2ring
        pi = self.next_pi
        label = self.next_label
        self.preload()
        return pc, vertex_2ring, pi, label


############################ SO NET ###########################
###############################################################
def make_dataset_modelnet40_10k(root, mode, opt):
    dataset = []
    rows = round(math.sqrt(opt.node_num))
    cols = rows

    f = open(os.path.join(root, 'modelnet%d_shape_names.txt' % opt.classes))
    shape_list = [str.rstrip() for str in f.readlines()]
    f.close()

    if 'train' == mode:
        f = open(os.path.join(root, 'modelnet%d_train.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    elif 'test' == mode:
        f = open(os.path.join(root, 'modelnet%d_test.txt' % opt.classes), 'r')
        lines = [str.rstrip() for str in f.readlines()]
        f.close()
    else:
        raise Exception('Network mode error.')

    for i, name in enumerate(lines):
        # locate the folder name
        folder = name[0:-5]
        file_name = name

        # get the label
        label = shape_list.index(folder)

        # som node locations
        som_nodes_folder = '%dx%d_som_nodes' % (rows, cols)

        item = (os.path.join(root, folder, file_name + '.npy'), label,
                os.path.join(root, som_nodes_folder, folder,
                             file_name + '.npy'))
        dataset.append(item)

    return dataset


class KNNBuilder:
    def __init__(self, k):
        self.k = k
        self.dimension = 3

    def build_nn_index(self, database):
        '''
        :param database: numpy array of Nx3
        :return: Faiss index, in CPU
        '''
        index = faiss.IndexFlatL2(self.dimension)  # dimension is 3
        index.add(database)
        return index

    def search_nn(self, index, query, k):
        '''
        :param index: Faiss index
        :param query: numpy array of Nx3
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        D, I = index.search(query, k)
        return D, I

    def self_build_search(self, x):
        '''
        :param x: numpy array of Nxd
        :return: D: numpy array of Nxk
                 I: numpy array of Nxk
        '''
        x = np.ascontiguousarray(x, dtype=np.float32)
        index = self.build_nn_index(x)
        D, I = self.search_nn(index, x, self.k)
        return D, I


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points)**2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(distances,
                                   self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


class ModelNet_Shrec_Loader(data.Dataset):
    def __init__(self, root, mode, opt):
        super(ModelNet_Shrec_Loader, self).__init__()
        self.root = root
        self.opt = opt
        self.mode = mode
        if self.opt.dataset == 'modelnet':
            self.dataset = make_dataset_modelnet40_10k(self.root, mode, opt)
        else:
            raise Exception('Dataset incorrect.')
        # kNN search on SOM nodes
        self.knn_builder = KNNBuilder(self.opt.som_k)
        # farthest point sample
        self.fathest_sampler = FarthestSampler()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.opt.dataset == 'modelnet':
            pc_np_file, class_id, som_node_np_file = self.dataset[index]
            data = np.load(pc_np_file)
            data = data[np.random.choice(
                data.shape[0], self.opt.input_pc_num, replace=False), :]
            pc_np = data[:, 0:3]  # Nx3
            surface_normal_np = data[:, 3:6]  # Nx3
            som_node_np = np.load(som_node_np_file)  # node_numx3
        else:
            raise Exception('Dataset incorrect.')

        # augmentation
        if self.mode == 'train':
            # rotate by 0/90/180/270 degree over z axis
            # pc_np = rotate_point_cloud_90(pc_np)
            # som_node_np = rotate_point_cloud_90(som_node_np)

            # rotation perturbation, pc and som should follow the same rotation, surface normal rotation is unclear
            # if self.opt.rot_horizontal:
            #     pc_np, surface_normal_np, som_node_np = rotate_point_cloud_with_normal_som(pc_np, surface_normal_np, som_node_np)
            # if self.opt.rot_perturbation:
            #     pc_np, surface_normal_np, som_node_np = rotate_perturbation_point_cloud_with_normal_som(pc_np, surface_normal_np, som_node_np)

            # random jittering
            pc_np = jitter_point_cloud(pc_np)
            surface_normal_np = jitter_point_cloud(surface_normal_np)
            som_node_np = jitter_point_cloud(som_node_np, sigma=0.04, clip=0.1)

            # random scale
            # scale = np.random.uniform(low=0.8, high=1.2)
            # pc_np = pc_np * scale
            # som_node_np = som_node_np * scale
            # surface_normal_np = surface_normal_np * scale
            #
            # # random shift
            # if self.opt.translation_perturbation:
            #     shift = np.random.uniform(-0.1, 0.1, (1,3))
            #     pc_np += shift
            #     som_node_np += shift

        # convert to tensor
        pc = torch.from_numpy(pc_np.transpose().astype(np.float32))  # 3xN
        # surface normal
        surface_normal = torch.from_numpy(surface_normal_np.transpose().astype(
            np.float32))  # 3xN
        # som
        som_node = torch.from_numpy(som_node_np.transpose().astype(
            np.float32))  # 3xnode_num

        # kNN search: som -> som
        if self.opt.som_k >= 2:
            D, I = self.knn_builder.self_build_search(som_node_np)
            som_knn_I = torch.from_numpy(I.astype(
                np.int64))  # node_num x som_k
        else:
            som_knn_I = torch.from_numpy(
                np.arange(start=0, stop=self.opt.node_num,
                          dtype=np.int64).reshape(
                              (self.opt.node_num, 1)))  # node_num x 1
        return pc, surface_normal, class_id, som_node, som_knn_I




#################### ECC ####################
import transforms3d
import torchnet as tnt
import open3d
import utils.pointcloud_utils as pcu

class ECC_ModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 pyramid_conf,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []
        self.pyramid_conf = pyramid_conf

        totalData = np.load(
            self.datafile, allow_pickle=True).tolist()  # array(dict) -> dict
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        # dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        for i in range(b):
            self.dataset.append((totalData['data'][i], totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]
        data, label = self.dataset[index]
        if self.train and self.argumentation:
            # data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        pc = torch.from_numpy(data.astype(np.float32))  # --->(3, point_nums)

        diameter = np.max(
            np.max(pc.numpy(), axis=0) - np.min(pc.numpy(), axis=0))
        M = transforms3d.zooms.zfdir2mat(32 / diameter)
        pc = np.dot(pc, M.T)
        # coarsen to initial resolution (btw, axis-aligned quantization of rigidly transformed cloud adds jittering noise)
        pc -= np.min(pc, axis=0)  
        # move to positive octant (voxelgrid has fixed boundaries at axes planes)
        cloud = pcu.create_cloud(pc)
        cloud = open3d.voxel_down_sample(cloud,
                                         voxel_size=self.pyramid_conf[0][0])
        F = np.ones((len(cloud.points), 1),
                    dtype=np.float32)  # no point features in modelnet
        graphs, poolmaps = pcu.create_graph_pyramid(0, cloud,
                                                    self.pyramid_conf)
        label = torch.tensor(label).long()
        return F, label, graphs, poolmaps




####################### KD Net###################
from torch.autograd import Variable
from other_models.kdnet.kdtree import make_cKDTree

class KDNet_ModelNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 depth=11,
                 point_nums=1024,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []
        self.depth = depth

        totalData = np.load(
            self.datafile, allow_pickle=True).tolist()  # array(dict) ---> dict
        self.class_nums = len(set(totalData['label']))
        print('class nums={}'.format(self.class_nums))
        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        # dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        for i in range(b):
            self.dataset.append((totalData['data'][i], totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # file_path, label = self.dataset[index]
        # data = np.loadtxt(file_path, dtype=np.float32, delimiter=',', usecols=(0, 1, 2))
        # data = data[np.random.choice(data.shape[0], self.point_nums, replace=False), :]
        data, label = self.dataset[index]
        if self.train and self.argumentation:
            # data = random_rotate_point_cloud(data)
            data = jitter_point_cloud(data)

        cutdim, tree = make_cKDTree(data, depth=self.depth)
        cutdim_v = [(torch.from_numpy(np.array(item).astype(np.int64)))
                    for item in cutdim]
        points = torch.FloatTensor(tree[-1])  # 1,1024,3
        points_v = Variable(torch.squeeze(points)).transpose(0, 1)  # (3,1024)
        label = torch.tensor(label).long()
        return points_v, cutdim_v, label


if __name__ == "__main__":
    # dataset = make_dataset_modelnet40(os.env('KCNET_ROOT') + 'modelnet/modelnet40_ply_hdf5_2048/', True)
    class VirtualOpt():
        def __init__(self):
            self.load_all_data = False
            self.input_pc_num = 5000
            self.batch_size = 8
            self.dataset = '10k'
            self.node_num = 64
            self.classes = 10
            self.som_k = 9

    opt = VirtualOpt()
    trainset = ModelNet_Shrec_Loader(os.env('KCNET_ROOT') + 'modelnet/modelnet40-normal_numpy/',
                                     'train', opt)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=4)
    KDNet_trainset = KDNet_ModelNetDataset(
                                'modelnet/modelNet40_train_16nn_GM_adaptive_knn_sparse.npy',
                                depth=10,
                                train=True)