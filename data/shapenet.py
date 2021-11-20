import os
import json
import numpy as np
import torch
import torch.utils.data as data
from utils.augmentation import *


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 point_nums=2048,
                 split='train',
                 argumentation=True):
        self.root = root
        self.point_nums = point_nums
        self.split = split
        self.argumentation = argumentation
        self.dataset = []

        categories = {}
        with open(os.path.join(self.root, 'synsetoffset2category.txt'),
                  'r') as file:
            i = 0
            for line in file:
                ls = line.strip().split()[1]
                categories[ls] = i
                i = i + 1
        self.category_nums = len(categories)
        self.class_nums = 6

        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_train_file_list.json'), 'r') as file:
            train_idxs = [(d.split('/')[1], d.split('/')[2])
                          for d in json.load(file)]
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_val_file_list.json'), 'r') as file:
            val_idxs = [(d.split('/')[1], d.split('/')[2])
                        for d in json.load(file)]
        with open(
                os.path.join(self.root, 'train_test_split',
                             'shuffled_test_file_list.json'), 'r') as file:
            test_idxs = [(d.split('/')[1], d.split('/')[2])
                         for d in json.load(file)]

        if split == 'train':
            for category, hash in train_idxs:
                self.dataset.append(
                    (os.path.join(self.root, category, 'points',
                                  hash + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  hash + '.seg'), categories[category]))
        elif split == 'val':
            for category, hash in val_idxs:
                self.dataset.append(
                    (os.path.join(self.root, category, 'points',
                                  hash + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  hash + '.seg'), categories[category]))
        elif split == 'test':
            for category, hash in test_idxs:
                self.dataset.append(
                    (os.path.join(self.root, category, 'points',
                                  hash + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  hash + '.seg'), categories[category]))
        elif split == 'train&val':
            for category, hash in train_idxs:
                self.dataset.append(
                    (os.path.join(self.root, category, 'points',
                                  hash + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  hash + '.seg'), categories[category]))
            for category, hash in val_idxs:
                self.dataset.append(
                    (os.path.join(self.root, category, 'points',
                                  hash + '.pts'),
                     os.path.join(self.root, category, 'points_label',
                                  hash + '.seg'), categories[category]))
        else:
            raise NameError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pc_path, seg_path, label = self.dataset[index]
        pc_data = np.loadtxt(pc_path, dtype=np.float32)
        seg_data = np.loadtxt(seg_path, dtype=np.long)
        if pc_data.shape[0] >= self.point_nums:
            sampling_idxs = np.random.choice(pc_data.shape[0],
                                             self.point_nums,
                                             replace=False)
        else:
            sampling_idxs = np.random.choice(pc_data.shape[0],
                                             self.point_nums -
                                             pc_data.shape[0],
                                             replace=True)
            sampling_idxs = np.append(sampling_idxs,
                                      [i for i in range(0, pc_data.shape[0])])
            np.random.shuffle(sampling_idxs)
        pc_data = pc_data[sampling_idxs, :]
        seg_data = seg_data[sampling_idxs] - 1

        if (self.split == 'train'
                or self.split == 'train&val') and self.argumentation:
            pc_data = jitter_point_cloud(pc_data)

        pc = torch.from_numpy(pc_data.transpose().astype(np.float32))
        seg = torch.from_numpy(seg_data.astype(np.long))
        return pc, seg, label


class AdaptiveShapeNetDataset(data.Dataset):
    def __init__(self,
                 datafile,
                 point_nums=2048,
                 train=True,
                 argumentation=True):
        self.datafile = datafile
        self.point_nums = point_nums
        self.train = train
        self.argumentation = argumentation
        self.dataset = []

        totalData = np.load(self.datafile).tolist()  # array(dict) ---> dict
        # self.class_nums = len(totalData['label_dict'])
        self.class_nums = 50
        self.category_nums = 16
        print('class nums={}'.format(self.class_nums))
        print(totalData['data'].shape)

        # self.shape_list = [str.rstrip() for str in file.readlines()]  #[bathtub bed chair desk
        #dresser monitor night_stand sofa table toilet]
        b, n, c = totalData['data'].shape
        for i in range(b):
            self.dataset.append(
                (totalData['data'][i], totalData['graph'][i],
                 totalData['seg_label'][i], totalData['label'][i]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, graph, seg, label = self.dataset[index]
        # sampling_idxs = np.random.choice(pc_data.shape[0], self.point_nums, replace=False)
        # pc_data = pc_data[sampling_idxs, :]
        # seg_data = seg_data[sampling_idxs]
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
            row_graph + [row] * (19 - len(row_graph))
            for row, row_graph in enumerate(graph)
        ]
        graph = torch.from_numpy(np.array(graph))  # (point_nums, 19)
        seg = torch.from_numpy(seg.astype(np.long))
        label = torch.tensor(label).long()
        return pc, graph, seg, label
