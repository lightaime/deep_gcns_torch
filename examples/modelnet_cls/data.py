#!/usr/bin/env python
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


def download(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], data_dir))
        os.system('rm %s' % (zipfile))


def load_data(data_dir, partition):
    download(data_dir)
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(data_dir, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    """
    for scaling and shifting the point cloud
    :param pointcloud:
    :return:
    """
    scale = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    shift = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, scale), shift).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    """
    This is the data loader for ModelNet 40
    ModelNet40 contains 12,311 meshed CAD models from 40 categories.

    num_points: 1024 by default
    data_dir
    paritition: train or test
    """
    def __init__(self, num_points=1024, data_dir="/data/deepgcn/modelnet40", partition='train'):
        self.data, self.label = load_data(data_dir, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]

    def num_classes(self):
        return np.max(self.label) + 1


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label.shape)
