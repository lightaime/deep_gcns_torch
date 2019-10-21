import numpy as np
import h5py
import os
import os.path as osp
import shutil
from glob import glob
import torch
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from tqdm import tqdm


def random_rotate(points):
    theta = np.random.uniform(0, np.pi * 2)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotation_matrix = torch.from_numpy(rotation_matrix).float()
    points[:, 0:2] = torch.matmul(points[:, [0, 1]].transpose(1, 3), rotation_matrix).transpose(1, 3)
    return points


def random_translate(points, mean=0, std=0.02):
    points += torch.randn(points.shape)*std + mean
    return points


def random_points_augmentation(points, rotate=False, translate=False, **kwargs):
    if rotate:
        points = random_rotate(points)
    if translate:
        points = random_translate(points, **kwargs)

    return points


class PartNet(InMemoryDataset):
    r"""The PartNet dataset from
    the `"PartNet: A Large-scale Benchmark for Fine-grained and Hierarchical Part-level 3D Object Understanding"
    <https://arxiv.org/abs/1812.02713>`_
    paper, containing 3D objects annotated with fine-grained, instance-level, and hierarchical 3D part information.

    Args:
        root (string): Root directory where the dataset should be saved.
        dataset (str, optional): Which dataset to use (ins_seg_h5, or sem_seg_h5).
            (default: :obj:`sem_seg_h5`)
        obj_category (str, optional): which category to load.
            (default: :obj:`Bed`)
        level (str, optional): Which level of part semantic segmentation to use.
            (default: :obj:`3`)
        phase (str, optional): If :obj:`test`, loads the testing dataset,
            If :obj:`val`, loads the validation dataset,
            otherwise the training dataset. (default: :obj:`train`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """
    # the dataset we use for our paper is pre-released version
    def __init__(self,
                 root,
                 dataset='sem_seg_h5',
                 obj_category='Bed',
                 level=3,
                 phase='train',
                 transform=None,
                 pre_transform=None,
                 pre_filter=None):
        self.dataset = dataset
        self.level = level
        self.obj_category = obj_category
        self.object = '-'.join([self.obj_category, str(self.level)])
        self.level_folder = 'level_'+str(self.level)
        self.processed_file_folder = osp.join(self.dataset, self.level_folder, self.object)
        super(PartNet, self).__init__(root, transform, pre_transform, pre_filter)
        if phase == 'test':
            path = self.processed_paths[1]
        elif phase == 'val':
            path = self.processed_paths[2]
        else:
            path = self.processed_paths[0]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [self.dataset]

    @property
    def processed_file_names(self):
        return osp.join(self.processed_file_folder, 'train.pt'), osp.join(self.processed_file_folder, 'test.pt'), \
               osp.join(self.processed_file_folder, 'val.pt')

    def download(self):
        path = osp.join(self.raw_dir, self.dataset)
        if not osp.exists(path):
            raise FileExistsError('PartNet can only downloaded via application. '
                                  'See details in https://cs.stanford.edu/~kaichun/partnet/')
        # path = download_url(self.url, self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        shutil.rmtree(self.raw_dir)
        name = self.url.split(os.sep)[-1].split('.')[0]
        os.rename(osp.join(self.root, name), self.raw_dir)

    def process(self):
        # save to processed_paths
        processed_path = osp.join(self.processed_dir, self.processed_file_folder)
        if not osp.exists(processed_path):
            os.makedirs(osp.join(processed_path))
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])
        torch.save(self.process_set('val'), self.processed_paths[2])

    def process_set(self, dataset):
        if self.dataset == 'ins_seg_h5':
            raw_path = osp.join(self.raw_dir, 'ins_seg_h5_for_sgpn', self.dataset)
            categories = glob(osp.join(raw_path, '*'))
            categories = sorted([x.split(os.sep)[-1] for x in categories])

            data_list = []
            for target, category in enumerate(tqdm(categories)):
                folder = osp.join(raw_path, category)
                paths = glob('{}/{}-*.h5'.format(folder, dataset))
                labels, nors, opacitys, pts, rgbs = [], [], [], [], []
                for path in paths:
                    f = h5py.File(path)
                    pts += torch.from_numpy(f['pts'][:]).unbind(0)
                    labels += torch.from_numpy(f['label'][:]).to(torch.long).unbind(0)
                    nors += torch.from_numpy(f['nor'][:]).unbind(0)
                    opacitys += torch.from_numpy(f['opacity'][:]).unbind(0)
                    rgbs += torch.from_numpy(f['rgb'][:]).to(torch.float32).unbind(0)

                for i, (pt, label, nor, opacity, rgb) in enumerate(zip(pts, labels, nors, opacitys, rgbs)):
                    data = Data(pos=pt[:, :3], y=label, norm=nor[:, :3], x=torch.cat((opacity.unsqueeze(-1), rgb/255.), 1))

                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
        else:
            raw_path = osp.join(self.raw_dir, self.dataset)
            categories = glob(osp.join(raw_path, self.object))
            categories = sorted([x.split(os.sep)[-1] for x in categories])
            data_list = []
            # class_name = []
            for target, category in enumerate(tqdm(categories)):
                folder = osp.join(raw_path, category)
                paths = glob('{}/{}-*.h5'.format(folder, dataset))
                labels, pts = [], []
                # clss = category.split('-')[0]

                for path in paths:
                    f = h5py.File(path)
                    pts += torch.from_numpy(f['data'][:].astype(np.float32)).unbind(0)
                    labels += torch.from_numpy(f['label_seg'][:].astype(np.float32)).to(torch.long).unbind(0)
                for i, (pt, label) in enumerate(zip(pts, labels)):
                    data = Data(pos=pt[:, :3], y=label)
                    # data = PartData(pos=pt[:, :3], y=label, clss=clss)
                    if self.pre_filter is not None and not self.pre_filter(data):
                        continue
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    data_list.append(data)
        return self.collate(data_list)


class PartData(Data):
    def __init__(self,
                 y=None,
                 pos=None,
                 clss=None):
        super(PartData).__init__(pos=pos, y=y)
        self.clss = clss


