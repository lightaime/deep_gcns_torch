import numpy as np
import h5py
import os
import os.path as osp
import shutil
from glob import glob
import torch
from torch_scatter import scatter
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from tqdm import tqdm
import torch_geometric as tg


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def process_indexes(idx_list):
    idx_dict = {}
    for i, idx in enumerate(idx_list):
        idx_dict[idx] = i

    return [idx_dict[i] for i in sorted(idx_dict.keys())]


def add_zeros(data):
    data.x = torch.zeros(data.num_nodes, dtype=torch.long)
    return data


def extract_node_feature(data, reduce='add'):
    if reduce in ['mean', 'max', 'add']:
        data.x = scatter(data.edge_attr,
                         data.edge_index[0],
                         dim=0,
                         dim_size=data.num_nodes,
                         reduce=reduce)
    else:
        raise Exception('Unknown Aggregation Type')
    return data

# random partition graph
def random_partition_graph(num_nodes, cluster_number=10):
    parts = np.random.randint(cluster_number, size=num_nodes)
    return parts


def generate_sub_graphs(adj, parts, cluster_number=10, batch_size=1):
    # convert sparse tensor to scipy csr
    adj = adj.to_scipy(layout='csr')

    num_batches = cluster_number // batch_size

    sg_nodes = [[] for _ in range(num_batches)]
    sg_edges = [[] for _ in range(num_batches)]

    for cluster in range(num_batches):
        sg_nodes[cluster] = np.where(parts == cluster)[0]
        sg_edges[cluster] = tg.utils.from_scipy_sparse_matrix(adj[sg_nodes[cluster], :][:, sg_nodes[cluster]])[0]

    return sg_nodes, sg_edges

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


def scale_translate_pointcloud(pointcloud, shift=[-0.2, 0.2], scale=[2. / 3., 3. /2.]):
    """
    for scaling and shifting the point cloud
    :param pointcloud:
    :return:
    """
    B, C, N = pointcloud.shape[0:3]
    scale = scale[0] + torch.rand([B, C, 1, 1])*(scale[1]-scale[0])
    shift = shift[0] + torch.rand([B, C, 1, 1]) * (shift[1]-shift[0])
    translated_pointcloud = torch.mul(pointcloud, scale) + shift
    return translated_pointcloud


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


# allowable multiple choice node and edge features
# code from https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)) + ['misc'],
    'possible_chirality_list' : [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list' : [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
        ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring_list': [False, True],
    'possible_bond_type_list' : [
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
        'misc'
    ],
    'possible_bond_stereo_list': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'possible_is_conjugated_list': [False, True],
}


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1


def atom_to_feature_vector(atom):
    """
    Converts rdkit atom object to feature list of indices
    :param mol: rdkit atom object
    :return: list
    """
    atom_feature = [
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
            ]
    return atom_feature


def get_atom_feature_dims():
    return list(map(len, [
        allowable_features['possible_atomic_num_list'],
        allowable_features['possible_chirality_list'],
        allowable_features['possible_degree_list'],
        allowable_features['possible_formal_charge_list'],
        allowable_features['possible_numH_list'],
        allowable_features['possible_number_radical_e_list'],
        allowable_features['possible_hybridization_list'],
        allowable_features['possible_is_aromatic_list'],
        allowable_features['possible_is_in_ring_list']
        ]))


def bond_to_feature_vector(bond):
    """
    Converts rdkit bond object to feature list of indices
    :param mol: rdkit bond object
    :return: list
    """
    bond_feature = [
                safe_index(allowable_features['possible_bond_type_list'], str(bond.GetBondType())),
                allowable_features['possible_bond_stereo_list'].index(str(bond.GetStereo())),
                allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated()),
            ]
    return bond_feature


def get_bond_feature_dims():
    return list(map(len, [
        allowable_features['possible_bond_type_list'],
        allowable_features['possible_bond_stereo_list'],
        allowable_features['possible_is_conjugated_list']
        ]))


def atom_feature_vector_to_dict(atom_feature):
    [atomic_num_idx,
    chirality_idx,
    degree_idx,
    formal_charge_idx,
    num_h_idx,
    number_radical_e_idx,
    hybridization_idx,
    is_aromatic_idx,
    is_in_ring_idx] = atom_feature

    feature_dict = {
        'atomic_num': allowable_features['possible_atomic_num_list'][atomic_num_idx],
        'chirality': allowable_features['possible_chirality_list'][chirality_idx],
        'degree': allowable_features['possible_degree_list'][degree_idx],
        'formal_charge': allowable_features['possible_formal_charge_list'][formal_charge_idx],
        'num_h': allowable_features['possible_numH_list'][num_h_idx],
        'num_rad_e': allowable_features['possible_number_radical_e_list'][number_radical_e_idx],
        'hybridization': allowable_features['possible_hybridization_list'][hybridization_idx],
        'is_aromatic': allowable_features['possible_is_aromatic_list'][is_aromatic_idx],
        'is_in_ring': allowable_features['possible_is_in_ring_list'][is_in_ring_idx]
    }

    return feature_dict


def bond_feature_vector_to_dict(bond_feature):
    [bond_type_idx,
    bond_stereo_idx,
    is_conjugated_idx] = bond_feature

    feature_dict = {
        'bond_type': allowable_features['possible_bond_type_list'][bond_type_idx],
        'bond_stereo': allowable_features['possible_bond_stereo_list'][bond_stereo_idx],
        'is_conjugated': allowable_features['possible_is_conjugated_list'][is_conjugated_idx]
    }

    return feature_dict
