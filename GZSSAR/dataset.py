import pdb

import numpy as np
from torch.utils.data import Dataset


class NTUDataset(Dataset):
    """
    Skeleton Dataset.
    Args:
        x (list): Input dataset, each element in the list is an ndarray corresponding to a joints matrix of a skeleton sequence sample
        y (list): Action labels
    """

    def __init__(self, x, y):
        self.x = np.load(x, mmap_mode='r')
        self.y = np.load(y)
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = np.array(self.x[index])
        y = self.y[index]
        
        return x, y


def create(sk_data_dir):
    train_dir, val_dir = sk_data_dir['train'], sk_data_dir['val']
    train_X = train_dir + '/train.npy'
    train_Y = train_dir + '/train_label.npy'
    zsl_X = train_dir + '/ztest.npy'
    zsl_Y = train_dir + '/z_label.npy'
    gzsl_X = train_dir + '/gtest.npy'
    gzsl_Y = train_dir + '/g_label.npy'
    
    val_X = val_dir + '/train.npy'
    val_Y = val_dir + '/train_label.npy'
    val_unseen_X = val_dir + '/ztest.npy'
    val_unseen_Y = val_dir + '/z_label.npy'
    val_seen_X = val_dir + '/val.npy'
    val_seen_Y = val_dir + '/val_label.npy'

    feeders = {
        'train': NTUDataset(train_X, train_Y), 'zsl': NTUDataset(zsl_X, zsl_Y), 'gzsl': NTUDataset(gzsl_X, gzsl_Y),
        'val': NTUDataset(val_X, val_Y), 'val_unseen': NTUDataset(val_unseen_X, val_unseen_Y), 'val_seen': NTUDataset(val_seen_X, val_seen_Y)
    }
    
    sk_emb_dim = feeders['train'].x.shape[-1]

    return feeders, sk_emb_dim