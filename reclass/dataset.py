import torch
from torch.utils.data import Dataset
import numpy as np

class DummyDataset(Dataset):
    def __init__(self, cfg, length=1000):
        self.cfg = cfg
        self.length = length
        self.T = cfg['train']['clip_len']
        self.H = cfg['train']['image_h']
        self.W = cfg['train']['image_w']
    def __len__(self): return self.length
    def __getitem__(self, idx):
        rgb = np.random.rand(self.T, 3, self.H, self.W).astype('float32')
        flow = np.random.rand(self.T, 2, self.H, self.W).astype('float32')
        nodes = np.random.rand(40, 32).astype('float32')
        adj = np.random.rand(40, 40).astype('float32')
        return {'rgb': torch.tensor(rgb), 'flow': torch.tensor(flow), 'nodes': torch.tensor(nodes), 'adj': torch.tensor(adj)}
