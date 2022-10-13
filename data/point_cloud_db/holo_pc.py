from data.point_cloud_db.point_cloud_dataset import PointCloudDataset, get_max_dist
import os
from pathlib import Path
import sys
from visualization.mesh_container import MeshContainer



import numpy as np
import torch
import itertools


from tqdm import tqdm


class HOLO_PC(PointCloudDataset):
 def __init__(self, hparams,split):
    super(HOLO_PC, self).__init__(hparams,split=split)
    if(self.split == 'train'):
        self.gt_map = None
    
    
def valid_pairs(self,gt_map):
    if(self.split == 'test'):
        return [[int(idx) for idx in k.split('_')] for k in list(gt_map.keys())]
    else:
        pairs = list(itertools.product(list(range(len(self.verts))), list(range(len(self.verts)))))
        return list(filter(lambda pair: pair[0] != pair[1],pairs))
    
def __getitem__(self, item):
    out_dict = super(HOLO_PC, self).__getitem__(item)
    return out_dict