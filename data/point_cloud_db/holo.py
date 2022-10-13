#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.point_cloud_db.point_cloud_dataset import PointCloudDataset, get_max_dist
import os
from pathlib import Path
import sys
from visualization.mesh_container import MeshContainer

import numpy as np
import torch
import itertools

from tqdm import tqdm

import open3d as o3d
from os import listdir
from os.path import isfile, join


def preprocess(dataset_path):
    # get all scenes in subfolders of all subjects (i.e ..\\subject_1\2022-03-18-143932\2022-03-18-143932..)
    scenes = []
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in dirs:
            scenes.append(os.path.join(root, name))

    os.makedirs(r'data/datasets/hololens_off', exist_ok=True)

    #####
    off_path = (r'data/datasets/hololens_off')
    for scene in scenes:
        path = scene
        takes = [f for f in listdir(path) if isfile(join(path, f))]
        for pc in takes:
            pc_path = path + '/' + pc
            pcd = o3d.io.read_point_cloud(pc_path)
            cloud_np = np.asarray(pcd.points)
            normalized_pc = 2.*(cloud_np-np.min(cloud_np))/np.ptp(cloud_np)-1 #normalize between -1 and 1 like shrec
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(normalized_pc)
            pcd.estimate_normals()

            new = off_path + '/' + pc[:-11] + r'.off'  # save to same folder
            # estimate radius for rolling ball
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector([radius, radius * 2]))
            mesh.triangle_normals = o3d.utility.Vector3dVector([])
            o3d.io.write_triangle_mesh(new, mesh)


def get_pairs(dataset_path):
    # get all scenes in subfolders of all subjects (i.e ..\\subject_1\2022-03-18-143932\2022-03-18-143932..)
    # scenes = []
    # for root, dirs, files in os.walk(dataset_path, topdown=False):
    #     for name in dirs:
    #         scenes.append(os.path.join(root, name))

    # get all directories of paris (source-t, target- t+1) for all scenes
    pairs = []
    # for scene_path in scenes:
    scene_path = dataset_path
    frames = [f for f in listdir(scene_path) if isfile(join(scene_path, f))]
    frames = [f for f in frames if not(f=='unified')]
    num_of_frames = len(frames)
    for f in range(num_of_frames - 1):
        pair = tuple([frames[f][:-4], frames[f + 1][:-4]])
        pairs.append(pair)
    return list(filter(lambda pair: pair[0] != pair[1], pairs))


hololens_data_path = 'data/datasets/hololens_dataset'
if not os.path.exists(r'data/datasets/hololens_off'):
    preprocess(hololens_data_path)

class hololens_dataset(PointCloudDataset):

    def __init__(self, hparams, split):
        super(hololens_dataset, self).__init__(hparams, split=split)
        if (self.split == 'train'):
            self.gt_map = None

    def valid_pairs(self, gt_map):
        # hololens_data_path = 'data/datasets/hololens_dataset' #123456789
        hololens_data_path = 'data/datasets/hololens_off'

        pairs = get_pairs(hololens_data_path)
        return pairs

    def __getitem__(self, item):
        out_dict = super(hololens_dataset, self).__getitem__(item)
        return out_dict

    @staticmethod
    def add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf=False):
        parser = PointCloudDataset.add_dataset_specific_args(parser, task_name, dataset_name, is_lowest_leaf)
        parser.set_defaults(test_on_hololens_dataset=True)
        return parser

    @staticmethod
    def load_data(*args):
        hololens_data_path = 'data/datasets/hololens_off'

        shapes_path = hololens_data_path
        if (not os.path.exists(f"{hololens_data_path}/unified")):
            all_verts, all_faces, all_d_max, all_maps = [], [], [], {}
            sorted_paths = sorted([str(path) for path in list(Path(shapes_path).rglob("*.off"))],key=lambda p:int(os.path.basename(p)[:-4]))

            for off in tqdm(sorted_paths, desc="Unifying Hololens Dataset"):
                mesh = MeshContainer().load_from_file(str(off))
                all_verts.append(mesh.vert)
                all_faces.append(mesh.face)
                all_d_max.append(get_max_dist(mesh.vert))

            torch.save((all_verts, all_faces, all_d_max, all_maps), f"{hololens_data_path}/unified")

        else:
            all_verts, all_faces, all_d_max, all_maps = torch.load(f"{hololens_data_path}/unified")
            all_maps = {k: v.astype(np.long) for k, v in all_maps.items()}

        return all_verts, all_faces, all_d_max, all_maps