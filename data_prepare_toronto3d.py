from sklearn.neighbors import KDTree
from os.path import join, exists, dirname, abspath
import numpy as np
import os, pickle
import sys
from plyfile import PlyData

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
from helper_ply import write_ply
from helper_tool1 import DataProcessing as DP

def read_ply_file(file_path):
    try:
        plydata = PlyData.read(file_path)
        pc = plydata['vertex'].data
        return pc
    except Exception as e:
        print("Error reading PLY file: {}".format(e))
        return None

grid_size = 0.06
dataset_path = '/media/ellie/Github/RandLA-Net/data/Toronto_3D'
train_files = ['merge1-30_r9', 'merge1-30_r8', 'merge1-30_r6', 'merge1-30_r7']
val_files = ['merge1-30_5v1']
UTM_OFFSET = [627285, 4841948, 0]
original_pc_folder = join(dataset_path, 'original_ply')
sub_pc_folder = join(dataset_path, 'input_{:.3f}'.format(grid_size))
os.mkdir(sub_pc_folder) if not exists(sub_pc_folder) else None

for pc_path in [join(original_pc_folder, fname + '.ply') for fname in train_files + val_files]:
    print(pc_path)
    file_name = pc_path.split('/')[-1][:-4]

    pc = read_ply_file(pc_path)
    if pc is None:
        print("Skipping file {} due to reading error.".format(file_name))
        continue

    print("Available fields:", pc.dtype.names)

    if file_name in train_files:
        xyz = np.vstack((pc['x'] - UTM_OFFSET[0], pc['y'] - UTM_OFFSET[1], pc['z'] - UTM_OFFSET[2])).T.astype(np.float32)
        #color = np.vstack((pc['red'], pc['green'], pc['blue'])).T.astype(np.uint8)
        color = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
        #intensity = pc['scalar_Intensity'].astype(np.uint8).reshape(-1,1)
        intensity = pc['scalar_Intensity'].astype(np.uint8).reshape(-1,1)
        #labels = pc['scalar_Label'].astype(np.uint8)
        labels = pc['scalar_Label'].astype(np.uint8)
    else:
        # For test file with generic field names
        #xyz = np.vstack((pc['f0'] - UTM_OFFSET[0], pc['f1'] - UTM_OFFSET[1], pc['f2'] - UTM_OFFSET[2])).T.astype(np.float32) # For unlabeled data
        xyz = np.vstack((pc['x'] - UTM_OFFSET[0], pc['y'] - UTM_OFFSET[1], pc['z'] - UTM_OFFSET[2])).T.astype(np.float32)
        #color = np.vstack((pc['f3'], pc['f4'], pc['f5'])).T.astype(np.uint8)
        color = np.zeros((xyz.shape[0], 3), dtype=np.uint8)
        #intensity = pc['f6'].astype(np.uint8).reshape(-1,1)
        intensity = pc['scalar_Intensity'].astype(np.uint8).reshape(-1,1)
        labels = np.zeros(xyz.shape[0], dtype=np.uint8)  # No labels for test data
        #labels = pc['scalar_Label'].astype(np.uint8)
    #  Subsample to save space
    sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(xyz, color, labels, grid_size)
    _, sub_intensity = DP.grid_sub_sampling(xyz, features=intensity, grid_size=grid_size)

    sub_colors = sub_colors / 255.0
    sub_intensity = sub_intensity[:,0] / 255.0
    sub_ply_file = join(sub_pc_folder, file_name + '.ply')

    if file_name in train_files:
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_intensity, sub_labels],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class'])
    else:
        write_ply(sub_ply_file, [sub_xyz, sub_colors, sub_intensity],
                  ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity'])

    search_tree = KDTree(sub_xyz, leaf_size=50)
    kd_tree_file = join(sub_pc_folder, file_name + '_KDTree.pkl')
    with open(kd_tree_file, 'wb') as f:
        pickle.dump(search_tree, f)

    if file_name not in train_files:
        proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        proj_save = join(sub_pc_folder, file_name + '_proj.pkl')
        with open(proj_save, 'wb') as f:
            pickle.dump([proj_idx, labels], f)

print("Processing completed.")