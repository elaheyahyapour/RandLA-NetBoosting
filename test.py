from glob import glob
import shutil

for fp in glob('/media/ellie/Linux/Clone/RandLA-Net/Data/dataset/sequences/**'):

    # /media/ellie/Linux/Clone/RandLA-Net/Data/dataset/sequences/00/labels
    # /media/ellie/Github/RandLA-Net/data/semantic_kitti/dataset/sequences/00

    new_n = fp.split('/')
    dir_counter = new_n[-1]
    orig_dir = fp + '/labels'
    print(orig_dir)
    dest_dir = '/media/ellie/Github/RandLA-Net/data/semantic_kitti/dataset/sequences/' + dir_counter+'/labels'
    print(dest_dir)
    shutil.copytree(orig_dir,dest_dir)