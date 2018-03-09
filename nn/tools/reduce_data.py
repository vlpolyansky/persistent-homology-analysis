import numpy as np
import shutil
import utils
import os
from os.path import join as path_join

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_all_rotation_3d/'
train_test = 'train'
# prefix = 'density_'
prefix = ''
part = 0.1

# ---------------------------------------- #

data_folder = path_join(results_folder, '_data')
old_data_folder = path_join(results_folder, '__data')
shutil.move(data_folder, old_data_folder)
utils.create_dir(data_folder)

images = np.load(path_join(old_data_folder, '%s_images.npy' % train_test))
labels = np.load(path_join(old_data_folder, '%s_labels.npy' % train_test))

n = images.shape[0]
k = int(n * part)
choice = np.random.choice(n, k, replace=False)

images = images[choice]
labels = labels[choice]

np.save(path_join(data_folder, '%s_images.npy' % train_test), images)
np.save(path_join(data_folder, '%s_labels.npy' % train_test), labels)
