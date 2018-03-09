import numpy as np
import shutil
import utils
import os
from os.path import join as path_join
import json

# -------------- PARAMETERS -------------- #
results_folder = '../../results/results_earth_v3_4d/'
run_id = 1
label_id = 0
repr_file = 'repr_0.txt'
train_test = 'train'
# prefix = 'density_'
prefix = ''
save_images = True

output_folder = '../../for_js/earth_4d_1_0/'
# ---------------------------------------- #

with open(results_folder + 'params.json') as data_file:
    params = json.load(data_file)

utils.create_dir(output_folder)

label_folder = '%srun%d/%slabel%s%s/' % (results_folder, run_id, prefix, str(label_id), train_test)

# 3D points
points_3d = np.load('cycle_data_post.npy')[:, 1:]
vertices = np.loadtxt('cycle_vertices.txt').astype(np.int64)
print vertices.shape
np.savetxt(path_join(output_folder, 'points_3d.txt'), points_3d, header=str(points_3d.shape[0]))
np.savetxt(path_join(output_folder, 'filtered_points.txt'), np.array(range(0, points_3d.shape[0])), fmt='%d')


# cycles
cycle_fname = label_folder + repr_file
if os.path.exists(cycle_fname):
    edges = np.loadtxt(cycle_fname, dtype=np.int32, skiprows=1)
    v_map = dict()
    for i, v in enumerate(vertices):
        v_map[v] = i
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            edges[i, j] = v_map[edges[i, j]]
    np.savetxt(path_join(output_folder, 'cycle_0.txt'), edges, fmt='%d')


# images
def norm_images(images):
    images = ((images + 0.5) * 255)
    images = np.maximum(0, images)
    images = np.minimum(255, images)
    return images.astype(np.uint8)


def write_images_uint8(filename, images):
    with open(filename, 'wb') as f:
        f.write(images.tobytes())

images = np.load('%s_data/%s_images.npy' % (results_folder, train_test))
labels = np.load('%s_data/%s_labels.npy' % (results_folder, train_test))
if label_id != 'all':
    images = images[labels == label_id]
    labels = labels[labels == label_id]
images = images[vertices]
assert points_3d.shape[0] == images.shape[0]
if save_images:
    write_images_uint8(path_join(output_folder, 'images.bin'), norm_images(images))
np.savetxt(path_join(output_folder, 'labels.txt'), labels, fmt='%d')

properties = {}
if save_images:
    properties['imageSize'] = params['input_shape']
with open(path_join(output_folder, 'properties.json'), 'w') as f:
    json.dump(properties, f)


shutil.copyfile(path_join(label_folder, 'plot_pairs.txt.png'), path_join(output_folder, 'plot.png'))
