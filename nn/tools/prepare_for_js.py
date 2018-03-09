import numpy as np
import shutil
import utils
import os
from os.path import join as path_join
import json

# # -------------- PARAMETERS -------------- #
# results_folder = '../../results/results_cifar_label3_rot64_dim3/'
# run_id = 1
# label_id = 0
# train_test = 'train'
# # prefix = 'density_'
# prefix = ''
# save_images = False
#
# output_folder = '../for_js/none/'
# # ---------------------------------------- #

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_cifar_label3_rot64_dim3/'
run_id = 1
label_id = 3
train_test = 'train'
to_load_decoder = False
save_images = False
# prefix = 'density_'
prefix = ''

output_folder = '../for_js/cifar_rot_dim3/'
# ---------------------------------------- #

with open(results_folder + 'params.json') as data_file:
    params = json.load(data_file)

utils.create_dir(output_folder)

label_folder = '%srun%d/%slabel%s%s/' % (results_folder, run_id, prefix, str(label_id), train_test)

# 3D points
shutil.copyfile(path_join(label_folder, 'data.txt'), path_join(output_folder, 'points_3d.txt'))


# cycles
def fix_cycle(edges):
    if edges.shape[1] > 2:
        return edges
    nxt = dict()
    for edge in edges:
        if edge[0] not in nxt.keys():
            nxt[edge[0]] = []
        nxt[edge[0]].append(edge[1])
        if edge[1] not in nxt.keys():
            nxt[edge[1]] = []
        nxt[edge[1]].append(edge[0])

    prev = edges[0, 0]
    cur = edges[0, 1]

    curve = np.zeros((edges.shape[0] + 1), dtype=np.int32)
    curve[0] = prev

    for i in range(1, curve.shape[0]):
        curve[i] = cur
        if nxt[cur][0] != prev:
            prev = cur
            cur = nxt[cur][0]
        else:
            prev = cur
            cur = nxt[cur][1]

    return curve

for i in range(5):
    cycle_fname = label_folder + 'repr_%d.txt' % i
    killer_fname = label_folder + 'kill_%d.txt' % i
    if os.path.exists(cycle_fname):
        edges = np.loadtxt(cycle_fname, dtype=np.int32, skiprows=1)
        edges = fix_cycle(edges)
        np.savetxt(path_join(output_folder, 'cycle_%d.txt' % i), edges, fmt='%d')
        if os.path.exists(killer_fname):
            np.savetxt(path_join(output_folder, 'killer_%d.txt' % i), np.loadtxt(killer_fname, skiprows=1), fmt='%d')


# filtered points
if os.path.exists(label_folder + 'filtered_ids.txt'):
    shutil.copyfile(label_folder + 'filtered_ids.txt', path_join(output_folder, 'filtered_points.txt'))


# images
def norm_images(images):
    images = ((images + 0.5) * 255)
    images = np.maximum(0, images)
    images = np.minimum(255, images)
    return images.astype(np.uint8)


def write_images_uint8(filename, images):
    with open(filename, 'wb') as f:
        f.write(images.tobytes())

data = np.loadtxt(path_join(label_folder, 'data.txt'), skiprows=1)
images = np.load('%s_data/%s_images.npy' % (results_folder, train_test))
labels = np.load('%s_data/%s_labels.npy' % (results_folder, train_test))
if label_id != 'all':
    images = images[labels == label_id]
    labels = labels[labels == label_id]
assert data.shape[0] == images.shape[0]
if save_images:
    write_images_uint8(path_join(output_folder, 'images.bin'), norm_images(images))
np.savetxt(path_join(output_folder, 'labels.txt'), labels, fmt='%d')

properties = {}
if save_images:
    properties['imageSize'] = params['input_shape'],
with open(path_join(output_folder, 'properties.json'), 'w') as f:
    json.dump(properties, f)


# decoded
def load_decoder():
    import main, algo, encoders

    params['save_dir'] = '%srun%d/%s' % (results_folder, run_id, params['save_dir'])
    params['logs_dir'] = '%srun%d/%s' % (results_folder, run_id, params['logs_dir'])
    kwargs = main.extract_kwargs(params)
    encoder = getattr(encoders, params['algo'])
    decode = algo.decoder_wrapper(encoder, params, **kwargs)
    return decode

if to_load_decoder:
    decode = load_decoder()
    batch = 4
    images_dec = []
    for i in range(0, data.shape[0], batch):
        images_dec.append(decode(data[i:i + batch]))
    images_dec = np.concatenate(images_dec)
    write_images_uint8(path_join(output_folder, 'images_decoded.bin'), norm_images(images_dec))


shutil.copyfile(path_join(label_folder, 'plot_pairs.txt.png'), path_join(output_folder, 'plot.png'))
