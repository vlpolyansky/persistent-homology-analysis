import sys
import numpy as np
from scipy.signal import convolve2d
from matplotlib.pyplot import *

# -------------- PARAMETERS -------------- #
results_folder = '../results_label6_rotation/'
label_id = 6
train_test = 'train'
# ---------------------------------------- #

images_fname = '%s_data/%s_images.npy' % (results_folder, train_test)
labels_fname = '%s_data/%s_labels.npy' % (results_folder, train_test)

images = np.load(images_fname)
labels = np.load(labels_fname)
if label_id != 'all':
    images = images[labels == label_id]

images_conv = []
for image in images:
    tmp = convolve2d(image, np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]) / 9.0, mode='valid')
    images_conv.append(tmp)
images = np.array(images_conv)

images = np.reshape(images, [images.shape[0], -1])

with open('simple.txt', 'w') as f:
    f.write('%d %d\n' % (images.shape[0], images.shape[1]))
    np.savetxt(f, images)
