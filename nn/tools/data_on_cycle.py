import numpy as np
import subprocess as sp
import matplotlib.pyplot as plt

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_cmu_walk/'
run_id = 1
label_id = 0
train_test = 'train'
repr_file = 'repr_0.txt'
output_name = 'cycle.mp4'
time_sec = 8
# ---------------------------------------- #

label_folder = '%srun%d/label%d%s/' % (results_folder, run_id, label_id, train_test)

images = np.load('%s_data/%s_images.npy' % (results_folder, train_test))
images = ((images + 0.5) * 255).astype(np.uint8)
labels = np.load('%s_data/%s_labels.npy' % (results_folder, train_test))
if label_id != 'all':
    images = images[labels == label_id, :]

edges = np.loadtxt(label_folder + repr_file, dtype=int, skiprows=1)
assert edges.shape[1] == 2

nxt = dict()
for edge in edges:
    if edge[0] not in nxt.keys():
        nxt[edge[0]] = []
    nxt[edge[0]].append(edge[1])
    if edge[1] not in nxt.keys():
        nxt[edge[1]] = []
    nxt[edge[1]].append(edge[0])

frames_cnt = len(nxt)
print 'Number of frames:', frames_cnt

command = ['ffmpeg',
           '-y',  # (optional) overwrite output file if it exists
           '-f', 'rawvideo',
           '-vcodec', 'rawvideo',
           '-s', '%dx%d' % (images.shape[1], images.shape[2]),  # size of one frame
           '-pix_fmt', 'gray' if len(images.shape) == 3 or images.shape[3] == 1 else 'rgb24',
           '-r', str(frames_cnt * 1.0 / time_sec),  # frames per second
           '-i', '-',  # The imput comes from a pipe
           '-an',  # Tells FFMPEG not to expect any audio
           # '-vcodec', 'mpeg',
           output_name]
pipe = sp.Popen(command, stdin=sp.PIPE)

frames = []

prev = edges[0, 0]
cur = edges[0, 1]
first = cur
while True:
    frame = np.expand_dims(images[cur], 2)
    frames.append(images[cur])
    pipe.stdin.write(frame.tostring())
    if nxt[cur][0] != prev:
        prev = cur
        cur = nxt[cur][0]
    else:
        prev = cur
        cur = nxt[cur][1]
    if cur == first:
        break
pipe.stdin.close()

frames_image = np.concatenate(frames, 1)
plt.imshow(frames_image)
plt.imsave('frames.png', frames_image)
