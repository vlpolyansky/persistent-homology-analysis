import numpy as np
from pylab import *
import subprocess as sp
import os

# -------------- PARAMETERS -------------- #
results_folder = '../res_iter_1/'
label_id = 1
# ---------------------------------------- #


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    buf = buf[:, :, :3]
    # buf = np.copy(buf, order='C')
    return buf


folder = results_folder + ('_distances/label%d/' % label_id)
files = os.listdir(folder)

is_open = False

for file in sorted(files):
    print file
    # if file.startswith('00100'):
    #     break
    mat = np.loadtxt(folder + file)

    fig, ax = subplots()

    p = ax.pcolor(range(0, mat.shape[0] + 1), range(0, mat.shape[1] + 1), mat, cmap=cm.RdBu, vmin=0, vmax=0.01)
    cb = fig.colorbar(p)
    title(file)
    arr = fig2data(fig)
    close(fig)

    if not is_open:
        command = ['ffmpeg',
                   '-y',  # (optional) overwrite output file if it exists
                   '-f', 'rawvideo',
                   '-vcodec', 'rawvideo',
                   '-s', '%dx%d' % (arr.shape[0], arr.shape[1]),
                   '-pix_fmt', 'rgb24',
                   '-r', '15',  # frames per second
                   '-i', '-',  # The imput comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   # '-vcodec', 'mpeg',
                   'dist_animation.mp4']

        pipe = sp.Popen(command, stdin=sp.PIPE)
        is_open = True

    pipe.stdin.write(arr.tostring())
