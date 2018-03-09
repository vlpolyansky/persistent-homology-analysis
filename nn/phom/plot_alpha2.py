import sys
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
import utils


diag = np.loadtxt(sys.argv[1])

if utils.has_arg('--nosqrt'):
    diag[:, 2:] = diag[:, 2:] ** 0.5

annotate_cnt = 4
if utils.has_arg('--nonums'):
    annotate_cnt = 0

no_cut = False
if utils.has_arg('--nocut'):
    no_cut = True
if utils.has_arg('--stdlim'):
    limit = 0.25
    no_cut = True
else:
    tmp = utils.get_arg('--lim')
    if tmp:
        limit = float(tmp)
        no_cut = True
    else:
        if diag[diag[:, 3] != Inf, 3].size == 0:
            limit = 1.
        else:
            limit = max(max(diag[:, 2]), max(diag[diag[:, 3] != Inf, 3]))
margin = limit * 0.01
diag[diag[:, 3] == Inf, 3] = limit + margin * 0.9

if no_cut:
    cut_x = cut_y = cut = [1., 0.]
else:
    srt = sort(diag[:, 2])
    cut_x = [srt[0] + margin, srt[1] - margin]
    for i in range(1, srt.shape[0] - 1):
        a = srt[i]
        b = srt[i + 1]
        if b - a > cut_x[1] - cut_x[0]:
            cut_x = [a + margin, b - margin]

    srt = sort(diag[:, 3])
    cut_y = [srt[0] + margin, srt[1] - margin]
    for i in range(1, srt.shape[0] - 1):
        a = srt[i]
        b = srt[i + 1]
        if b - a > cut_y[1] - cut_y[0]:
            cut_y = [a + margin, b - margin]

    cut = [0, 0]
    cut[0] = max(cut_x[0], cut_y[0])
    cut[1] = min(cut_x[1], cut_y[1])
    cut_x = cut
    cut_y = cut


colors = ['black', 'blue', 'red', 'purple', 'cyan']
# figs = ['o', 's', '^', '^']
figs = ['.', '.', '.', '.', '.']
max_dim = 0

if no_cut or cut[0] >= cut[1] or cut[1] - cut[0] < 0.33 * limit:
    fig, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 10))
    ax.set_xlim(0 - margin, limit + margin)
    ax.set_ylim(0 - margin, limit + margin)

    ax.plot([0, limit], [0, limit], color='gray')
    for i, (c, f) in enumerate(zip(colors, figs)):
        data = diag[diag[:, 0] == i, 2:]
        if data.size > 0:
            max_dim = i
        ax.plot(data[:, 0], data[:, 1], f, color=c, mfc='none')

    if annotate_cnt > 0:
        if utils.has_arg('--dim1'):
            data = diag[diag[:, 0] == 1, :]
        else:
            data = diag[diag[:, 0] != 0, :]
        data = data[:annotate_cnt, :]
        for i in range(data.shape[0]):
            ax.annotate(str(int(data[i, 1])), (data[i, 2], data[i, 3]))

else:
    fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 10))
    # fig.tight_layout()

    # ASSERTIONS !!!
    assert not np.any(np.logical_and(diag[:, 2] > cut_x[0], diag[:, 2] < cut_x[1]))
    assert not np.any(np.logical_and(diag[:, 3] > cut_y[0], diag[:, 3] < cut_y[1]))

    for i in range(2):
        ax[1, i].set_ylim(0 - margin, cut_y[0])
        ax[0, i].set_ylim(cut_y[1], limit + margin)

        ax[0, i].spines['bottom'].set_visible(False)
        ax[0, i].xaxis.tick_top()
        ax[0, i].tick_params(labeltop='off')
        ax[1, i].spines['top'].set_visible(False)
        ax[1, i].xaxis.tick_bottom()

    for i in range(2):
        ax[i, 0].set_xlim(0 - margin, cut_x[0])
        ax[i, 1].set_xlim(cut_x[1], limit + margin)

        ax[i, 1].spines['left'].set_visible(False)
        ax[i, 1].yaxis.tick_right()
        ax[i, 1].tick_params(labeltop='off')
        ax[i, 0].spines['right'].set_visible(False)
        ax[i, 0].yaxis.tick_left()

    for x in range(2):
        for y in range(2):
            ax[x, y].plot([0, limit], [0, limit], color='gray')
    for i, (c, f) in enumerate(zip(colors, figs)):
        data = diag[diag[:, 0] == i, 2:]
        if data.size > 0:
            max_dim = i
        for x in range(2):
            for y in range(2):
                ax[x, y].plot(data[:, 0], data[:, 1], f, color=c, mfc='none')

    if annotate_cnt > 0:
        for x in range(2):
            for y in range(2):
                if utils.has_arg('--dim1'):
                    data = diag[diag[:, 0] == 1, :]
                else:
                    data = diag[diag[:, 0] != 0, :]
                data = data[:annotate_cnt, :]
                for i in range(data.shape[0]):
                    ax[x, y].annotate(str(int(data[i, 1])), (data[i, 2], data[i, 3]))

legend(['y=x', '0-dim', '1-dim', '2-dim', '3-dim', '4-dim'][:max_dim + 2], loc=4)

label_str = utils.get_arg('--label')
if label_str:
    title(label_str + (" [!]" if limit + margin <= max(diag[:, 3]) else ""))

savefig('plot_' + sys.argv[1] + '.png')
# show()
# fig.savefig('tmp.png')
