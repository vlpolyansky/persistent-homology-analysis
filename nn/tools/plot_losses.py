import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_earth_v3_4d/'
num_runs = 10
# ---------------------------------------- #

runs = range(1, num_runs + 1)
for i in runs:
    print 'Run', i
    event_acc = EventAccumulator('%srun%d/logs/' % (results_folder, i))
    event_acc.Reload()
    _, times, losses = zip(*event_acc.Scalars('train/loss/main'))
    plt.plot(times, losses)

plt.legend(map(str, runs))

plt.show()
