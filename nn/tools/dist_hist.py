import numpy as np
import matplotlib.pyplot as plt

# -------------- PARAMETERS -------------- #
results_folder = '../results/results_cifar_none_blur1/'
run_id = 1
label_id = 3
train_test = 'train'
# ---------------------------------------- #

data = np.loadtxt(results_folder + 'run' + str(run_id) + '/' + train_test + '_encoded.txt')
data = data[data[:, 0] == label_id]
print 'Data read'

n = data.shape[0]
data = data[:n / 5]
n = data.shape[0]
dists = np.zeros(n * (n - 1) / 2)
k = 0
for i in range(n):
    for j in range(i + 1, n):
        dists[k] = np.linalg.norm(data[i] - data[j])
        k += 1
print 'Dists calculated'

plt.hist(np.array(dists))
plt.show()
