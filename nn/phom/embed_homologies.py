import numpy as np
from sklearn.manifold import MDS
from pylab import *


def main():
    distances = np.loadtxt('homology_distances.txt')
    mds = MDS(n_components=2, dissimilarity='precomputed')
    coordinates = mds.fit_transform(distances)
    np.savetxt('embedding_2d.txt', coordinates)
    plot(coordinates[:, 0], coordinates[:, 1], '.')
    for i in range(coordinates.shape[0]):
        text(coordinates[i, 0], coordinates[i, 1], i + 1)
    # show()
    savefig('embedding_2d.png')

if __name__ == '__main__':
    main()
