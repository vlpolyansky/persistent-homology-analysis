import sys
import numpy as np
import utils


def main():
    argv = sys.argv  # filename label
    in_name = argv[1]
    label = argv[2]

    out_name = 'data.txt'   # output file

    table = np.loadtxt(in_name)
    if label == 'all':
        data = table[:, 1:]
    else:
        data = table[table[:, 0] == int(label), 1:]

    with open(out_name, 'w') as f:
        if utils.has_arg('--dim'):
            f.write('%d %d\n' % (data.shape[0], data.shape[1]))
        else:
            f.write('%d\n' % data.shape[0])
        np.savetxt(f, data)


if __name__ == '__main__':
    main()
