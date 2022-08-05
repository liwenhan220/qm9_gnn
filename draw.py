import matplotlib.pyplot as plt
import numpy as np
import sys

TARGET = ['A','B','C','mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv','u0_atom','u298_atom','h298_atom','g298_atom']

# Usage: python draw.py [target] [model_dir] [graph]
# target: 0 to 18
# model_dir: Your model directory
# graph: (loss/acc/tacc)

try:
    if len(sys.argv) != 4:
        raise NameError('')

    target = int(sys.argv[1])
    path = sys.argv[2]
    file = path + '/' +'{}_{}.npy'.format(sys.argv[3], target)

    ls = np.load(file, allow_pickle=True)

    if sys.argv[3] != 'loss':
        plt.ylim(top = 1.1)
        plt.ylim(bottom = -0.1)


    plt.plot(np.arange(len(ls)), ls)
    plt.title('{} of {}'.format(sys.argv[3], TARGET[target]))
    plt.xlabel('Num Iters')
    plt.ylabel('{}'.format(sys.argv[3]))
    plt.show()

except NameError:
    print('\n')
    print('Usage: python draw.py [target] [model_dir] [graph]')
    print('\n')
    print('target: 0 to 18')
    print('model_dir: Your model directory')
    print('graph: (loss/acc/tacc)')
