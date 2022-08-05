import matplotlib.pyplot as plt
import numpy as np
import sys

TARGET = ['A','B','C','mu','alpha','homo','lumo','gap','r2','zpve','u0','u298','h298','g298','cv','u0_atom','u298_atom','h298_atom','g298_atom']

## Usage: python draw.py [target] [model path] [loss/acc]
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
