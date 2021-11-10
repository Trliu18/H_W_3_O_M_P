""" decode message """

import os
import os.path as pth
import matplotlib.pyplot as plt
from omp import *
import numpy as np
import scipy.io
from tqdm import tqdm

if os.path.exists('figure/decode'):
    print('decode folder exists')
else:
    os.makedirs('figure/decode')


if not (pth.exists('x1.npy') and pth.exists('x2.npy') and pth.exists('x3.npy')):
    # read in the data
    mat = scipy.io.loadmat('omp_test.mat')
    A1 = mat['A1']
    A2 = mat['A2']
    A3 = mat['A3']
    y1 = mat['y1']
    y2 = mat['y2']
    y3 = mat['y3']

    print('A1 size: {}'.format(A1.shape))
    print('A2 size: {}'.format(A2.shape))
    print('A3 size: {}'.format(A3.shape))
    print('Y1 size: {}'.format(y1.shape))
    print('Y2 size: {}'.format(y2.shape))
    print('Y3 size: {}'.format(y3.shape))


if __name__ == "__main__":
    if not (pth.exists('x1.npy') and pth.exists('x2.npy') and pth.exists('x3.npy')):
        # read in data
        x1 = OMP(A1, y1, iteration=10000, acc_tolerance=0.001)
        x2 = OMP(A2, y2, iteration=10000, acc_tolerance=0.001)
        x3 = OMP(A3, y3, iteration=10000, acc_tolerance=0.001)
        
        print('x1 size: {}'.format(x1.shape))
        print('x2 size: {}'.format(x2.shape))
        print('x3 size: {}'.format(x3.shape))
        
        np.save('x1.npy', x1)
        np.save('x2.npy', x2)
        np.save('x3.npy', x3)
    
    x1 = np.load('x1.npy')
    x1 = x1.T
    x1 = x1.reshape(160, 90)
    x1 = x1.T
    p1 = plt.figure()
    plt.imshow(x1, cmap='gray')
    plt.title('the original signal x1')
    plt.show()
    p1.savefig('figure/decode/x1.png')

    x2 = np.load('x2.npy')
    x2 = x2.T
    x2 = x2.reshape(160, 90)
    x2 = x2.T
    p2 = plt.figure()
    plt.imshow(x2, cmap='gray')
    plt.title('the original signal x2')
    plt.show()
    p2.savefig('figure/decode/x2.png')

    x3 = np.load('x3.npy')
    x3 = x3.T
    x3 = x3.reshape(160, 90)
    x3 = x3.T
    p3 = plt.figure()
    plt.imshow(x3, cmap='gray')
    plt.title('the original signal x3')
    plt.show()
    p3.savefig('figure/decode/x3.png')
