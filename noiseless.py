""" noiseless OMP """

import sys
import os
import os.path as pth
import matplotlib.pyplot as plt
from omp import *
import numpy as np
from tqdm import tqdm


if os.path.exists('figure/noiseless'):
    print('noiseless folder exists')
else:
    os.makedirs('figure/noiseless')


if __name__ == "__main__":

    for N in [20, 50, 100]:
        file_path = "no_noise_{}.npy".format(N)
        if N == 20: STEP_SIZE = 3
        elif N == 50: STEP_SIZE = 5
        elif N == 100: STEP_SIZE = 10
        Ms = list(range(1, int(N * 1.5), STEP_SIZE))
        Ss = list(range(1, N, STEP_SIZE))
        
        if not os.path.exists("no_noise_{}.npy".format(N)):
            transition_map = np.zeros((len(Ms), len(Ss)))
            for s in Ss:
                for M in Ms:
                    res_list = repeated_expermients(s, M, N, repeated_time=2000, noise_sigma=0, know_s=False)
                    res_list = [1 if err < 1e-5 else 0 for err in res_list]    # 1 if error rate < 1e-3 else 0
                    success_rate = sum(res_list) / len(res_list)
                    print("N:{}, M: {}, s: {}, success rate: {}".format(N, M, s, success_rate))
                    transition_map[M // STEP_SIZE, s // STEP_SIZE]  = success_rate
            print(transition_map)
            np.save(file_path, transition_map)
            print("save to path: {}".format(file_path))
            
        img_name = 'no_noise_{}'.format(N)
        image = np.load('{}.npy'.format(img_name))
        image = image.T
        plt.figure()
        plt.imshow(image)

        xtick = range(1, int(N*1.5), STEP_SIZE)
        ytick = range(1, N, STEP_SIZE)

        plt.xticks(np.arange(len(xtick)), xtick)
        plt.yticks(np.arange(len(ytick)), ytick)
        plt.xlabel('M', fontsize=12)
        plt.ylabel('s max', fontsize=12)
        plt.title('Noiseless, N = {}'.format(N))
        plt.colorbar()
        plt.savefig('figure/noiseless/{}.png'.format(img_name))
        print("save to path: {}".format('figure/noiseless/{}.png'.format(img_name)))
