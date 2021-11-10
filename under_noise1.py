# under noise OMP know s 
import os
import matplotlib.pyplot as plt
from omp import *
import numpy as np
from tqdm import tqdm

if os.path.exists('figure/under_noise1'):
    print('under_noise1 folder exists')
else:
    os.makedirs('figure/under_noise1')


if __name__ == "__main__":
    
    for N in [20, 50, 100]:
        if N == 20: STEP_SIZE = 3
        elif N == 50: STEP_SIZE = 5
        elif N == 100: STEP_SIZE = 10
        Ms = list(range(1, int(N*1.5), STEP_SIZE))
        s = N // 10    # we choose s = N/10 (<< N) as the number of support
        noise_range = [1e-2,  1e-4]
        transition_map = np.zeros((len(Ms), len(noise_range)))
        
        for M in Ms:
            for n in noise_range:
                res_list = repeated_expermients(s, M, N, noise_sigma=n, know_s=True)
                res_list = [1 if err < 1e-3 else 0 for err in res_list]    # 1 if error rate < 1e-3 else 0
                success_rate = sum(res_list) / len(res_list)
                print("N:{}, M: {}, noise:{}  success rate: {}".format(N, M, n, success_rate))
                transition_map[M//STEP_SIZE, noise_range.index(n)] = success_rate

        print(transition_map)
        file_path = "under_noise1_{}.npy".format(N)
        np.save("under_noise1_{}.npy".format(N), transition_map)
        img_name = 'under_noise1_{}'.format(N)
        image = np.load('{}.npy'.format(img_name))
        image = image.T
        
        plt.figure()
        plt.imshow(image)
            
        xtick = range(1, int(N*1.5), STEP_SIZE)
        ytick = list(noise_range)
        
        plt.xticks(np.arange(len(xtick)), xtick)
        plt.yticks(np.arange(len(ytick)), ytick)
        plt.xlabel('M', fontsize=12)
        plt.ylabel('noise', fontsize=12)
        plt.title('Noise, know s, N = {}'.format(N))
        plt.colorbar()
        plt.savefig('figure/under_noise1/{}.png'.format(img_name))


