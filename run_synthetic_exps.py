import numpy as np
import os
from main_synthetic_semi import repeat

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

n_list   = np.arange(10,105,10)
exp_list = ['linear', 'nonlinear', 'PCA', 'random']

for exp in exp_list:
    Res_array = np.zeros((len(n_list), 8))
    log = open('log_synthetic.txt', 'a')
    print(exp, file=log)
    for i,n in enumerate(n_list):
        print(n, file=log)
        Res = repeat(N=20, n=n, exp=exp)
        print(Res, file=log)
        Res_array[i] = Res
    log.close()

    np.save('synthetic_result/{}_res500-500.npy'.format(exp), Res_array)


