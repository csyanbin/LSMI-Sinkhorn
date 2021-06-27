import numpy as np
import random
import sys
import torch
import os


os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True


gpu = True
if gpu:
    from utils_gpu import *
else:
    from utils_cpu import *


def PCA(X, k=2):
    """X: NxD"""
    n,d = X.shape
    X = X - X.mean(axis=0)
    CX = np.dot(X.T,X) / (n-1)
    eigval, eigvec = np.linalg.eigh(CX)
    eigvec = eigvec[:, -k:]

    X_pca = np.dot(X, eigvec)

    return X_pca


def run(seed=0, n=10, exp='linear'):
    # fix the random seed
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    D1 = 5 # X-dim
    D2 = 5 # Y-dim
    
    NN = 10000   # large number to estimate True SMI
    Nx = n+500  
    Ny = n+500   
    b  = 500   
    
    print("Experiment{}:{}".format(seed,exp))
    print("Nlarge:", NN, 'n_pair:', n, 'N_x:', Nx-n, 'N_y:',Ny-n)
    
    
    if exp=='random':
        X = np.random.randn(NN, D1)
        Y = np.random.randn(NN, D2)
    
    elif exp=='linear':
        X = np.random.randn(NN, D1)
        noise = np.random.randn(NN,D1) / 100.
        Y = 0.5*X + noise
    
    elif exp=='nonlinear':
        X = np.random.randn(NN,D1)
        Y = np.sin(X)
    
    elif exp=='PCA':
        D1 = 10 # 10-->5 dimmension reduction
        X = np.random.randn(NN,D1)
        Y = PCA(X, k=5)
    
    
    ## Hyper-parameters
    beta_list = [0.2,0.4,0.6,0.8,1.0]
    lam_list  = [0.1,0.01,0.001,0.0001]
    
    
    ## ---- LSMI full: Ground truth ---- ##
    MI_cv = np.zeros(len(lam_list))
    for k in range(len(lam_list)):
        lam = lam_list[k]
        MI_pair, MI_cv_tmp = SMI_pair_CV(X.T, Y.T, b=b, lam=lam)
        MI_cv[k] = MI_cv_tmp
    lam_opt_gt = lam_list[MI_cv.argmin()]
    print("best lambda for LSMI full:", lam_opt_gt)
    MIh_pair_full = SMI_pair(X.T, Y.T, b=b, lam=lam_opt_gt)
    
    
    ## ---- LSMI (only paired): baseline ---- ##
    MI_cv = np.zeros(len(lam_list))
    for k in range(len(lam_list)):
        lam = lam_list[k]
        MI_pair, MI_cv_tmp = SMI_pair_CV(X.T[:,0:n], Y.T[:,0:n], b=b, lam=lam)
        MI_cv[k] = MI_cv_tmp
    lam_opt = lam_list[MI_cv.argmin()]
    print("best lambda for LSMI (only paired):", lam_opt)
    MIh_pair = SMI_pair(X.T[:,0:n], Y.T[:,0:n], b=b, lam=lam_opt)

    ## ---- LSMI (opt): only paired, but use optimal lambda from LSMI full
    MIh_pair_gt = SMI_pair(X.T[:,0:n], Y.T[:,0:n], b=b, lam=lam_opt_gt)
    
    
    ## ---- LSMI-Sinkhorn: proposed ---- ##
    MI_cv = np.zeros((len(beta_list), len(lam_list)))
    ind_semi = np.array(range(n,Ny))
    idx = ind_semi
    for i in range(len(beta_list)):
      for k in range(len(lam_list)):
        beta = beta_list[i]
        lam = lam_list[k]
        epsilon = 1.0
        PI, MIs,MI_cv_tmp = SMI_sinkhorn_semi_CV_main(X.T[:,0:n], Y.T[:,0:n], X.T[:,n:Nx], Y.T[:,idx], n_iter=5, b=b, beta=beta, epsilon=epsilon, lam=lam, warm=False)
        MI_cv[i,k] = MI_cv_tmp
    ix = np.unravel_index(np.argmin(MI_cv, axis=None), MI_cv.shape)
    beta_opt = beta_list[ix[0]]
    lam_opt = lam_list[ix[1]]
    
    print('LSMI-Sinkhorn, best beta:', beta_opt, 'best lambda:', lam_opt)
    epsilon = 0.3
    PI, MIs, MIh_pair_semi = SMI_sinkhorn_semi(X.T[:, 0:n], Y.T[:, 0:n], X.T[:, n:Nx], Y.T[:, idx], n_iter=10, b=b, beta=beta_opt, epsilon=epsilon, lam=lam_opt, warm=False)
    
    print(MIh_pair_full.shape,MIh_pair_full)
    np.set_printoptions(precision=2)
    if np.isnan(MIh_pair_semi):
        print('Nan encountered in log-sinkhorn algorithm, ignore this result')
    else:
        print('LSMI (full):{}, LSMI:{}, LSMI (opt):{}, LSMI-Sinkhorn:{}'.format(MIh_pair_full, MIh_pair, MIh_pair_gt, MIh_pair_semi))
    print('-----------------------------')
    print()

    return MIh_pair_full, MIh_pair, MIh_pair_gt, MIh_pair_semi


def repeat(N=20,n=10,exp='linear'):
    # Repeat the experiment Several times
    N_TIMES = N
    seed_list = np.arange(1, N_TIMES+1)
    full_SMI_list = []
    pair_SMI_list = []
    pair_gt_SMI_list = []
    semi_SMI_list = []
    for seed in seed_list:
        MIh_pair_full, MI_pair, MIh_pair_gt, MIh_pair_semi = run(seed, n, exp)
        full_SMI_list.append(MIh_pair_full)
        pair_SMI_list.append(MI_pair)
        pair_gt_SMI_list.append(MIh_pair_gt)
        semi_SMI_list.append(MIh_pair_semi)
    
    SMI_full_mean = np.array(full_SMI_list).mean()
    SMI_full_std  = np.array(full_SMI_list).std()
    
    SMI_mean = np.array(pair_SMI_list).mean()
    SMI_std  = np.array(pair_SMI_list).std()
    
    SMI_opt_mean = np.array(pair_gt_SMI_list).mean()
    SMI_opt_std  = np.array(pair_gt_SMI_list).std()
    
    semi_SMI_list = np.array(semi_SMI_list)
    semi_SMI_list = semi_SMI_list[~np.isnan(semi_SMI_list)]
    SMI_sink_mean = np.array(semi_SMI_list).mean()
    SMI_sink_std  = np.array(semi_SMI_list).std()
    
    print('Final mean and std:')
    print('LSMI_full: {:.4f} +/- {:.4f}'.format(SMI_full_mean, SMI_full_std))
    print('LSMI: {:.4f} +/- {:.4f}'.format(SMI_mean, SMI_std))
    print('LSMI_Opt: {:.4f} +/- {:.4f}'.format(SMI_opt_mean, SMI_opt_std))
    print('LSMI_Sink: {:.4f} +/- {:.4f}'.format(SMI_sink_mean, SMI_sink_std))
    print('-----')
    print()
    
    return SMI_full_mean, SMI_full_std, SMI_mean, SMI_std, SMI_opt_mean, SMI_opt_std, SMI_sink_mean, SMI_sink_std


## N: number of repeats, n:number of labelled examples
#repeat(N=50, n=50, exp='nonlinear')


