import numpy as np
import torch

from sinkhorn_stab import sinkhorn_stabilized

sinkhorn_algorithm = 'stable' # own:we implement sinkhorm ourselves, stable:use log-stabilized sinkhorn

def euclidean_distance(x, y):
    """ x:nxd, y:mxd
        dist: nxm
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    
    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    dist = torch.pow(x - y, 2).sum(2)

    dist[torch.isnan(dist)] = 0
    dist = dist + 1e-16

    return dist

def compmedDist(X):
    """ X: NxD
        Compute kernel width using median heuristic
    """
    n = X.shape[0]
    if n>1000:
        index = torch.randperm(n)
        X = X[index[0:1000],:]
        n = 1000

    dists = euclidean_distance(X,X)
    dists = dists-torch.tril(dists)
    
    dists = dists.view(n**2,1)
    sigma = torch.sqrt(0.5*torch.median(dists[dists>0]))

    return sigma


def kernel_gaussian(X1, X2, sigma):
    """ X1:dxn1, X2:dxn2
        K: n1xn2
    """
    n1 = X1.shape[1]
    n2 = X2.shape[1]
    
    dist = euclidean_distance(X1.t(), X2.t())
    K = torch.exp(-dist / (2*sigma.pow(2)))

    return K


def perform_sinkhorn(C,epsilon,mu,nu,a,warm=False,niter=100,tol=10e-10):
    """self implemented sinkhorn algorithm"""
    if not warm:
        a = torch.ones((C.shape[0],1)) / C.shape[0]
        a = a.cuda()
    
    K = torch.exp(-C/epsilon)

    #Err = torch.zeros((niter,2)).cuda()
    Err = torch.zeros((niter,2)).cuda()
    for i in range(niter):
        b = nu/torch.mm(K.t(), a)
        if i%5==0:
            Err[i,0] = torch.norm(a*(torch.mm(K, b)) - mu, p=1)

        a = mu / torch.mm(K, b)
        
        if i%5==0:
            Err[i,1] = torch.norm(b*(torch.mm(K.t(), a)) - nu, p=1)

        #print(Err[i,:])
        #print(i)
        if (Err[i,:]).max() < tol:
            break

    PI = torch.mm(torch.mm(torch.diag(a[:,-1]),K), torch.diag(b[:,-1]))

    return PI,a,b,Err


def LSMI_PI(Kx, Ky, Hh, PI, lam):
    """ Kx:bxn1, Ky:bxn2"""
    n1,n2 = Kx.shape[1], Ky.shape[1]
    b     = Kx.shape[0]

    Phi = Kx * torch.mm(Ky, PI.t())
    hh = Phi.sum(1) # b
    #Hh = torch.mm(Kx, Kx.t()) * torch.mm(Ky, Ky.t()) / (n1*n2) # bxb

    Hh = Hh + lam*torch.eye(b).cuda()
    alphah = torch.mm(torch.inverse(Hh), hh.view(-1,1))
    
    Kxa = torch.mm(Kx.t(), torch.diag(alphah.view(-1)))
    C = -torch.mm(Kxa, Ky)

    Cc = torch.mm(Kx.t(), torch.diag(alphah.view(-1)))
    Cc = torch.mm(Cc, Ky)

    MIh0 = 0.5*((Cc-1)*(Cc-1)).sum() / (n1*n2)

    return alphah, MIh0, C


def LSMI_PI_semi(K1, K2, Kx, Ky, Hh, PI, beta, lam, loss=False):
    """ K1:bxn, Kx:bxn1, Ky:bxn2 """
    n,b = K1.shape[1], K1.shape[0]
    n1 = Kx.shape[1]
    n2 = Ky.shape[1]

    Kxt = torch.cat((K1,Kx), dim=1) # bx(n1+n2)
    Kyt = torch.cat((K2,Ky), dim=1) # bx(n1+n2)

    h = (K1*K2).sum(1) / n  # b
    Phi = Kx * torch.mm(Ky, PI.t()) # b
    hh = Phi.sum(1) # b
    #Hh = torch.mm(Kxt, Kxt.t()) * torch.mm(Kyt, Kyt.t()) / ((n1+n)*(n2+n)) # bxb

    Hh = Hh + lam*torch.eye(b).cuda()
    hb = beta*h + (1-beta)*hh
    alphah = torch.mm(torch.inverse(Hh), hb.view(-1,1)) # only bxb inverse, fast
    #alphah = torch.Tensor(alphah.get()).cuda()

    Kxa = torch.mm(Kx.t(), torch.diag(alphah.view(-1)))
    C = -torch.mm(Kxa, Ky)

    Cc = torch.mm(Kxt.t(), torch.diag(alphah.view(-1)))
    Cc = torch.mm(Cc, Kyt)

    MIh0 = 0.5*((Cc-1)*(Cc-1)).sum()/((n+n1)*(n+n2))
    #MIh0 = torch.mm(alphah.t(), hb.view(-1,1)) / 2 - 0.5

    ## Return part of loss value w.r.t alpha only
    if loss:
        loss_value = 0.5*torch.mm(torch.mm(alphah.t(),Hh), alphah) - beta*torch.mm(alphah.t(), h.view(-1,1))
        return alphah, MIh0, C, loss_value

    return alphah, MIh0, C


def SMI_sinkhorn(X, Y, n_iter=20, b=200, epsilon=0.1, lam=0.0001, warm=False):
    """ Unsupervised SMI-sinkhorn Algorithm"""
    X = torch.Tensor(X).cuda()
    Y = torch.Tensor(Y).cuda()
    n1,n2 = X.shape[1], Y.shape[1]
    
    b = min(b, min(n1,n2))
    index1 = torch.randperm(n1)
    index2 = torch.randperm(n2)
    Xb = X[:, index1[0:b]]
    Yb = Y[:, index2[0:b]]
    
    sigma1 = compmedDist(X.t())
    sigma2 = compmedDist(Y.t())
    Kx = kernel_gaussian(Xb, X, sigma1)
    Ky = kernel_gaussian(Yb, Y, sigma2)
    ## Precompute H since it does not depend on PI or alpha
    Hh = torch.mm(Kx, Kx.t()) * torch.mm(Ky, Ky.t()) / (n1*n2) # bxb
    
    del X; del Y


    PI = torch.rand(n1,n2).cuda()
    PI = PI/PI.sum() # normalize

    mu = (torch.ones((n1,))/n1).cuda()
    nu = (torch.ones((n2,))/n2).cuda()
    
    MIh = torch.zeros(n_iter)
    tol = 1e-9
    for iter in range(n_iter):
        PI_prev = PI
        alphah,MIh_t,C = LSMI_PI(Kx,Ky,Hh,PI,lam=lam)
        if sinkhorn_algorithm=="own":
            if iter==0:
                a = torch.ones((C.shape[0],1)).cuda()
            mu = mu.unsqueeze(1)
            nu = nu.unsqueeze(1)
            PI,a,b,Err = perform_sinkhorn(C,epsilon,mu,nu,a)
        else: # log-stabilized Sinkhorn algorithm
            PI = sinkhorn_stabilized(mu, nu, C, epsilon, method='sinkhorn_stabilized', numItermax=100, cuda=True)

        if iter%10==0:
            err = torch.norm(PI - PI_prev)
            if err<tol:
                MIh = MIh[0:iter]
                break

        MIh[iter] = MIh_t

    Cc = torch.mm(Kx.t(), torch.diag(alphah.view(-1)))
    Cc = torch.mm(Cc, Ky)

    MIh_pair = 0.5*((Cc-1)*(Cc-1)).sum() / (n1*n2)

    return PI.cpu().data.numpy(), MIh.cpu().data.numpy()


def SMI_sinkhorn_semi(Xp, Yp, Xu, Yu, n_iter=20, b=200, beta=0.5, epsilon=0.1, lam=0.0001, warm=False):
    """semi-supervised SMI sinkhorn algorithm"""
    Xp = torch.Tensor(Xp).cuda()
    Yp = torch.Tensor(Yp).cuda()
    Xu = torch.Tensor(Xu).cuda()
    Yu = torch.Tensor(Yu).cuda()
    n, n1, n2 = Xp.shape[1], Xu.shape[1], Yu.shape[1]

    Xpu = torch.cat((Xp,Xu), dim=1) # dx(n+n1)
    Ypu = torch.cat((Yp,Yu), dim=1) # dx(n+n2)

    b = min(b, min(n1,n2))
    index1 = torch.randperm(n1)
    index2 = torch.randperm(n2)
    Xb = Xu[:, index1[0:b-n]]
    Xb = torch.cat((Xp,Xb), dim=1)
    Yb = Yu[:, index2[0:b-n]]
    Yb = torch.cat((Yp,Yb), dim=1)
    
    del Xp; del Xu; del Yp; del Yu

    sigma1 = compmedDist(Xpu.t())
    sigma2 = compmedDist(Ypu.t())
    Kxt = kernel_gaussian(Xb, Xpu, sigma1) # bx(n+n1)
    Kyt = kernel_gaussian(Yb, Ypu, sigma2) # bx(n+n2)
    K1 = Kxt[:, 0:n]
    K2 = Kyt[:, 0:n]
    Kx = Kxt[:, n:]
    Ky = Kyt[:, n:]
    ## Pre-compute H since it does not depends on PI or alpha
    Hh = torch.mm(Kxt, Kxt.t()) * torch.mm(Kyt, Kyt.t()) / ((n1+n)*(n2+n)) # bxb

    del Xpu; del Ypu

    
    PI = torch.rand(n1,n2).cuda()
    PI = PI/PI.sum() # normalize

    mu = (torch.ones((n1,))/n1).cuda()
    nu = (torch.ones((n2,))/n2).cuda()
    
    MIh = torch.zeros(n_iter)
    tol = 1e-9
    for iter in range(n_iter):
        PI_prev = PI
        alphah,MIh_t,C = LSMI_PI_semi(K1,K2,Kx,Ky,Hh,PI,beta,lam=lam)
        C = (1-beta)*C
        if sinkhorn_algorithm=="own":
            if iter==0:
                a = torch.ones((C.shape[0],1)).cuda()
                mu = mu.unsqueeze(1)
                nu = nu.unsqueeze(1)
            PI,a,b,Err = perform_sinkhorn(C,epsilon,mu,nu,a)
        else: # log-stabilized Sinkhorn algorithm
            PI = sinkhorn_stabilized(mu, nu, C, epsilon, method='sinkhorn_stabilized', numItermax=100, cuda=True)
        
        if iter%10==0: # check SMI value and early stop
            err = torch.norm(PI - PI_prev)
            if err<tol:
                MIh = MIh[0:iter]
                break

        MIh[iter] = MIh_t

    Cc = torch.mm(Kxt.t(), torch.diag(alphah.view(-1)))
    Cc = torch.mm(Cc, Kyt)

    MIh_pair = 0.5*((Cc-1)*(Cc-1)).sum() / ((n+n1)*(n+n2))

    return PI.cpu().data.numpy(), MIh.cpu().data.numpy(), MIh_pair.cpu().data.numpy()


def SMI_sinkhorn_semi_CV(Xp, Yp, Xu, Yu, n_iter=20, b=200, beta=0.5, epsilon=0.1, lam=0.0001, warm=False):
    """ semi-supervised SMI sinkhorn algorithm
        Using Paired Data for 2-fold CV
    """
    Xp = torch.Tensor(Xp).cuda()
    Yp = torch.Tensor(Yp).cuda()
    Xu = torch.Tensor(Xu).cuda()
    Yu = torch.Tensor(Yu).cuda()
    n, n1, n2 = Xp.shape[1], Xu.shape[1], Yu.shape[1]

    n_cv = int(np.round(n/2))
    index_cv = torch.randperm(n)
    Xp_cv_tr = Xp[:, index_cv[0:n_cv]]
    Yp_cv_tr = Yp[:, index_cv[0:n_cv]]
    Xp_cv_te = Xp[:, index_cv[n_cv:]]
    Yp_cv_te = Yp[:, index_cv[n_cv:]]

    Xpu = torch.cat((Xp_cv_tr,Xu), dim=1) # dx(n_cv+n1)
    Ypu = torch.cat((Yp_cv_tr,Yu), dim=1) # dx(n_cv+n2)
    
    # basis sampling
    b = min(b, min(n1,n2))
    index1 = torch.randperm(n1)
    index2 = torch.randperm(n2)
    Xb = Xu[:, index1[0:b-n_cv]]
    Xb = torch.cat((Xp_cv_tr,Xb), dim=1)
    Yb = Yu[:, index2[0:b-n_cv]]
    Yb = torch.cat((Yp_cv_tr,Yb), dim=1)
    
    del Xp; del Xu; del Yp; del Yu

    sigma1 = compmedDist(Xpu.t())
    sigma2 = compmedDist(Ypu.t())
    Kxt = kernel_gaussian(Xb, Xpu, sigma1) # bx(n_cv+n1)
    Kyt = kernel_gaussian(Yb, Ypu, sigma2) # bx(n_cv+n2)
    K1 = Kxt[:, 0:n_cv]
    K2 = Kyt[:, 0:n_cv]
    Kx = Kxt[:, n_cv:]
    Ky = Kyt[:, n_cv:]
    ## Pre-compute H since it does not depends on PI or alpha
    Hh = torch.mm(Kxt, Kxt.t()) * torch.mm(Kyt, Kyt.t()) / ((n1+n_cv)*(n2+n_cv)) # bxb

    del Kxt; del Kyt; del Xpu; del Ypu

    PI = torch.rand(n1, n2).cuda()
    PI = PI/PI.sum()  # normalize

    mu = (torch.ones((n1,))/n1).cuda()
    nu = (torch.ones((n2,))/n2).cuda()

    MIh = torch.zeros(n_iter)
    tol = 1e-9
    for iter in range(n_iter):
        PI_prev = PI
        alphah,MIh_t,C = LSMI_PI_semi(K1,K2,Kx,Ky,Hh,PI,beta,lam=lam)
        C = (1-beta)*C
        if sinkhorn_algorithm=="own":
            if iter==0:
                a = torch.ones((C.shape[0],1)).cuda()
            PI,a,b,Err = perform_sinkhorn(C,epsilon,mu,nu,a)
        else: # log-stabilized Sinkhorn algorithm
            PI = sinkhorn_stabilized(mu, nu, C, epsilon, method='sinkhorn_stabilized', numItermax=100, cuda=True)

        if iter%10==0: # check SMI value and early stop
            err = torch.norm(PI - PI_prev)
            if err.abs()<tol:
                MIh = MIh[0:iter]
                break

        MIh[iter] = MIh_t

    nte = Xp_cv_te.shape[1]
    Kx_te = kernel_gaussian(Xb, Xp_cv_te, sigma1)
    Ky_te = kernel_gaussian(Yb, Yp_cv_te, sigma2)

    Hh = torch.mm(Kx_te, Kx_te.t()) * torch.mm(Ky_te, Ky_te.t()) / (nte*nte)
    hh = (Kx_te*Ky_te).sum(1) / nte
    MIh_cv = 0.5*torch.mm(torch.mm(alphah.t(), Hh),alphah) - torch.mm(alphah.t(), hh.view(-1,1))


    return PI.cpu().data.numpy(), MIh.cpu().data.numpy(), MIh_cv.cpu().data.numpy()


def SMI_sinkhorn_semi_CV2(Xp_cv_tr, Yp_cv_tr, Xp_cv_te, Yp_cv_te, Xu, Yu, n_iter=20, b=200, beta=0.5, epsilon=0.1, lam=0.0001, warm=False):
    """ semi-supervised SMI sinkhorn algorithm
        Using Paired Data for 2-fold CV
        Train-Test split are predefined as input
    """
    Xp_cv_tr = torch.Tensor(Xp_cv_tr).cuda()
    Yp_cv_tr = torch.Tensor(Yp_cv_tr).cuda()
    Xp_cv_te = torch.Tensor(Xp_cv_te).cuda()
    Yp_cv_te = torch.Tensor(Yp_cv_te).cuda()
    Xu = torch.Tensor(Xu).cuda()
    Yu = torch.Tensor(Yu).cuda()

    n1, n2 = Xu.shape[1], Yu.shape[1]
    n_cv = Xp_cv_tr.shape[1]

    Xpu = torch.cat((Xp_cv_tr,Xu), dim=1) # dx(n_cv+n1)
    Ypu = torch.cat((Yp_cv_tr,Yu), dim=1) # dx(n_cv+n2)
    
    # basis sampling
    b = min(b, min(n1,n2))
    index1 = torch.randperm(n1)
    index2 = torch.randperm(n2)
    Xb = Xu[:, index1[0:b-n_cv]]
    Xb = torch.cat((Xp_cv_tr,Xb), dim=1)
    Yb = Yu[:, index2[0:b-n_cv]]
    Yb = torch.cat((Yp_cv_tr,Yb), dim=1)
    
    del Xu; del Yu

    sigma1 = compmedDist(Xpu.t())
    sigma2 = compmedDist(Ypu.t())
    Kxt = kernel_gaussian(Xb, Xpu, sigma1) # bx(n_cv+n1)
    Kyt = kernel_gaussian(Yb, Ypu, sigma2) # bx(n_cv+n2)
    K1 = Kxt[:, 0:n_cv]
    K2 = Kyt[:, 0:n_cv]
    Kx = Kxt[:, n_cv:]
    Ky = Kyt[:, n_cv:]
    ## Pre-compute H since it does not depends on PI or alpha
    Hh = torch.mm(Kxt, Kxt.t()) * torch.mm(Kyt, Kyt.t()) / ((n1+n_cv)*(n2+n_cv)) # bxb

    del Kxt; del Kyt; del Xpu; del Ypu

    PI = torch.rand(n1, n2).cuda()
    PI = PI/PI.sum()  # normalize

    mu = (torch.ones((n1,))/n1).cuda()
    nu = (torch.ones((n2,))/n2).cuda()

    MIh = torch.zeros(n_iter)
    tol = 1e-9
    for iter in range(n_iter):
        PI_prev = PI
        alphah,MIh_t,C = LSMI_PI_semi(K1,K2,Kx,Ky,Hh,PI,beta,lam=lam)
        C = (1-beta)*C
        if sinkhorn_algorithm=="own":
            if iter==0:
                a = torch.ones((C.shape[0],1)).cuda()
            PI,a,b,Err = perform_sinkhorn(C,epsilon,mu,nu,a)
        else: # log-stabilized Sinkhorn algorithm
            PI = sinkhorn_stabilized(mu, nu, C, epsilon, method='sinkhorn_stabilized', numItermax=100, cuda=True)

        if iter%10==0: # check SMI value and early stop
            err = torch.norm(PI - PI_prev)
            if err.abs()<tol:
                MIh = MIh[0:iter]
                break

        MIh[iter] = MIh_t

    nte = Xp_cv_te.shape[1]
    Kx_te = kernel_gaussian(Xb, Xp_cv_te, sigma1)
    Ky_te = kernel_gaussian(Yb, Yp_cv_te, sigma2)

    Hh = torch.mm(Kx_te, Kx_te.t()) * torch.mm(Ky_te, Ky_te.t()) / (nte*nte)
    hh = (Kx_te*Ky_te).sum(1) / nte
    MIh_cv = 0.5*torch.mm(torch.mm(alphah.t(), Hh),alphah) - torch.mm(alphah.t(), hh.view(-1,1))


    return PI.cpu().data.numpy(), MIh.cpu().data.numpy(), MIh_cv.cpu().data.numpy()


def SMI_sinkhorn_semi_CV_main(Xp, Yp, Xu, Yu, n_iter=20, b=200, beta=0.5, epsilon=0.1, lam=0.0001, warm=False):
    """ Main function for cross-validation
        split data into 2 folds and exchange the train-test split
    """
    n = np.shape(Xp)[1]
    n_cv = int(np.round(n/2))
    index_cv = np.random.permutation(n)
    Xp_cv_tr = Xp[:, index_cv[0:n_cv]]
    Yp_cv_tr = Yp[:, index_cv[0:n_cv]]
    Xp_cv_te = Xp[:, index_cv[n_cv:]]
    Yp_cv_te = Yp[:, index_cv[n_cv:]]

    PI, MIs, MI_cv1 = SMI_sinkhorn_semi_CV2(Xp_cv_tr,Yp_cv_tr,Xp_cv_te,Yp_cv_te,Xu,Yu,n_iter,b,beta,epsilon,lam,warm)
    PI, MIs, MI_cv2 = SMI_sinkhorn_semi_CV2(Xp_cv_te,Yp_cv_te,Xp_cv_tr,Yp_cv_tr,Xu,Yu,n_iter,b,beta,epsilon,lam,warm)

    return PI, MIs, 0.5*(MI_cv1+MI_cv2)


def SMI_pair(X, Y, b=200, lam=0.0001):
    """SMI with Paired Data"""
    X = torch.Tensor(X).cuda()
    Y = torch.Tensor(Y).cuda()
    n1,n2 = X.shape[1], Y.shape[1]
    if n1!=n2:
        print(X.shape,Y.shape)
        print('Paired SMI estimation: n1!=n2')
        exit(-1)
    n = n1

    b = min(b, n) 
    index1 = torch.randperm(n1)
    index2 = torch.randperm(n2)
    Xb = X[:, index1[0:b]]
    Yb = Y[:, index2[0:b]]
    
    PI = (torch.eye(n1)/n1).cuda()

    sigma1 = compmedDist(X.t())
    sigma2 = compmedDist(Y.t())
    Kx = kernel_gaussian(Xb, X, sigma1)
    Ky = kernel_gaussian(Yb, Y, sigma2)
    ## Precompute H since it does not depend on PI or alpha
    Hh = torch.mm(Kx, Kx.t()) * torch.mm(Ky, Ky.t()) / (n1*n2) # bxb
    
    del X; del Y
    
    alphah,MIh_pair,C = LSMI_PI(Kx,Ky,Hh,PI,lam=lam)

    return MIh_pair.cpu().data.numpy()
 

def SMI_pair_CV(Xp, Yp, b=200, lam=0.0001):
    """ Paired SMI estimation with CV"""
    Xp = torch.Tensor(Xp).cuda()
    Yp = torch.Tensor(Yp).cuda()
    n = Xp.shape[1]

    n_cv = int(np.round(n/2))
    index_cv = torch.randperm(n)
    Xp_cv_tr = Xp[:, index_cv[0:n_cv]]
    Yp_cv_tr = Yp[:, index_cv[0:n_cv]]
    Xp_cv_te = Xp[:, index_cv[n_cv:]]
    Yp_cv_te = Yp[:, index_cv[n_cv:]]

    b = min(b, n)
    index1 = torch.randperm(n)
    index2 = torch.randperm(n)
    Xb = Xp[:, index1[0:b]]
    Yb = Yp[:, index2[0:b]]
    
    sigma1 = compmedDist(Xp.t())
    sigma2 = compmedDist(Yp.t())
    Kx_tr = kernel_gaussian(Xb, Xp_cv_tr, sigma1) 
    Ky_tr = kernel_gaussian(Yb, Yp_cv_tr, sigma2)
    ## Precompute H since it does not depend on PI or alpha
    Hh = torch.mm(Kx_tr, Kx_tr.t()) * torch.mm(Ky_tr, Ky_tr.t()) / (n_cv*n_cv) # bxb
    
    del Xp; del Yp

    PI = (torch.eye(n_cv)/n_cv).cuda()

    alphah,MIh_pair,C = LSMI_PI(Kx_tr,Ky_tr,Hh,PI,lam=lam)
    
    nte = Xp_cv_te.shape[1]
    Kx_te = kernel_gaussian(Xb, Xp_cv_te, sigma1)
    Ky_te = kernel_gaussian(Yb, Yp_cv_te, sigma2)

    Hh = torch.mm(Kx_te, Kx_te.t()) * torch.mm(Ky_te, Ky_te.t()) / (nte * nte)
    hh = (Kx_te*Ky_te).sum(1) / nte
    MIh_cv = 0.5*torch.mm(torch.mm(alphah.t(), Hh),alphah) - torch.mm(alphah.t(), hh.view(-1,1))
    

    return MIh_pair.cpu().data.numpy(), MIh_cv.cpu().data.numpy()
 

def SMI_pair_CV2(Xp_cv_tr, Yp_cv_tr, Xp_cv_te, Yp_cv_te, b=200, lam=0.0001):
    """ Paired SMI estimation with CV"""
    Xp_cv_tr = torch.Tensor(Xp_cv_tr).cuda()
    Yp_cv_tr = torch.Tensor(Yp_cv_tr).cuda()
    Xp_cv_te = torch.Tensor(Xp_cv_te).cuda()
    Yp_cv_te = torch.Tensor(Yp_cv_te).cuda()
    n_cv = Xp_cv_tr.shape[1] # train num

    Xp = torch.cat((Xp_cv_tr,Xp_cv_te), dim=1)
    Yp = torch.cat((Yp_cv_tr,Yp_cv_te), dim=1)
    n = Xp.shape[1]

    b = min(b, n)
    index1 = torch.randperm(n)
    index2 = torch.randperm(n)
    Xb = Xp[:, index1[0:b]]
    Yb = Yp[:, index2[0:b]]
    
    sigma1 = compmedDist(Xp.t())
    sigma2 = compmedDist(Yp.t())
    Kx_tr = kernel_gaussian(Xb, Xp_cv_tr, sigma1) 
    Ky_tr = kernel_gaussian(Yb, Yp_cv_tr, sigma2)
    ## Precompute H since it does not depend on PI or alpha
    Hh = torch.mm(Kx_tr, Kx_tr.t()) * torch.mm(Ky_tr, Ky_tr.t()) / (n_cv*n_cv) # bxb
    
    del Xp; del Yp

    PI = (torch.eye(n_cv)/n_cv).cuda()

    alphah,MIh_pair,C = LSMI_PI(Kx_tr,Ky_tr,Hh,PI,lam=lam)
    
    nte = Xp_cv_te.shape[1]
    Kx_te = kernel_gaussian(Xb, Xp_cv_te, sigma1)
    Ky_te = kernel_gaussian(Yb, Yp_cv_te, sigma2)

    Hh = torch.mm(Kx_te, Kx_te.t()) * torch.mm(Ky_te, Ky_te.t()) / (nte * nte)
    hh = (Kx_te*Ky_te).sum(1) / nte
    MIh_cv = 0.5*torch.mm(torch.mm(alphah.t(), Hh),alphah) - torch.mm(alphah.t(), hh.view(-1,1))
    
    return MIh_pair.cpu().data.numpy(), MIh_cv.cpu().data.numpy()
 





