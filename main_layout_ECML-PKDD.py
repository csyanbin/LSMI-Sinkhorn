from utils_im import lab
from PIL import Image
import numpy as np
import random
import torch
import os



os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
torch.set_num_threads(2)
torch.backends.cudnn.benchmark = True


gpu = True
if gpu:
    from utils_gpu import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)



def Flickr(seed=2021, frame='rectangle'):
    seed = 2021
    set_seed(seed)

    b = 200
    beta = 0.2
    epsilon = 0.5
    lam = 0.0001
    n_iter = 10

    ino = 20
    jno = 20
    psize = 40
    iname = 'images/img'
    
    ## ---- paired data ---- #
    # blue:295,319,240,15, orange:31,107,168,169, green:16,20,78,247, black:33,49,99,131 

    ## ---- << Rectangle>> ---- ##
    if frame=='rectangle':
        save_name = 'layout/flickr_16x16_sorted.jpg'
        pimlist = [295,31,16,33]
        pgrid = np.array([[1,1],[1,jno],[ino,1],[ino,jno]])

    ## ---- << Triangle >> ---- ##
    if  frame=="triangle":
        save_name = 'layout/flickr_triangle_sorted.jpg'
        pimlist = [16,31,33]
        pgrid = np.array([[1,10],[ino,1],[ino,jno]])


    pgrid = pgrid.T # 2x4
    
    pimgdata = []
    pdata = []
    for cnt in pimlist:
        fname = iname + str(cnt) + '.jpg'
        im = Image.open(fname)
        aim = np.asarray(im)
        [M,N,L] = aim.shape
        mno = np.fix(M/psize).astype('int')
        nno = np.fix(N/psize).astype('int')
        aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
        pdata.append(aim.flatten())
        daim = np.double(aim)/255.0
        # convert from RGB to Lab
        daimlab = lab(daim)
        pimgdata.append(daimlab.flatten())
    pdata = np.array(pdata)
    pimgdata = np.array(pimgdata).T
    print(pdata.shape,pimgdata.shape,pgrid.shape)
    
    
    ## ---- unpaired data ---- ##
    imgdata = []
    data = []
    counter = 0
    for i in range(16):
        for j in range(20):
            counter = counter + 1
            if counter in pimlist:
                continue
            fname = iname + str(counter) + '.jpg'
            im = Image.open(fname)
            aim = np.asarray(im)
            [M,N,L] = aim.shape
            mno = np.fix(M/psize).astype('int')
            nno = np.fix(N/psize).astype('int')
            aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
            data.append(aim.flatten())
            daim = np.double(aim)/255.0
            # convert from RGB to Lab
            daimlab = lab(daim)
            imgdata.append(daimlab.flatten())
            
    data = np.array(data)
    imgdata = np.array(imgdata).T
    print(data.shape,imgdata.shape)
    
    
    griddata = []
    ## ---- << rectangle grid >> ---- ##
    if  frame=="rectangle":
        for i in range(1,ino+1):
            for j in range(1,jno+1):
                if i in pgrid[0,:] and j in pgrid[1,:]: # exclude paired data
                    continue
                griddata.append([i,j])

    ## ---- << triangle grid >> ---- ##  
    if  frame=="triangle":
        for i in range(1,ino+1):
            for j in range(1,jno+1):
                if j<=jno/2 and j>-0.5*i+jno/2:
                    griddata.append([i,j])
                if j>jno/2 and (j-jno/2)<=0.5*i:
                    griddata.append([i,j])
    griddata = np.array(griddata).T
    print(griddata.shape)
    

    PI, MIs, MI_pair = SMI_sinkhorn_semi(pimgdata, pgrid, imgdata, griddata, n_iter, b, beta, epsilon, lam)
    print(MIs)
    
    i_sorting = PI.argmax(axis=0)
    imgdata_sorted = data[i_sorting,]
    irange = range(0,psize*ino,psize)
    jrange = range(0,psize*jno,psize)
    
    patching = np.ones((ino*psize, jno*psize, 3))*255
    for i in range(pgrid.shape[1]):
        sx = (pgrid[0,i]-1)*psize
        tx = pgrid[0,i]*psize
        sy = (pgrid[1,i]-1)*psize
        ty = pgrid[1,i]*psize
        patching[sx:tx, sy:ty, :] = np.reshape(pdata[i,], [psize,psize,3]);
    
    for i in range(griddata.shape[1]):
        sx = (griddata[0,i]-1)*psize
        tx = griddata[0,i]*psize
        sy = (griddata[1,i]-1)*psize
        ty = griddata[1,i]*psize
        patching[sx:tx, sy:ty, :] = np.reshape(imgdata_sorted[i,], [psize,psize,3]);
    
    im = Image.fromarray(patching.astype(np.uint8))
    im.save(save_name, 'JPEG')
    


def ECML(seed):
    ## Algorithm Parameters
    #params  = [300, 0.2, 0.15, 0.005, 20] # seed=2021
    #params  = [250, 0.25, 0.1, 0.005, 15]  # seed=0,20,2021
    #params  = [300, 0.18, 0.15, 0.007, 20]  # seed=10,200,2021
    params  = [250, 0.4, 0.1, 0.002, 25]  # seed=0,20,2021
    seed    = 2021
    b       = params[0]
    beta    = params[1]
    epsilon = params[2]
    lam     = params[3]
    n_iter  = params[4]
    

    ## --- Template and Predefined Positions ---- ##
    im_template = Image.open('layout/ECML-PKDD_square.jpg')
    save_name = 'layout/flickr_ECML-PKDD_square.jpg'

    color_im_dict = {295:'blue',319:'blue',240:'blue',15:'blue',
            31:'orange',107:'orange',168:'orange',169:'orange',
            16:'green',20:'green',78:'green',247:'green',
            33:'black',49:'black',99:'black',131:'black',
            175:'white',76:'white', 27:'red',97:'red',  284:'purple', 40:'grey'}
    ## ---- Squared ECML-PKDD configuration ---- ##
    #pimlist = [31,27,284,295,175,16,40,33]
    pimlist = [76,295,16,31,40,284,27,33]
    random.shuffle(pimlist)
    set_seed(seed)
    for im in pimlist:
        print(color_im_dict[im])
    pgrid = np.array([[7,3],[7,16],[7,30],[7,45],[23,4],[23,15],[23,28],[23,42]]) 

    pgrid = pgrid.T # 2x4
    
    ino,jno = im_template.size  ## 13x40 or 13x80
    psize = 40
    iname = 'images/img'
    pimgdata = []
    pdata = []
    for cnt in pimlist:
        fname = iname + str(cnt) + '.jpg'
        im = Image.open(fname)
        aim = np.asarray(im)
        [M,N,L] = aim.shape
        mno = np.fix(M/psize).astype('int')
        nno = np.fix(N/psize).astype('int')
        aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
        pdata.append(aim.flatten())
        daim = np.double(aim)/255.0
        # convert from RGB to Lab
        #daimlab = lab(daim)
        daimlab = daim
        pimgdata.append(daimlab.flatten())
    pdata = np.array(pdata)
    pimgdata = np.array(pimgdata).T
    print(pdata.shape,pimgdata.shape,pgrid.shape)
    
    
    ## ---- unpaired data ---- ##
    imgdata = []
    data = []
    counter = 0
    for i in range(320):
            counter = counter + 1
            if counter in pimlist:
                continue
            fname = iname + str(counter) + '.jpg'
            im = Image.open(fname)
            aim = np.asarray(im)
            [M,N,L] = aim.shape
            mno = np.fix(M/psize).astype('int')
            nno = np.fix(N/psize).astype('int')
            aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
            data.append(aim.flatten())
            daim = np.double(aim)/255.0
            # convert from RGB to Lab
            daimlab = lab(daim)
            imgdata.append(daimlab.flatten())
            
    data = np.array(data)
    imgdata = np.array(imgdata).T
    print(data.shape,imgdata.shape)
    
    
    #### grid data ####
    pos = np.where(np.array(im_template)<100)
    print('num postions:',len(pos[0]),len(pos[1]))
    griddata = []
    for i in range(len(pos[0])):
            griddata.append([pos[0][i], pos[1][i]])
    griddata = np.array(griddata).T
    print(griddata.shape)
    
    # SMI_sinkhorn run
    PI, MIs, MI_pair = SMI_sinkhorn_semi(pimgdata, pgrid, imgdata, griddata, n_iter, b, beta, epsilon, lam)
    #print(MIs)
    
    i_sorting = PI.argmax(axis=0)
    #print('unique:',len(np.unique(i_sorting)))
    imgdata_sorted = data[i_sorting,]
    irange = range(0,psize*ino,psize)
    jrange = range(0,psize*jno,psize)
    
    patching = np.ones((jno*psize, ino*psize, 3))*255
    print(ino,jno)
    
    for i in range(griddata.shape[1]):
        sx = (griddata[0,i])*psize
        tx = (griddata[0,i]+1)*psize
        sy = (griddata[1,i])*psize
        ty = (griddata[1,i]+1)*psize
        patching[sx:tx, sy:ty, :] = np.reshape(imgdata_sorted[i,], [psize,psize,3]);
    
    im = Image.fromarray(patching.astype(np.uint8))
    im.save(save_name, 'JPEG')



def ECML_CIFAR(seed):
    
    ## Algorithm Parameters
    #params  = [500, 0.3, 0.15, 0.001, 25] # seed=0,20,2021
    #params  = [400, 0.25, 0.1, 0.0016, 25]  # seed=0,20,2021
    params  = [400, 0.18, 0.15, 0.00008, 20]  # seed=0,20,2021
    seed = 688
    b       = params[0]
    beta    = params[1]
    epsilon = params[2]
    lam     = params[3]
    n_iter  = params[4]


    ## --- Template and Predefined Positions ---- ##
    im_template = Image.open('layout/ECML-PKDD_square.jpg')
    save_name = 'layout/cifar_ECML2021_square.jpg'

    # 10000x32x32x3
    cifar_images = np.load('pytorch_resnet_cifar10/cifar10_img.npy')
    # 10000x64,10000x1
    data = np.load('pytorch_resnet_cifar10/resnet20_feat_cifar_test.npz')
    feats = data['feats']

    # ---- paired data ---- #
    cifar_dict = {98:'airplane',235:'airplace', 231:'car',9:'car',104:'car', 
            75:'bird',67:'bird',149:'bird', 294:'cat',77:'cat', 
            365:'deer',159:'deer', 361:'dog',230:'dog',16:'dog', 
            112:'frog',142:'frog', 288:'horse',208:'horse',
            199:'ship',185:'ship',72:'ship', 133:'truck', 11:'truck'}
            
    ## ---- Squared ICML2020 configuration ---- ##
    #pimlist = [231,67,361,199,98,288,365,133]
    #pimlist = [231,133,235,67,199,230,288,365]
    pimlist = [231,133,235,67,199,361,208,159]
    #random.shuffle(pimlist)
    set_seed(seed)
    for im in pimlist:
        print(cifar_dict[im])
    #pgrid = np.array([[6,3],[6,15],[6,30],[6,45],[20,5],[20,16],[20,28],[20,42]]) 
    pgrid = np.array([[7,3],[7,16],[7,30],[7,45],[23,4],[23,15],[23,28],[23,42]]) 
    pgrid = pgrid.T # 2x4

    ino,jno = im_template.size  ## 13x40 or 13x80
    psize = 32
    pimgdata = []
    pdata = []
    for idx in pimlist:
        aim = cifar_images[idx]
        [M,N,L] = aim.shape
        mno = np.fix(M/psize).astype('int')
        nno = np.fix(N/psize).astype('int')
        aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
        pdata.append(aim.flatten())
        # image feature
        pimgdata.append(feats[idx])
    pdata = np.array(pdata)
    pimgdata = np.array(pimgdata).T
    print(pdata.shape,pimgdata.shape,pgrid.shape)
    
    ## ---- unpaired data ---- ##
    n_unpair = 1000
    idx_list = np.random.permutation(9000) # select 1000:10000
    idx_list = idx_list[0:n_unpair]+1000   # idx starts from 1000
    imgdata = []
    data = []
    counter = 0
    for i in idx_list:
            aim = cifar_images[i]
            [M,N,L] = aim.shape
            mno = np.fix(M/psize).astype('int')
            nno = np.fix(N/psize).astype('int')
            aim = aim[0:psize*mno:mno,0:psize*nno:nno,:]
            data.append(aim.flatten())
            # image feature
            imgdata.append(feats[i].flatten())
            
    data = np.array(data)
    imgdata = np.array(imgdata).T
    print(data.shape,imgdata.shape)

    #### grid data ####
    pos = np.where(np.array(im_template)<100)
    print('num postions:',len(pos[0]),len(pos[1]))
    griddata = []
    for i in range(len(pos[0])):
            griddata.append([pos[0][i], pos[1][i]])
    griddata = np.array(griddata).T
    print(griddata.shape)
    
    PI, MIs, MI_pair = SMI_sinkhorn_semi(pimgdata, pgrid, imgdata, griddata, n_iter, b, beta, epsilon, lam)
    print(MIs)
    
    i_sorting = PI.argmax(axis=0)
    #print('unique:',len(np.unique(i_sorting)))
    imgdata_sorted = data[i_sorting,]
    irange = range(0,psize*ino,psize)
    jrange = range(0,psize*jno,psize)
    
    patching = np.ones((jno*psize, ino*psize, 3))*255
    
    for i in range(griddata.shape[1]):
        sx = (griddata[0,i])*psize
        tx = (griddata[0,i]+1)*psize
        sy = (griddata[1,i])*psize
        ty = (griddata[1,i]+1)*psize
        patching[sx:tx, sy:ty, :] = np.reshape(imgdata_sorted[i,], [psize,psize,3]);
    
    im = Image.fromarray(patching.astype(np.uint8))
    im.save(save_name, 'JPEG')
    



seed = 2021

print("Triangle layout for Flickr images")
Flickr(seed,frame='triangle')
print()
print("Rectangle 16x20 layout for Flickr images")
Flickr(seed,frame='rectangle')
print()

print("Layout for character <<ECML PKDD>>")
ECML(seed)
print()
print("Layout for character <<ECML PKDD>> on CIFAR dataset")
ECML_CIFAR(seed)
