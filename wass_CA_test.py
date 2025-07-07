#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 10:10:02 2025

@author: dx1
"""

import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sdf_0
import master_wass_metric_binary as mwmb
import window_sampling_algorithm as wsa
from scipy import optimize


def convert_sdf_binary(wds):
    for i in range(len(wds)):
        w = wds[i]
        w[np.where(w > 0)] = -1
        w += 1
        wds[i] = w
    return wds

def convert_slices(slices, sln, axis, plot):
    ### takes level set slices ###
    ### every c slices ###
    ### this c is subdividing the volume ###
    c = int(np.min(np.shape(slices))/sln)
    c = 5
    if axis == 'x':
        n = np.shape(slices[:,:,0])
        sl = np.zeros((sln, n[0], n[1]))
        for i in range(sln):
            sl[i] = slices[:,:,c*(i+1)]
            
    if axis == 'y':
        n = np.shape(slices[:,0,:])
        sl = np.zeros((sln, n[0], n[1]))
        for i in range(sln):
            sl[i] = slices[:,c*(i+1),:]
            
    if axis == 'z':
        n = np.shape(slices[0,:,:])
        sl = np.zeros((sln, n[0], n[1]))
        for i in range(sln):
            sl[i] = slices[c*(i+3),:,:]
            
    for i in range(sln):
        sdfslice = sdf_0.fastSweepSDF(sl[i])
        sl[i] = convert_sdf_binary([sdfslice])[0]
        
        if plot:
            plt.figure()
            plt.imshow(sdfslice)
            plt.show()
            plt.figure()
            plt.title(axis + ' axis cross section')
            plt.imshow(sl[i], cmap = 'binary')
            plt.show()
            
    return sl
        

### runs a comparison between two distributions ###
### finds the distance matrix and solves it too ###
def linear_distr_dist(U,V):
    import time
    dmat = np.zeros((40,40))
    s = time.time()
    for i in range(1):
        for j in range(1):
            dmat[i,j] = mwmb.window_distance(U[i], V[j], l, iwB)/m
    print(time.time() - s, 'seconds')
    rindy,cindy = optimize.linear_sum_assignment(dmat)
    distrib_dist = dmat[rindy, cindy].sum()/40
    return dmat, distrib_dist    


### --- ### --- ### --- ### --- ### --- ### --- ### --- ### --- ### 

fp = '/Users/dx1/Research/microstructure_data/'
slices = np.load(fp + 'r1_P22_gids.npy')
name = 'r1_P22'

### prints raw grain id data slices ###
if False:    
    for i in range(5):
        plt.figure()
        plt.imshow(slices[i*100])
        plt.show()   
 
    
### converts data to sdf and binary, saves slices in npy file ###
if False:
    slx = convert_slices(slices, 10, 'x', 0)
    sly = convert_slices(slices, 10, 'y', 0)
    slz = convert_slices(slices, 10, 'z', 0)
    
    np.save(fp + name + '_x.npy', slx[:, 100:-25, 50:-50])
    np.save(fp + name + '_y.npy', sly[:, 100:-25, 50:-50])
    np.save(fp + name + '_z.npy', slz[:, 25:-25,  25:-25])
    
    
### load in the slice data for each axis from the npy files ###
slx = np.load(fp + name + '_x.npy')
sly = np.load(fp + name + '_y.npy')
slz = np.load(fp + name + '_z.npy')


### plots slices ###
if False:
    for i in range(10):
        plt.figure()
        plt.imshow(slx[i])
        plt.figure()
        plt.imshow(sly[i])
        plt.figure()
        plt.imshow(slz[i])
    plt.show()
    
    
### create bipartite graph for all pixels ###
l = 50        # px side length of windows
m = int(l**2)  # px area of windows
# iwB = np.load(fp+'iwB150.npy')

iwB = mwmb.intrawindow_block(l)


### a sweeping study, within a window ###
### shows the variation in distance as you move across a dataset ### 
if False:
    dn = 100
    dists = np.zeros(dn)
    for i in range(dn):
        ai = 0
        aj = 0
        bi = 0
        bj = i*2  
        A = slx[0, ai:ai+l, aj:aj+l]
        B = slx[0, bi:bi+l, bj:bj+l]
        wd = mwmb.window_distance(A, B, l, iwB)/m
        
        if False:    
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","yellow"])
            plt.figure()
            plt.title('distance = ' + str(wd))
            plt.imshow(A + 2*B, cmap = cmap)
        dists[i] = wd

    plt.figure()
    plt.ylim(0,np.max(dists)*1.1)
    plt.scatter(np.arange(dn), dists)
    plt.show()
    print(np.average(dists))


### randomly samples non-overlapping windows from each slice from each axis ###
if False:
    wn = 1
    sln = 10
    winds_x = np.zeros((int(sln*wn), l, l))
    winds_y = np.zeros((int(sln*wn), l, l))
    winds_z = np.zeros((int(sln*wn), l, l))

    for i in range(sln):
        indices = wsa.find_valid_wnds(np.min(np.shape(slx[0])), l, wn)
        #print(indices)
        for j in range(wn):
            a = indices[0][0, j]
            b = indices[0][1, j]
            winds_x[i*wn+j] = slx[i, a:a+l, b:b+l]
            winds_y[i*wn+j] = sly[i, a:a+l, b:b+l]
            winds_z[i*wn+j] = slz[i, a:a+l, b:b+l]

def sample_randomly(slices, l, wn = 0):
    s = np.shape(slices)
    # sample half the area of the micrograph slice # 
    if wn == 0:
        wn = int(s[1]*s[2] / l**2 / 2 - 1)
    sln = s[0]
    winds = np.zeros((int(sln*wn), l, l))

    for i in range(sln):
        indices = wsa.find_valid_wnds(np.min(s[1:]), l, wn)
        for j in range(wn):
            a = indices[0][0, j]
            b = indices[0][1, j]
            winds[i*wn+j] = slices[i, a:a+l, b:b+l]
    return winds


# plateslice = np.load(fp+'enlarged_plate.npy')

### enlarge slices ###
if False:
    slice_z_every5 = np.copy(slices[::5][:10])
    enlarged_slices = np.zeros((10,1203,1203))
    
    for i in range(10):
        for j in range(401):
            for k in range(401):
                enlarged_slices[i,j*3:(j+1)*3, k*3:(k+1)*3] = slice_z_every5[i,j,k]
        enlarged_slices[i] = sdf_0.fastSweepSDF(enlarged_slices[i])
    enlarged_slices = 1 - convert_sdf_binary(enlarged_slices)
    np.save(fp+'3x_z_every5.npy',enlarged_slices)



# for i in range(100):
#     plt.figure()
#     plt.imshow(1 - winds[i], 'binary')
# plt.show()

### compute distance between two windows ###
if False:
    import time
    s = time.time()
    dx2x3 = mwmb.window_distance(winds_x[2],winds_x[3],l,iwB)/m
    print(np.around(time.time() - s, 3))
    print(dx2x3)
    plt.figure()
    plt.imshow(winds_x[2])
    plt.figure()
    plt.imshow(winds_x[3])
    plt.show()
    #dmatxz[6,7] = dx2x3

if False:
    d_M = np.zeros((3,3))
    Us = [winds_x[:40], winds_y[:40], winds_z[:40]] 
    Vs = [winds_x[40:], winds_y[40:], winds_z[40:]] 
    for i in range(3):
        U = Us[i]
        for j in range(i,3):
            V = Vs[j]
            dmat, distrib_dist = linear_distr_dist(U, V)
            d_M[i,j] = distrib_dist
            d_M[j,i] = distrib_dist
            print(distrib_dist)
    print(d_M)
    

### calculates the distribution distance ### 
### and plots and overlays the optimal matches ###
if False:
    rindy,cindy = optimize.linear_sum_assignment(dmat)
    distrib_dist = dmat[rindy, cindy].sum()/40
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","darkviolet"])
    for i in range(40):
        plt.figure()
        plt.title('distance = ' + str(dmat[rindy[i],cindy[i]]))
        plt.imshow(U[rindy[i]] + 2*V[cindy[i]], cmap = cmap)    
    plt.show()
    print(distrib_dist)


### calculates the distribution distance for maximal weights ### 
### and plots and overlays the optimal (maximized) matches ###
if False:
    rindy,cindy = optimize.linear_sum_assignment(dmat, maximize = True)
    distrib_dist = dmat[rindy, cindy].sum()/40
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","darkviolet"])
    for i in range(40):
        plt.figure()
        plt.title('distance = ' + str(dmat[rindy[i],cindy[i]]))
        plt.imshow(U[rindy[i]] + 2*V[cindy[i]], cmap = cmap)
    plt.show()
    print(distrib_dist)


# mwmb.compare_distribs(5, U, V, iwB, l)

# ai = 0
# aj = 0
# bi = 0
# bj = 0

# A = slx[0, ai:ai+l, aj:aj+l]
# B = slx[1, bi:bi+l, bj:bj+l]
# wd = mwmb.window_distance(A, B, l, iwB)/m

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","yellow"])
# plt.figure()
# plt.title('distance = ' + str(wd))
# plt.imshow(A + 2*B, cmap = cmap)


if __name__ == '__main__' :
    enlarged_slices = np.load(fp+'3x_z_every5.npy')
    winds = sample_randomly(enlarged_slices, 50, wn = 100)
    np.save(fp+'3x_z_100windows_every5.npy', winds)

    winds = np.load(fp+'3x_z_100windows_every5.npy')
    dmat = np.zeros((100,100))
    for i in range(100):
        for j in range(i+1,100):
            dmat[i,j] =  mwmb.window_distance(winds[i], winds[j], l, iwB)/m

    np.save(fp+'3x_z_dmat.npy',dmat)
    
