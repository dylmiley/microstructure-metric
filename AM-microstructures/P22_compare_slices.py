#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 07:11:10 2025

@author: dx1
"""
import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sdf_0
import master_wass_metric_binary as mwmb
import window_sampling_algorithm as wsa
from scipy import optimize
import wass_CA_test as wcat
import double_lattice_gbs as dlgb

fp = '/Users/dx1/Research/microstructure_data/'

### create bipartite graph for all pixels ###
l = 50        # px side length of windows
m = int(l**2)  # px area of windows
iwB = mwmb.intrawindow_block(l)

#enlarged_slices =  np.load(fp+'3x_z_every5.npy')
#winds = wcat.sample_randomly(enlarged_slices, l, wn = 100)
#np.save(fp+'3x_p22_10slices.npy', winds)
#winds = 1 - np.load(fp+'3x_p22_10slices.npy') + 1

# slices = np.load(fp + 'r1_P22_gids.npy')[25::50][:10]
# gbslices = np.zeros((10,802,802))
# for i in range(10):
#     gbslices[i] = dlgb.double_lattice_gbs(slices[i])[1]

wn = 60
#inds = wcat.sample_randomly(gbslices, l, wn = wn)
#np.save(fp+'dbl_lattice_2x_p22_10slices.npy', winds)

winds = np.load(fp+'dbl_lattice_2x_p22_10slices.npy')

def serial_distances(A,B,iwB,l):
    n = np.shape(A)[0]
    dmat = np.zeros((n,n))
    if (A == B).all():
        for i in range(n):
            for j in range(0, n):
                dmat[i,j] = mwmb.window_distance(A[i],B[j],l,iwB)
        dmat += dmat.T
        for i in range(n):
            dmat[i,i] /= 2
    else: 
        for i in range(n):
            for j in range(n):
                dmat[i,j] = mwmb.window_distance(A[i],B[j],l,iwB)
    
    return dmat

def plot_distances(dmat):
    alabl = 'slice '
    materials = []
    for i in range(10):
        materials.append(str(i))
    
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(ds,cmap = 'viridis')
    
    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(materials)), labels=materials)
    ax1.set_yticks(np.arange(len(materials)), labels=materials)
    
    # Loop over geometric_data dimensions and create text annotations.
    for i in range(len(materials)):
        for j in range(len(materials)):
            text = ax1.text(j, i, dmat[i, j], ha="center", va="center", color="w", size=10)
            if dmat[i,j] > np.average(dmat):# and geometric_data[i,j] < 2:
                text = ax1.text(j, i, dmat[i, j], ha="center", va="center", color="k", size=10)
    ax1.set_title("Distance Between R1_P22 Slices [5, 10, .. 50]")
    fig1.tight_layout()
    #plt.savefig('./windows/EIBII/EIBIImatrix')
    plt.show()

    return


if False:
    distances = np.zeros((10,10))
    for i in range(10):
        wa0 = (wn*i)
        wa1 = int(wa0+wn/2)
        for j in range(i,10):
            wb0 = int(wn*(j+1)-wn/2)
            wb1 = wn*(j+1)
            A = winds[wa0:wa1]
            B = winds[wb0:wb1]
            
            wn2 = int(wn/2)
            dmat = np.zeros((wn2,wn2))
            for i0 in range(wn2):
                for j0 in range(i0,wn2):
                    dmat[i0,j0] = mwmb.window_distance(A[i0], B[j0], l, iwB)
            dmat += dmat.T
            for k in range(wn2):
                dmat[k,k] /= 2
            
            name = str(i)+str(j)
            np.save(fp+'/dmats_dbl_lattice/dmat' + name + '_dbl_p22_10slices.npy', dmat)

            rindy,cindy = optimize.linear_sum_assignment(dmat)
            distrib_dist = dmat[rindy, cindy].sum()/l**2/30
            distances[i,j] = distrib_dist
            print('layers ' + name + ': ' + str(distrib_dist))
     
            
if False:
    ds = np.load(fp+'zslice_1000w_50x50_3x_wass.npy')
    #for i in range(10):
    #    ds[i]/=ds[i,i]
    
    mindist = np.loadtxt(fp+'dmat_min_z.txt')
    mindmat = np.zeros((10,10))
    count = 0
    for i in range(10):
        for j in range(i,10):
            mindmat[i,j] = mindist[count]
            count += 1
    mindmat += mindmat.T
    for i in range(10):
        mindmat[i,i] /= 2
    
    plot_distances(np.around(mindmat,2))
    
    
if False:
    bigwindows = np.load(fp+'bigwindows.npy')
    bigwindow_dmat = np.load(fp+'bigwindow_dmat.npy')
    for k in range(1,10):
        for l in range(1,10):
            for i in range(25):
                for j in range(i+1,25):
                    A = bigwindows[k,i]
                    B = bigwindows[l,j]
                    bigwindow_dmat[k,l,i,j] = mwmb.window_distance(A,B,150,iwB)
                    print(k,l,i,j,bigwindow_dmat[k,l,i,j])
                np.save(fp+'bigwindow_dmat.npy',bigwindow_dmat)
    
if __name__ == '__main__':
    fp = '/Users/dx1/Research/microstructure_data/'
    l = 100        # px side length of windows
    m = int(l**2)  # px area of windows
    iwB = mwmb.intrawindow_block(l)
    
    take_new_slices = True
    if take_new_slices:
        p22 = np.load(fp + 'r1_P22_gids.npy')
        slices = p22[:,30::40,:]
        gbslices = np.zeros((10,401,401))
        for i in range(10):
            gbslices[i] = dlgb.double_lattice_gbs(slices[:,i,:])[0][100:501,:]
            #plt.figure()
            #plt.imshow(gbslices[i],'binary')
        #plt.show()
        wn = 7
        winds_y = wcat.sample_randomly(gbslices, l, wn = wn)
        np.save(fp+'dbl_lattice_p22_levelset_yaxis.npy', winds_y)
        
    #winds_z = np.load(fp+'dbl_lattice_p22_levelset_zaxis.npy')
    
    slices = []
    for i in range(10):
        slices.append(winds[i*wn:(i+1)*wn])
    
    compare_slices = False
    if compare_slices:
        for i in range(1):
            A = slices[i]
            for j in range(i,10):
                B = slices[j]
                dmat = serial_distances(A, B, iwB, l)
                plt.figure()
                plt.imshow(dmat)
                plt.show()
                np.save(fp+'dmats_p22/levelset/yaxis'+str(i)+str(j)+'_100px.npy', dmat)
                
    save_dmat = False
    if save_dmat:         
        full_dm = np.zeros((800,800))
        for i in range(10):
            for j in range(i,10):
                full_dm[i*80:(i+1)*80, j*80:(j+1)*80] = np.load(fp+'dmats_p22/levelset/yaxis'+str(i)+str(j)+'.npy')
        for i in range(1,800):
            full_dm[i,:i] = 0
        full_dm += full_dm.T
        plt.imshow(full_dm)
        plt.show()
        
        np.save(fp+'dmats_p22/levelset/full_dm_yaxis.npy',full_dm)
        
        full_dm = np.load(fp+'dmats_p22/levelset/full_dm_yaxis.npy')
    
    dif_axes = False
    if dif_axes:
        x_winds = np.load(fp+'dbl_lattice_p22_levelset_xaxis.npy')[0::5][:100]
        y_winds = np.load(fp+'dbl_lattice_p22_levelset_yaxis.npy')[0::5][:100]
        z_winds = np.load(fp+'dbl_lattice_p22_levelset.npy')[0::5][:100]
        axes = [x_winds, y_winds, z_winds]
        dij = np.zeros((3,3))
        for i in range(3):
            A = axes[i]
            for j in range(i+1,3):
                B = axes[j]
                dmat = serial_distances(A, B, iwB, l)
                r,c = optimize.linear_sum_assignment(dmat)
                dij[i,j] = dmat[r,c].sum()/100
    
                
                
    
#plot_distances(np.around(distances,2))