#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 11:25:02 2025

@author: dx1
"""


import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')
import double_lattice_gbs as dlgb
import imageio
import numpy as np
import window_sampling_algorithm as wsa

import matplotlib.pyplot as plt
import time 
from scipy import optimize
import master_wass_metric_binary as mwmb
import matplotlib

def W_dmat(dmat, wn, ws, n):
    ABs = []
    cs = []
    for i in range(wn*n):
        dmat[i,i] = 1000000
    W_abcd = np.zeros((n,n))
    for i in range(n):
        for j in range(i,n):
            A = np.copy(dmat[i*wn:(i+1)*wn, j*wn:(j+1)*wn])
            r,c = optimize.linear_sum_assignment(A, maximize = False)
            ABs.append(A)
            cs.append(c)
            W_abcd[i,j] = A[r,c].sum()/wn
            W_abcd[j,i] = W_abcd[i,j]
            
    W_abcd/=ws**2
    W_abcd = np.around(W_abcd,3)
    
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(W_abcd,cmap = 'viridis')
    
    labels = ['R1 P22', 'R1 P61', 'R1 P63', 'R2 P22', 'R2 P61', 'R2 P63', 'EBSD P22']
    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(labels)), labels=labels, size = 13)
    ax1.set_yticks(np.arange(len(labels)), labels=labels, size = 13)
    
    # Loop over geometric_data dimensions and create text annotations.
    for i in range(len(labels)):
        for j in range(len(labels)):
            text = ax1.text(j, i, W_abcd[i, j], ha="center", va="center", color="w", size=16)
            if W_abcd[i,j] > 1.5:# and geometric_data[i,j] < 2:
                text = ax1.text(j, i, W_abcd[i, j], ha="center", va="center", color="k", size=16)
    ax1.set_title("Confusion Matrix for All Samples", size=16)
    fig1.tight_layout()
    #ax1.colorbar()
    #plt.show()
    
    return W_abcd, ABs, cs
    
def overlay(winds_A, winds_B, AB, c):
    n = np.shape(c)[0]
    r = np.arange(n)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","yellow"])
    for i in range(n):
        plt.figure()
        plt.imshow(winds_A[r[i]] + 2*winds_B[c[i]], cmap = cmap)
        plt.title(AB[r[i], c[i]])    
    plt.show()
    

def tile_winds(winds, title):
    bigA = np.ones((1010,1010))
    for i in range(10):
        for j in range(10):
            bigA[i*100+i:(i+1)*100+i, j*100+j:(j+1)*100+j] = winds[i*10+j]
    plt.imshow(bigA)
    plt.axis('off')
    #plt.title(title + ' windows')
    plt.show()
    
if __name__ == '__main__':
    
    # winds_A = np.load('./microstructure_data/TestMicrostructures/micro_winds_A_yz.npy')
    # winds_B = np.load('./microstructure_data/TestMicrostructures/micro_winds_B_yz.npy')
    # winds_C = np.load('./microstructure_data/TestMicrostructures/micro_winds_C_yz.npy')
    # winds_D = np.load('./microstructure_data/TestMicrostructures/micro_winds_D_yz.npy')
    
    # tile_winds(winds_D, 'D')
    
    # dmat_xy = np.load('./microstructure_data/TestMicrostructures/dmat_ABCD_400_windows.npy')
    # dmat_xy+=dmat_xy.T
    # dmat_yz = np.load('./microstructure_data/TestMicrostructures/dmat_ABCD_400_windows_yz.npy')
    # dmat_yz+=dmat_yz.T
    
    fp = '/Users/dx1/Research/microstructure_data/gerrys_set/'
    dmat = np.load(fp+'dmat_allgerrys_700_windows_xy.npy')
    dmat[:, 600:] = np.roll(np.load(fp+'dmat_ebsd_all_2x.npy')[:100,:].T, 600, axis = 0)
    dmat[600:, 600:] = dmat[600:, 600:].T
    # dmat = np.load(fp+'dmat_p22_100_everyvoxel_windows_xy.npy')
    dmat += dmat.T
    #dmat = dmat[:100]
    plt.imshow(dmat)
    plt.show()

    W_abcd, ABs, cs = W_dmat(dmat, 100, 100, 7)
    plt.imshow(W_abcd)
    plt.show()
    overlay(ebsd_windows, ebsd_windows, ebsd_dist/100**2, cs[27])
    
