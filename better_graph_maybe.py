#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 16:23:26 2025

@author: dx1
"""

import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')
import double_lattice_gbs as dlgb
import imageio
import numpy as np


def make_custom_iwB(a,b):
    ''' window - window block '''
    n = np.shape(a)[0]
    a_b = a - b
    indA = np.where(a_b == 1)
    aij = np.shape(indA)[1]
    
    indB = np.where(a_b == -1)
    bij = np.shape(indB)[1]
    
    dAB = np.zeros((aij,bij))
    for i in range(aij):
        for j in range(bij):
            dAB[i,j] = abs(indA[0][i] - indB[0][j]) + abs(indA[1][i] - indB[1][j])
    
    ''' dummy matrix '''
    D = np.ones((n,n))
    o = int(np.ceil(n/2))
    
    for i in range(1,o):
        D[i:n-i,i:n-i] += 1
        
    for i in range(1,o):
        D[n-i-1] = D[i]
    
    D = D.flatten()
    Da = np.copy(D)[n*indA[0] + indA[1]]
    Db = np.copy(D)[n*indB[0] + indB[1]]
    nB = np.shape(dAB)
    for i in range(nB[0]):
        for j in range(nB[1]):
            if dAB[i,j] > Da[i] + Db[j]:
                dAB[i,j] = 2*o
    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((aij+bij,aij+bij))
    bipartite[:aij, :bij] = dAB
    bipartite[:aij, bij:] += np.reshape(Da, (aij,1))
    bipartite[aij:, :bij] += Db
    
    return bipartite


if __name__ ==  '__main__':
    import matplotlib.pyplot as plt
    import time 
    from scipy import optimize
    import master_wass_metric_binary as mwmb
    import matplotlib
    n = 100
    l = n #2*n
    fp = '/Users/dx1/Research/microstructure_data/'
    #p63 = np.load(fp+'p63_gbs_6slices_sl-id0_every90_windows.npy')[:,:n,:n]
    sl_n = 5
    sl_id = 0
    interval = 10
    A = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')[:,:100,:100]
    B = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')[:,:100,:100]
    p22_gbs = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
    p63_gbs = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
   
    
    test_0 = False
    if test_0:
        a = np.zeros((l,l))
        a[:n, :n] = p63[0]
        a[n:, :n] = p63[0]
        a[:n, n:] = p63[0]
        a[n:, n:] = p63[0]
        
        
        b = np.zeros((l,l))
        b[:n, :n] = p63[1]
        b[n:, :n] = p63[1]
        b[:n, n:] = p63[1]
        b[n:, n:] = p63[1]
    
    
    test_1 = False
    if test_1:
        a = p63[3]
        b = p63[17]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","yellow"])
    
        a_b = a - b
        plt.figure()
        plt.imshow(a + 2*b, cmap = cmap)
        plt.show()
        
        s = time.time()
        dAB = make_custom_iwB(a, b)
        print(n,'x', n,'windows:', time.time() - s)
    
        
        r,c = optimize.linear_sum_assignment(dAB)
        d_new = dAB[r,c].sum()/n**2
        print('time to form and solve (new):', time.time() - s)
        print('distance between windows:', d_new)
    
    
    run_0 = False
    if run_0:
        ref = A[8]
        plt.figure()
        plt.imshow(ref)
        plt.show()
        for i in range(20):
            dAB = make_custom_iwB(ref, B[i])            
            r,c = optimize.linear_sum_assignment(dAB)
            d_new = dAB[r,c].sum()/l**2
            print(d_new)
            plt.figure()
            plt.title(d_new)
            plt.imshow(ref - B[i])
            plt.show()
            
    
    run_1 = True
    if run_1:
        slicesP22 = np.load(fp + 'r1_P22_gids.npy')
        slicesP63 = np.load(fp + 'r1_P63_gids.npy')
        sln = 50
        interval = 10
        l = 100
        gbsP22 = np.zeros((sln, l, l))
        gbsP63 = np.zeros((sln, l, l))

        for i in range(sln):
            gbsP22[i] = dlgb.double_lattice_gbs(slicesP22[i*interval][175:225,175:225])[1]
            plt.figure()
            plt.imshow(gbsP22[i])
            
            gbsP63[i] = dlgb.double_lattice_gbs(slicesP63[i*interval][175:225,175:225])[1]
            plt.figure()
            plt.imshow(gbsP63[i])
            
        plt.show()
        dmat_P22_P63 = np.zeros((sln,sln))
        for i in range(sln):
            ref = gbsP22[i]
            for j in range(i+1,sln):
                dAB = make_custom_iwB(ref, gbsP22[j])            
                r,c = optimize.linear_sum_assignment(dAB)
                d_new = dAB[r,c].sum()/l**2
                dmat_P22_P63[i,j] = d_new
        
        plt.figure()
        plt.imshow(dmat_P22_P63)
        plt.colorbar()
        plt.show()
        

        np.save(fp+'/dmats_p22/0to490_every10_2x.npy', dmat_P22_P63)
        imageio.mimsave(fp+'p63_dlgb_0to490_every10_2x.gif', gbsP63*200, loop = 4)
