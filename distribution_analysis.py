#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 08:21:19 2025

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


def distribution_analysis(dmat):
    '''
    Randomly samples 
    
    Parameters
    ----------
    dmat : a distance matrix between microstructure windows

    Returns
    -------
    d_k : the distance between subsets of the window distrubtion from size
           1 to size k

    '''
    n = np.shape(dmat)[1]
    window_set = np.arange(n)
    d_k = np.zeros(n)
    for k in range(1,n):
        indA = np.zeros(n, dtype=bool)
        indB = np.zeros(n, dtype=bool)
        subsetA = rng.choice(window_set, replace = False, size=k)
        subsetB = rng.choice(window_set, replace = False, size=k)
        indA[subsetA] = 1
        indB[subsetB] = 1
        sub = dmat[indA].T[indB]
        r,c = optimize.linear_sum_assignment(sub, maximize = False)
        d_k[k] = sub[r,c].sum()/k
    
    return d_k


def plot_area_analysis(dmat, wn, ws, sn, sl_id):
    A = ws*sn
    plt.figure()
    t = 100
    newmax=0
    for i in range(sn):
        d_k = np.zeros(wn)
        for j in range(t):
            d_k += distribution_analysis(dmat[i])
        d_k/=t
        labl = str(i) 
        area = np.arange(wn)*ws**2
        plt.plot(area, d_k/ws**2, label = labl) 
        #plt.scatter(area, d_k/ws**2, label = labl) 
        if newmax < np.max(d_k/ws**2):
            newmax = np.max(d_k/ws**2)
            
    plt.xlim(ws**2, ws**2*(wn+3))
    plt.legend(loc='lower right')
    plt.title('Distance from slice ' + sl_id + ' w.r.t. sampled area')
    plt.ylim(0,newmax*1.1)
    plt.xlim(0,area[-1]*1.2)
    plt.show()
    return d_k


if __name__ == '__main__':
    plt.close('all')
    
    rng = np.random.default_rng(seed=None)
    fp ='/Users/dx1/Research/microstructure_data/'

    '''
    ### 10 slices taken from 5th vxl to 55th vxl every 10 vxl, 50x50 px per window, 50 windows per distribution from the 3x data.
    ### the double lattice is NOT implemented here. Less resolution. 
    '''
    if False:
        wn = 50
        ws = 50
        sn = 10
        
        dmat = np.zeros((sn,wn,wn))
        for i in range(sn):
            dmat[i] = (np.load(fp+'dmats_p22/3x/dmat'+str(0)+str(i)+'_3x_p22_10slices.npy'))
        
        plot_area_analysis(dmat, wn, ws, sn, '0')


    '''
    ### 10 slices taken from 5th vxl to 55th vxl every 10 vxl, 150 x 150 px per window, 25 windows per slice, not partitioned into distributions
    ### double lattice seems to have the effect of greatly increasing the resolution.
    '''
    if False:
        dm = np.load(fp+'bigwindow_dmat.npy')[0]
        wn = 25
        ws = 150
        sn = 10
        
        for i in range(sn):
            dm[i] += dm[i].T
            for j in range(wn):
                dm[i,j,j] = 100000
        plot_area_analysis(dm, wn, ws, sn, '0')
    
    
    '''
    ### 10 slices, taken from the 100th vxl to the 460th vxl, 50x50 px per window, 80 windows per slice, not partitiioned into distributions
    ### double lattice is used
    '''
    if False:
        dm = np.load(fp+'dmats_p22/levelset/full_dm_yaxis.npy')
        
        wn = 80
        ws = 50
        sn = 10

        sl_id = 9
        dm = dm[sl_id*80:(sl_id+1)*80,:]
        dm_res = np.zeros((10,80,80))
        for i in range(10):
            dm_res[i] = dm[:,i*80:(i+1)*80]
        for i in range(wn):
            #dm_res[:,i,i] /= 2
            dm_res[sl_id,i,i] = 00000

        plot_area_analysis(dm_res, wn, ws, sn, str(sl_id))


    '''
    ### Distance vs sampled area from single xy region of r1(p22 and p63);
    ### Windows sampled randomly from slices 300, 302, …, 308
    ### 7 windows from each slice.
    ### NOTE: At this sampling interval, substantial correlation between slices. 

    '''
    if False:
        dm = np.load(fp+'dmat_p22_p63.npy')
        
        wn = 35
        ws = 100
        sn = 2

        sl_id = 0
        dm = dm[sl_id*wn:(sl_id+1)*wn,:]
        dm_res = np.zeros((sn,wn,wn))

        for i in range(sn):
            dm_res[i] = dm[:,i*wn:(i+1)*wn]
        for i in range(wn):
            dm_res[:,i,i] = 0

        #dm_res = np.copy(dm) # zeros((sn,wn,wn))
        
        plot_area_analysis(dm_res, wn, ws, sn, str(sl_id))
    
    if True:
        

        wn = 3
        ws = 100
        sn = 6
        dm = np.load(fp+'dmat_p22-p63_'+str(sn)+'slices'+str(90)+'interval'+str(wn*sn)+'windsperslices.npy')
        wn=6
        sl_id = 6
        dm = dm[sl_id*wn:(sl_id+1)*wn,:]
        dm_res = np.zeros((sn,int(wn),int(wn)))

        for i in range(sn):
            dm_res[i] = dm[:,i*wn:(i+1)*wn]

        #dm_res = np.copy(dm) # zeros((sn,wn,wn))
        
        plot_area_analysis(dm_res, wn, ws, sn, str(sl_id))
    