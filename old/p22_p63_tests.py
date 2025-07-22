#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:38:46 2025

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
import distribution_analysis as da
import P22_compare_slices as p22cs
import better_graph_maybe as bgm

if __name__ == '__main__':
    fp = '/Users/dx1/Research/microstructure_data/'
    l = 200        # px side length of windows
    m = int(l**2)  # px area of windows
    # iwB = mwmb.intrawindow_block(l)
    sl_id = 0
    interval = 10
    sl_n = 5
    
    take_new_slices = True
    if take_new_slices:
        p22 = np.load(fp + 'r1_P22_gids.npy')
        p63 = np.load(fp + 'r1_P63_gids.npy')
        
        p22_slx_300 = p22[sl_id::interval,:,:][:sl_n]
        p63_slx_300 = p63[sl_id::interval,:,:][:sl_n]
        p22_gbs = np.zeros((sl_n,802,802))
        p63_gbs = np.zeros((sl_n,802,802))
        
        for i in range(sl_n):
            p22_gbs[i] = dlgb.double_lattice_gbs(p22_slx_300[i])[1]
            p63_gbs[i] = dlgb.double_lattice_gbs(p63_slx_300[i])[1]

        if False:
            for i in range(sl_n):
                plt.figure()
                plt.imshow(p22_gbs[i],'binary')
                plt.show()
                
                plt.figure()
                plt.imshow(p63_gbs[i],'binary')
                plt.show()
        
        if False:
            np.save(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy', p22_gbs)
            np.save(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy', p63_gbs)
    
    p22_gbs = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
    p63_gbs = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
    
    new_windows = False
    if new_windows:
        wn = 4
        winds_p63 = wcat.sample_randomly(p63_gbs, l, wn = wn)
        winds_p22 = wcat.sample_randomly(p22_gbs, l, wn = wn)
        #np.save(fp+'dbl_lattice_p22_levelset_yaxis.npy', winds_y)
        wn = wn*sl_n
        
        for i in range(wn):
            plt.figure()
            plt.title('P63 window ' + str(i))
            plt.imshow(winds_p63[i],'binary')
        for i in range(wn):
            plt.figure()
            plt.title('P22 window ' + str(i))
            plt.imshow(winds_p22[i],'binary')
        plt.show()
        if True:
            np.save(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy', winds_p22)
            np.save(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy', winds_p63)

    winds_p22 = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')
    winds_p63 = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')

    A = winds_p22
    B = winds_p63
    for i in range(10,20):
        a = A[0]
        b = B[i]
        dAB = bgm.make_custom_iwB(a, b)
        r,c = optimize.linear_sum_assignment(dAB)
        d = dAB[r,c].sum()/l**2
        print(d)
    
    compare = False
    if compare:
        A = winds_p22
        B = winds_p63
        dmat_p22_p63 = np.zeros((2*wn,2*wn))
        dmat_p22_p63[:wn,:wn] = p22cs.serial_distances(A, A, iwB, l)
        dmat_p22_p63[:wn,wn:] = p22cs.serial_distances(A, B, iwB, l)
        dmat_p22_p63[wn:,wn:] = p22cs.serial_distances(B, B, iwB, l)
        dmat_p22_p63[wn:, :wn] = dmat_p22_p63[:wn,wn:].T
        
        plt.figure()
        plt.imshow(dmat_p22_p63)
        plt.show()
        for i in range(2*wn):
            dmat_p22_p63[i,i] += 100000
            
        r22,c22 = optimize.linear_sum_assignment(dmat_p22_p63[:wn,:wn])
        d22self = dmat_p22_p63[:wn,:wn][r22,c22].sum()/wn/l**2
        r63,c63 = optimize.linear_sum_assignment(dmat_p22_p63[wn:,wn:])
        d63self = dmat_p22_p63[wn:,wn:][r63,c63].sum()/wn/l**2
        r,c = optimize.linear_sum_assignment(dmat_p22_p63[:wn,wn:])
        d22_63 = dmat_p22_p63[:wn,wn:][r,c].sum()/wn/l**2
        
        np.save(fp+'dmat_p22-p63_'+str(sl_n)+'slices'+str(interval)+'interval'+str(wn)+'windsperslices', dmat_p22_p63)
        





