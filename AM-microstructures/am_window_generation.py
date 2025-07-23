#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 09:53:32 2025

@author: dx1
"""


import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')
import double_lattice_gbs as dlgb
import numpy as np
import window_sampling_algorithm as wsa
import matplotlib.pyplot as plt



def save_winds(gbs,wn,ws,sln):
    winds_sim = np.zeros((wn,ws,ws))
    for i in range(sln):
        wn_sl = int(wn/sln)
        inds_sim = wsa.find_valid_wnds(l,wn,wn_sl)[0]
        for j in range(wn_sl):
            a = inds_sim[0][j]
            b = inds_sim[1][j]
           # winds_sim[i*wn_sl + j] = gbs[i, a:a+ws, b:b+ws]
            winds_sim[i*wn_sl + j] = gbs[a:a+ws, b:b+ws]

    return winds_sim


sln = 100
start = 200
interval = 1
l = 3072
wn = 1
ws = 100
fp = '/Users/dx1/Research/microstructure_data/gerrys_set/'


create_windows_gerry = False
gbs = np.zeros((6, sln, l, l))

if create_windows_gerry:
    material_sets = ['r1_p22', 'r1_p61', 'r1_p63',
                     'r2_p22', 'r2_p61', 'r2_p63']
    for i in range(1):
        slices = np.load(fp+material_sets[i]+'.npy')[:,175:226,175:226]
        for j in range(sln):
            gbs[i,j] = dlgb.double_lattice_gbs(slices[start + j*interval])[1][:100,:100]
            plt.figure()
            plt.title(material_sets[i])
            plt.imshow(gbs[i,j])
        plt.show()
        np.save(fp+material_sets[i]+'_everyvoxel_gbs_xy.npy', save_winds(gbs[i],wn,ws,sln))

create_windows_ABCD = False
if create_windows_ABCD:
    fp+='TestMicrostructures/micro_'
    slicesA = np.load(fp+'A.npy')
    slicesB = np.load(fp+'B.npy')
    slicesC = np.load(fp+'C.npy')
    slicesD = np.load(fp+'D.npy')
    gbsA = np.zeros((sln, l, l))
    gbsB = np.zeros((sln, l, l))
    gbsC = np.zeros((sln, l, l))
    gbsD = np.zeros((sln, l, l))

    for i in range(sln):
        gbsA[i] = dlgb.double_lattice_gbs(slicesA[:, :, start + i*interval])[1]
        plt.figure()
        plt.title('A'+str(i))
        plt.imshow(gbsA[i])
        
        gbsB[i] = dlgb.double_lattice_gbs(slicesB[:, :, start + i*interval])[1]
        plt.figure()
        plt.title('B'+str(i))
        plt.imshow(gbsB[i])
        
        gbsC[i] = dlgb.double_lattice_gbs(slicesC[:, :, start + i*interval])[1]
        plt.figure()
        plt.title('C'+str(i))
        plt.imshow(gbsC[i])
        
        gbsD[i] = dlgb.double_lattice_gbs(slicesD[:, :, start + i*interval])[1]
        plt.figure()
        plt.title('D'+str(i))
        plt.imshow(gbsD[i])

plt.show()


save_windows = False
if save_windows:
    np.save(fp+'winds_A_xz.npy', save_winds(gbsA,wn,ws,sln))
    np.save(fp+'winds_B_xz.npy', save_winds(gbsB,wn,ws,sln))
    np.save(fp+'winds_C_xz.npy', save_winds(gbsC,wn,ws,sln))
    np.save(fp+'winds_D_xz.npy', save_winds(gbsD,wn,ws,sln))

