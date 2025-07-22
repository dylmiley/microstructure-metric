#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:59:58 2024

@author: dylmiley

This python script can be used to calculate the Wasserstein distance between
pairs of windows as well as between microstructure distributions

two auxillary files are needed:
    1.) master_wass_metric_binary is used to compute the wasserstein distance.
    2.) master_window_generator is used to sample windows from a dream3d file.

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import master_wass_metric_binary as mwmb
import master_window_generator as mwg

'''
### window file parameters ###
'''
fpath = './' #../d3samples/'             # file path to dream3d files !!!           
names = ['E1','R1','B1','E2','B2']  # directory name for each microstructure 

'''
### set the window parameters and create the unreduced transport graph ###
'''
l = 50        # px side length of windows
m = int(l**2) # px area of windows
iwB = mwmb.intrawindow_block(l)

window_num = 10     # number of windows to generate for each distribution
slices = np.arange(0,100,20)    # slices taken from these voxel locations
samples = [fpath + 'EI.dream3d', fpath + 'EII.dream3d', fpath + 'BI.dream3d', 
           fpath + 'BII.dream3d', fpath + 'RI.dream3d']
ustr_num = len(samples)

"""
generate window distributions by sampling from dream3d files. 
"""
window_distribution_1 = np.zeros((ustr_num, window_num, l, l))

for i in range(ustr_num):
    wd = mwg.generate_distribution(samples[i], window_num, l, slices)
    for j in range(window_num):
        window_distribution_1[i][j] = wd[j]  

EI  = window_distribution_1[0]
EII = window_distribution_1[1]
BI  = window_distribution_1[2]
BII = window_distribution_1[3]
RI  = window_distribution_1[4]


'''
### example computation between two windows #######
'''
distance_A_B = 0    # on/off switch
if distance_A_B:
    A = EI[0]       # a window
    B = BI[0]    # another window

    # convert csv data to nparrays #
    if type(A) is str:
        A = mwmb.file_to_array(A,l)
        B = mwmb.file_to_array(B,l)
    
    # compute distance #
    wd = mwmb.window_distance(A, B, l, iwB)
    print('the distance between windows A and B is:', wd)


'''
### example computation between two window distributions #######
'''
distance_U_V = 0    # on/off switch
if distance_U_V:
    cores = 10     # number of cores to use in mp
    n = 50         # the size of window distribs to create    
 
    # create window distributions, U and V #
    U = EI
    V = BI
    
    # compute distance between distributions   #
    # and regularize by window size and number #
    d, mat = mwmb.compare_distribs(cores, U, V, iwB, l)
    d /= (m*np.shape(U)[0])
    mat /= (m*np.shape(U)[0])
    print(np.around(d,3))

'''
### compute the distances between all microstructures in ###
'''
distance_all_U_V = 0    # on/off switch
if distance_all_U_V:
    cores = 10      # number of cores to use in mp
    index = 0       # an offset for creating different window distribs
    dmat = np.zeros((5,5))      # distance matrix for all 5 microstructures
    ustrs1 = [EI,EII,BII,RI,BI] # list of microstructures
    
    # generate a second list of microstructures #
    window_distribution_2 = np.zeros((ustr_num, window_num, l, l))

    for i in range(ustr_num):
        wd2 = mwg.generate_distribution(samples[i], window_num, l, slices)
        for j in range(window_num):
            window_distribution_2[i][j] = wd2[j]  

    EI2  = window_distribution_2[0]
    EII2 = window_distribution_2[1]
    BI2  = window_distribution_2[2]
    BII2 = window_distribution_2[3]
    RI2  = window_distribution_2[4]
    
    ustrs2 = [EI2,EII2,BII2,RI2,BI2]
    for i in range(5):
        for j in range(5):
            # window distributions, U and V, from two seperate samplings of microstructure #
            U = ustrs1[i]       
            V = ustrs2[j]
            
            # compute distance between distributions   #
            # and regularize by window size and number #
            d, mat = mwmb.compare_distribs(cores, U, V, iwB, l)
            d /= (m*len(U))
            mat /= (m*len(U))
            print(d)
            
            # assign distribution distances to distance matrix, with rounding
            dmat[i,j] = np.around(d,3)
        print(dmat[i])
    print(dmat)
