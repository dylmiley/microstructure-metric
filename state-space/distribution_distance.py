#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 14:49:27 2025

@author: dx1
"""
import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')
import numpy as np
from scipy import optimize
import mputil_0 as mpu
import matplotlib.pyplot as plt
import better_graph_maybe as bgm
import double_lattice_gbs as dlgb

def window_distance(A,B):
    l = np.shape(A)[0]
    dAB = bgm.make_custom_iwB(A, B)            
    r,c = optimize.linear_sum_assignment(dAB)
    d = dAB[r,c].sum()/l**2
    return d

def compare_distribs(cores, U, V):
    '''
    compute the distance between a distribution of windows
    
    Parameters
    ----------
    cores : int
        number of cores to use for multiprocessing
        
    U, V : lists of strings OR lists of nparrays 
        the filepath to the windows OR the windows

    iwB : nparray
        output of "intrawindow_block". the unreduced bipartite graph
            
    l : int 
        window length

    Returns
    -------
    distrib_dist : float
        solution to the AP, the distance between the window distributions.
    
    bipartite : nparray
        the distance matrix between the windows of each distribution, 
        in case you want to compute the distribution distance some other way.
    '''
    
    compare = window_distance
    if not mpu.mpOn: mpu.startMP(cores=cores)
    jobNum = 0
    kwargs = {}
    n1 = np.shape(U)[0]
    n2 = np.shape(V)[0]
    
    for i in range(n1):
        for j in range(n2):
            mpu.pushJobMP(((i, j), compare, [U[i], V[j]], kwargs), verbose = False)
            jobNum += 1 
    
    mpu.waitResMP(jobNum)
    resList = mpu.popResQueue()
    bipartite = np.zeros((n1, n2))

    for tup, dist in resList:
        i,j = tup
        bipartite[i,j] = dist
    rindy,cindy = optimize.linear_sum_assignment(bipartite)
    distrib_dist = bipartite[rindy, cindy].sum()
    
    return distrib_dist, bipartite


fp = '/Users/dx1/Research/microstructure_data/'
slicesP22 = np.load(fp + 'r1_P22_gids.npy')
slicesP63 = np.load(fp + 'r1_P63_gids.npy')
sln = 5
interval = 10
l = 100
gbsP22 = np.zeros((sln, l, l))
gbsP63 = np.zeros((sln, l, l))

for i in range(sln):
    gbsP22[i] = dlgb.double_lattice_gbs(slicesP22[i*interval][175:225,175:225])[1]    
    gbsP63[i] = dlgb.double_lattice_gbs(slicesP63[i*interval][175:225,175:225])[1]

cores = 3
U = gbsP22
V = gbsP63

d, graph = compare_distribs(cores, U, V)