#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 14:29:36 2024

@author: dylmiley

###########################################################################
# This is the Master file for comparing distributions of windows          #
#                                                                         #
# The primary internal files that are need are mputil and sym_wass_metric #
# mputil is a multiprocessing utility                                     #
# sym_wass_metric contains the functions needed to compute Wass. metric   #
#                                                                         #
################################## NOTE ###################################
# This version is meant to be easily accesible given that it doesnt       #
# require the installation of many different libraries e.g. lemonlibrary. #
# Therefore: this does not contain the most computationally efficient     #
# solver, nor does it use the entropic regularization and knoop sinkhorn  # 
# methods given that they are not strictly necessary for the binarized    #
# windows reported in this work.                                          #
# Use of the entropic reg. and knoop-snkhorn methods described in the     #
# manuscript is more theoretically sound and efficient but not strictly   #
# necessary for binarized winodws.                                        #
#                                                                         #
###########################################################################

file_to_array: creates nparrays for windows stored as csv files.

intrawindow_block: creates an nparray of an unreduced bipartite graph for two
                   windows of size l x l

full_bipartite: poses the similarity of two windows as an unbalanced 
                assignment problem
                
window_distance: computes the unbalanced wasserstein distance between windows

compare_distribs: computes the distance between two window distributions
"""

import sys
sys.path.append('../../src')
import numpy as np
import csv
from scipy import optimize
import mputil_0 as mpu
import matplotlib.pyplot as plt
import time

def file_to_array(filepath,l):
    '''
    Read in filepath to csv files containing windows of length l
    and convert them to a useable format AKA numpy arrays
    Parameters
    ----------
    filepath : str 
    
    l : int 
        window length

    Returns
    -------
    A: numpy array 
        a window

    '''
    A = np.zeros((l,l))
    i = 0
    j = 0
    with open(filepath, newline='') as csvfile:
        file = csv.reader(csvfile, delimiter=',')
        for row in file:
            for value in row:
                A[i,j] = value
                j+=1
            j = 0
            i+=1	
    
    # optional plotting 
    show_window = 0
    if show_window:
        plt.figure()
        plt.imshow(A,'binary')

    return(A)

def intrawindow_block(n):
    '''
    creates the block of the transport matrix for window - window transport.
    This block contains the path lengths for every node in two n x n windows.
    We only need to create this block one time and then copies of it can be used
    for each comparison.
    It is advisable to compute this once and then save the result to a .csv
    to avoid redundant computations
    
    Parameters
    ----------
    n: int  
        length of the square window 

    Returns
    -------
    bipartite: nparray 
        unreduced transport graph for assignment problem    
        read as input to "full_bipartite"
    '''
    
    n2 = n**2
    G = np.zeros((n2,n2))
    
    ''' make intrawindow transport graph '''
    a = np.arange(0,n)
    b = np.ones([n,1])
    
    ab = np.kron(a,b)
    ab -= ab.T
    ab = abs(ab)
    
    for i in range(n):
        G[:n, n*i:n*(i+1)] = ab + np.ones(np.shape(ab))*i
  
    G[:,:n] = G[:n,:].T
    
    for i in range(1,n2-1):
        G[n*i:, n*i:] = G[:-i*n, :-i*n]
    
    ''' dummy matrix '''
    D = np.ones((n,n))
    o = int(np.ceil(n/2))
    
    for i in range(1,o):
        D[i:n-i,i:n-i] += 1
        
    for i in range(1,o):
        D[n-i-1] = D[i]
    
    D = D.flatten()
    
    ### EDIT NEW Reservoir ###
#    D[:] = int(n/2) 
    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((n2+1,n2+1))
    bipartite[:n2, :n2] = G
    bipartite[:n2, n2] = D
    bipartite[n2, :n2] = D.T
    
    return bipartite


def make_custom_iwB(a,b):
    ''' window - window block '''
    n = np.shape(a)[0]
    n2 = n**2
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
    Da = np.copy(D)[indA[1]]
    Db = np.copy(D)[indB[1]]

    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((aij+bij+1,aij+bij+1))
    bipartite[:aij, :bij] = dAB
    bipartite[:aij, aij+bij] = Da
    bipartite[aij+bij, :bij] = Db.T
    
    return dAB 


def full_bipartite(Graph, A, B):
    '''
    Constructs bipartite graph for a pair of windows.
    In other words, we are phrasing the matching of windows as an assignment
    problem and creating the graph which lets us compute the solution.

    Parameters
    ----------
    Graph : nparray
        output of "intrawindow_block". the unreduced bipartite graph
        
    A : nparray
        a window with integer values
    B : nparray 
        a window with integer values

    Returns
    -------
    Graph : nparray
        the reduced bipartite graph, fully dense transport matrix
        which contains nodes for each window (A and B) as well as their 
        associated "boundaries" or "dummy points" or "reservoirs"
    '''
    
    A = A.flatten()
    B = B.flatten()

    # append 1 to both for the dummy pixel #
    A = np.append(A,1)
    B = np.append(B,1)
    A = A.astype(bool)
    B = B.astype(bool)
    
    # hit Graph with A and B to introduce 0's # 
    Graph = Graph[A].T[B]
    
    # add rest of dummy points #
    s = np.shape(Graph) 
    
    dummyA = np.zeros((s[0],s[0]))
    dummyA += Graph[:,s[1]-1]
    Graph = np.append(Graph, dummyA.T, axis=1)

    dummyB = np.zeros((s[1],s[0]+s[1]))    
    dummyB += Graph[s[0]-1, :]
    Graph = np.append(Graph, dummyB, axis=0)[:-2,:-2]
    
    return Graph

def window_distance(A, B, l, iwB = np.load('/Users/dx1/Research/microstructure_data/iwB150.npy')):
    '''
    compute the distance between a pair of windows
    
    Parameters
    ----------
    A, B : nparrays OR strings
        the windows for computing the distance between
        OR
        the filepaths to the windows stored as csvs
    
    l : int 
        window length

    iwB : nparray
        output of "intrawindow_block". the unreduced bipartite graph
        
    Returns
    -------
    dist: float
        solution to the AP, the distance between the windows.

    '''
    
    if type(A) is str:
        A = file_to_array(A,l)
        B = file_to_array(B,l)
        
    Ac = np.copy(A)
    Bc = np.copy(B)
    
    # optional plotting 
    show_window = 0
    if show_window:
        plt.figure()
        plt.imshow(Ac,'binary')
        plt.figure()
        plt.imshow(Bc, 'binary')    
    
    # remove overlapping mass, this match is trivial and does not need to be considered #
    overlap = np.where(Ac+Bc == 2)
    Ac[overlap] = 0
    Bc[overlap] = 0
    
    show_overlap_removed = 0
    if show_overlap_removed:
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(Ac,'binary')
        ax[1].imshow(Bc, 'binary')    
        
    # create the graph for the AP # 
    t1 = time.time()
    Graph = full_bipartite(iwB, Ac, Bc)
    t2 = time.time()
    # solve the assignment problem #    
    r,c = optimize.linear_sum_assignment(Graph)
    t3 = time.time()
    # full, unnormalized cost # 
    dist = Graph[r,c].sum()
    #print('time to make graph: ' + str(t2 - t1))
    #print('time to solve graph: ' + str(t3 - t2))
    
    '''
    ### optional unnecessary functionality ###
    # number of dummy points for A and B #
    n_dA = np.sum(Ac, dtype = int)
    n_dB = np.sum(Bc, dtype = int)
    
    # where rows and columns, i.e. A and B, get mapped to boundary: #
    cBd = c[np.where(c[np.where(r < n_dB)] >= n_dA)]
    rBd = r[np.where(c[np.where(r < n_dB)] >= n_dA)]
    
    cAd = c[np.where(c[np.where(r >= n_dB)] < n_dA)]
    rAd = r[np.where(c[np.where(r >= n_dB)] < n_dA)]    

    dBound = (Graph[rAd,cAd].sum() + Graph[rBd,cBd].sum())/l**2
    dWins = np.around(dist - dBound,4)
    '''

    return dist #, dBound, dWins, r, c, Ac, Bc

def compare_distribs(cores, U, V, iwB, l):
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
            mpu.pushJobMP(((i, j), compare, [U[i], V[j], l, iwB], kwargs), verbose = False)
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


