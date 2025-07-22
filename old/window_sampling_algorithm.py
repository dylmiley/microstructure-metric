#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 19:35:47 2024

@author: dylmiley
"""
import numpy as np
import matplotlib.pyplot as plt


def gen_windows(m, n, wn):
    indices = np.zeros((2,wn),dtype = int)
    count = 0
    for i in range(wn):
        success = 0
        its = 0
        while indices[0,i] == 0 and indices[1,i] == 0:
            its += 1
            k,l = np.random.randint(0,m-n,2)
            for j in range(count+1):
                a = abs(indices[0,j] - k)
                b = abs(indices[1,j] - l)
                if a > n or b > n:
                    success = 1
                else:
                    success = 0
                    break
            if success == 1:
                indices[:,i] = k,l
                
            # quit trying after M attempts to place a window down #
            if its > 1000:
                return indices, success
        count += 1
    return indices, success

def find_valid_wnds(m, n, wn):
    success = 0
    its = 0
    while success == 0: 
        indices, success = gen_windows(m, n, wn)
        its += 1
        
        # quit algorithm after N attempts to complete the sampling #
        if its > 100:
            # print('failure')   
            return indices, 0
    # print('Number of iterations:',its)    
    return indices, 1

if __name__ == '__main__':
    m = 500 # micrograph side length
    n = 40  # window size
    wn = 100 # number of windows in distribution

    ### edit these parameters to choose how many steps to run the algorithm for ###
    start = 1.5
    stop = 2.5
    steps = 20
    ###
    
    ms = np.floor((n**2*wn*np.linspace(start, stop, steps))**0.5)
    success = np.zeros(len(ms))
    for i in range(len(ms)):
        m = int(ms[i])
        print(m)
        indices, success[i] = find_valid_wnds(m, n, wn)    
        
        graph = np.zeros((m,m))
        for j in range(wn):    
            i0 = indices[0,j]
            j0 = indices[1,j]
            graph[i0:i0+n,j0:j0+n] += 1
        print(np.max(graph))
        if np.max(graph) > 1:
            success[i] = 0
        
        plt.figure()
        plt.imshow(graph, cmap = 'binary')
    
    plt.figure()
    rat = ms**2/n**2/wn
    plt.plot(rat,success)
