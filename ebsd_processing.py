#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 08:20:24 2025

@author: dx1
"""

import numpy as np
import csv
import sys
sys.path.append('/Users/dx1/Research/new_metric_code/')
sys.path.append('/Users/dx1/Research/microstructure_data/')
import matplotlib.pyplot as plt
import double_lattice_gbs as dlgb
rng = np.random.default_rng()

fp = '/Users/dx1/Research/microstructure_data/'
# data = np.zeros((786433,8))
# i = 0
# with open(fp+'P22_EBSD.csv', newline='') as csvfile:
#     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
#     for row in spamreader:
#         j = 0
#         for entry in row:
#             try:
#                 data[i,j] = float(entry)
#                 j+=1
#             except:
#                 pass
#         i+=1       
#np.save(fp+'p22_ebsd.npy',data[1:])

data = np.load(fp+'p22_ebsd.npy')
data_grid = np.reshape(data, (768,1024,8))

data_grid = np.load(fp+'p22_ebsd_cleaned.npy')
plt.figure()
plt.imshow(data_grid[:,:,5:])
plt.show()


def replace_nulls(data):
    data_grid = data
    ind = np.where(data_grid[:,:,:3] == [0,0,0])
    n = np.shape(ind)[1]
    neighborlist = np.array(([[-1,1], [0,1], [1,1], 
                           [-1,0], [1,0],
                           [-1,-1], [-1,0], [-1,1]]))
    
    #dgcp = np.copy(data_grid)
    for i in range(n):
        neighbors = np.zeros((8,8))
        for k in range(8):
            a = (ind[0][i] + neighborlist[k,0])%768
            b = (ind[1][i] + neighborlist[k,1])%1024
            neighbors[k] = data_grid[a, b]
        val = neighbors[rng.integers(0,8)]
        data_grid[ind[0][i], ind[1][i]] = val
       # data_grid[ind[0][i], ind[1][i], :3] = [1,1,1]

            
    plt.figure()
    plt.imshow(data_grid[:,:,5:])
    plt.show()
    return data_grid

def to_gray(data):
    dgcp = np.zeros((768,1024))
    dgcp = 0.299 * data_grid[:,:,5] + 0.587 * data_grid[:,:,6] + 0.114 * data_grid[:,:,7]
    plt.figure()
    plt.imshow(dgcp)
    plt.show()
 
    return dgcp
    

def mono_grains(data, thresh):
    data_cp = np.copy(data)
    for i in range(1,767):
        for j in range(1,1023):
            if abs(data[i,j] - data[i+1,j]) < thresh*data[i,j]:
                data_cp[i+1,j] = data_cp[i,j]
                
            if abs(data[i,j] - data[i,j+1]) < thresh*data[i,j]:
                data_cp[i,j+1] = data_cp[i,j]
                
            if abs(data[i,j] - data[i+1,j+1]) < thresh*data[i,j]:
                data_cp[i+1,j+1] = data_cp[i,j]   
                
            if abs(data[i,j] - data[i-1,j]) < thresh*data[i,j]:
                data_cp[i-1,j] = data_cp[i,j]
                
            if abs(data[i,j] - data[i,j-1]) < thresh*data[i,j]:
                data_cp[i,j-1] = data_cp[i,j]
                
            if abs(data[i,j] - data[i+1,j-1]) < thresh*data[i,j]:
                data_cp[i+1,j-1] = data_cp[i,j]   
            
            if abs(data[i,j] - data[i-1,j-1]) < thresh*data[i,j]:
                data_cp[i-1,j-1] = data_cp[i,j]
            
            if abs(data[i,j] - data[i-1,j+1]) < thresh*data[i,j]:
                data_cp[i-1,j+1] = data_cp[i,j]
    return data_cp


def double_lattice_gbs(layer):
    m,n = np.shape(layer)
    layer_double = np.zeros((m*2, n*2))
    
    for i in range(m):
        for j in range(n):
            layer_double[i*2:(i+1)*2, j*2:(j+1)*2] = layer[i,j]
    
    gb_dbl = np.zeros(np.shape(layer_double))
    
    for i in range(m*2-1):
        for j in range(n*2-1):
            a = layer_double[i:i+2, j:j+2]
            if (a[0] != a[1]).all():
                gb_dbl[i, j] = 1
            if (a[:,0] != a[:,1]).all():
                gb_dbl[i, j] = 1
                
    return gb_dbl


while np.shape(np.where(data_grid[:,:,:3] == [0,0,0]))[1]:
    data_grid = replace_nulls(data_grid)

np.save(fp+'p22_ebsd_cleaned.npy',data_grid)

data_gray = to_gray(data_grid)
gbs = double_lattice_gbs((data_gray[:100,:100]*10).astype(int))[1]
plt.figure()
plt.imshow(gbs[:200,:200])
plt.show()

data_mono = mono_grains(data_gray, 0.01)*10

plt.figure()
plt.imshow(data_mono)
plt.show()

gbs1 = dlgb.double_lattice_gbs(data_mono[:100,:100].astype(int))[1]
plt.figure()
plt.imshow(gbs1[:200,:200])
plt.show()
