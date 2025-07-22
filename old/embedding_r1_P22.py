#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:12:27 2025

@author: dx1
"""

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True,threshold=np.inf)
from matplotlib.colors import ListedColormap as lcm

#dmat = np.load('dmatxz_150px.npy')
fp = '/Users/dx1/Research/microstructure_data/'
dmat = np.load(fp+'3x_z_dmat.npy')
ds = np.load(fp+'zslice_1000w_50x50_3x_wass.npy')

if False:
    lap = -1/np.copy(dmat)
    for i in range(100):
        lap[i,i] = 0
        lap[i,i] = -sum(lap[:,i])
    
    plt.figure()
    plt.imshow(lap)
    plt.show()
        
    l,x = linalg.eig(lap)
    
    c = np.zeros(100)
    for i in range(10):
        c[i*10:(i+1)*10] = i
    plt.figure()
    plt.scatter(x[1,:], x[2,:], c = c)
    #plt.xlim(-0.2,0.2)
    #plt.ylim(-0.2,0.2)
    plt.show()
    
    wij = np.copy(dmat)
    t = 2
    wij = np.e**-(wij**2/t)
    
    lap = -1/np.copy(wij)
    for i in range(100):
        lap[i,i] = 0
        lap[i,i] = -sum(lap[:,i])
    
    plt.figure()
    plt.imshow(lap)
    plt.show()
    
    plt.close('all')    
    l,x = linalg.eig(lap)
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[np.where(l==np.sort(l)[1]),:], x[np.where(l==np.sort(l)[2]),:], x[np.where(l==np.sort(l)[3]),:], c = c)
    ax.set_xlim(-0.1,0.1)
    ax.set_ylim(-0.1,0.1)
    #ax.ylim(-0.1,0.1)
    plt.show()

if False:
    wij = np.copy(ds)
    t = 10
    wij = np.e**-(wij**2/t)
    
    lap = -1/np.copy(wij)
    for i in range(10):
        lap[i,i] = 0
        lap[i,i] = -sum(lap[:,i])
    l,x = linalg.eig(lap)
    
    plt.figure()
    plt.scatter(x[2,:], x[4,:], c = np.arange(10))
    plt.show()
    
    plt.imshow(ds)
    
    dmats = np.zeros((300,300))
    for i in range(10):
        for j in range(i,10):
            dmats[i*30:(i+1)*30, j*30:(j+1)*30] = (np.load(fp+'dmats_dbl_lattice/dmat'+str(i)+str(j)+'_dbl_p22_10slices.npy'))
    
    for i in range(300):
        for j in range(i,300):
            dmats[j,i] = dmats[i,j]
    
    fp = '/Users/dx1/Research/microstructure_data/'
    dmats = np.load(fp+'bigwindow_dmat.npy')[0,0]
    plt.figure()
    plt.imshow(dmats)
    plt.show()

if False:
    wij = np.copy(dmats/150**2)
    t = 100
    wij = np.e**-(wij**2/t)
    
    lap = -1/np.copy(wij)
    for i in range(25):
        #lap[i,i] = 0
        lap[i,i] = -sum(lap[:,i])
    l,x = linalg.eig(lap)
    
    plt.close('all')
    
    c = np.arange(25)
    #c = np.zeros(300)
    #for i in range(300):
    #    c[i*50:(i+1)*50] = i
        
    plt.figure()
    plt.scatter(x[np.where(abs(l)==np.sort(abs(l))[0]),:], x[np.where(abs(l)==np.sort(abs(l))[1]),:], c = c) #494
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x[np.where(abs(l)==np.sort(abs(l))[1]),:], x[np.where(abs(l)==np.sort(abs(l))[2]),:], x[np.where(abs(l)==np.sort(abs(l))[3]),:], c = c)
    
    # bigwindows = np.load(fp+'bigwindows.npy')
    # for i in range(25):
    #     plt.figure()
    #     plt.imshow(bigwindows[0,i], 'binary')
    #     plt.show()

dmat = np.load(fp+'dmats_p22/levelset/full_dm.npy')
wij = np.copy(dmat/50**2)
t = 10
wij = np.e**-(wij**2/t)
n= np.shape(dmat)[0]
wn = 80
lap = -1/np.copy(wij)
for i in range(n):
    lap[i,i] = -sum(lap[:,i])
l,x = linalg.eig(lap)

plt.close('all')
c = np.zeros(n)
for i in range(10):
    c[i*80:(i+1)*80] = i
cmap = lcm(['darkviolet','navy','olive','darkorange','maroon',
            'violet','dodgerblue','yellow','sandybrown','red'])
xax = x[np.where(abs(l)==np.sort(abs(l))[1]),:]
yax = x[np.where(abs(l)==np.sort(abs(l))[2]),:]
zax = x[np.where(abs(l)==np.sort(abs(l))[3]),:]

plt.figure()
plt.xlim(-0.025,0.025)
plt.ylim(-0.025,0.025)
plt.scatter(xax, yax, c = c, cmap = cmap) 
plt.colorbar()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(xax, yax, zax, c = c, cmap = cmap)
ax.set_xlim(-0.025,0.025)
ax.set_ylim(-0.025,0.025)
ax.set_zlim(-0.025,0.025)

plt.show()