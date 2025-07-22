#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 11:53:21 2025

@author: dylmiley
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import master_wass_metric_binary as mwmb
import master_window_generator as mwg
import csv

def generate_new_distribs(names, samples, window_num, l):
    '''
    ### window file parameters ###
    '''
    ustr_num = len(samples)
    slices = np.arange(0,100,20)    # slices taken from these voxel locations

    """
    generate window distributions by sampling from dream3d files. 
    """
    window_distribution_1 = np.zeros((ustr_num, window_num, l, l))
    
    for i in range(ustr_num):
        wd = mwg.generate_distribution(samples[i], window_num, l, slices)
        for j in range(window_num):
            window_distribution_1[i][j] = wd[j]  
    
    Vs = []
    for i in range(np.shape(window_distribution_1)[0]):
        Vs.append(window_distribution_1[i])

    return Vs

def plot_windows(U):
    for i in range(np.shape(U)[0]):
        fig, ax = plt.subplots()
        plt.imshow(U[i],cmap='binary')
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        # plt.savefig('./windows/eq0',bbox_inches='tight')
        plt.show()
        # write_csvs(U[i], 'anisotropy/EI/EIz' + str(i))

    return

def write_csvs(A, name):
    file = './csvs/' + name + '.csv'
    np.savetxt(file, A, delimiter=",")
    return

def sweep_compare(A, Bs, Bsize, nameA, nameB):
    # A
    # Bs = [V[5], V[6], V[7], V[8], V[9]]
    dmat = np.zeros((Bsize,1))
        
    for j in range(Bsize):   
        B = Bs[j]    
        write_csvs(B, nameB + str(j))
        wd = mwmb.window_distance(A, B, l, iwB)
        dis = np.around(wd/l**2,3)
        print('the distance between windows A and B is:', dis)
        fig, ax = plt.subplots()
        winds = A + 2*B
        
        cmap = ListedColormap(["white","lightsalmon","mediumblue","navy"])
        
        plt.imshow(winds, cmap=cmap)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        # ax.text(5, 5, 'd = ' + str(dis), bbox={'facecolor': 'white', 'pad': 10})
        
        # plt.savefig('./windows/' + nameA + nameB + '/' + nameA + nameB + str(i) + str(j), bbox_inches='tight')
        plt.show()
        dmat[j] = dis
    
    print('----------------------')
    return dmat

def plot_distances(dmat):
    
    alabl = 'eq small '
    blabl = 'bimodal large '
    materials = [alabl+str(0),alabl+str(1),alabl+str(2),alabl+str(3),alabl+str(4)]
    materialsB = [blabl+str(0),blabl+str(1),blabl+str(2),blabl+str(3),blabl+str(4)]
    
    
    fig1, ax1 = plt.subplots()
    im = ax1.imshow(dmat,cmap = 'viridis')
    
    # Show all ticks and label them with the respective list entries
    ax1.set_xticks(np.arange(len(materials)), labels=materialsB)
    ax1.set_yticks(np.arange(len(materials)), labels=materials)
    
    # Loop over geometric_data dimensions and create text annotations.
    for i in range(len(materials)):
        for j in range(len(materials)):
            text = ax1.text(j, i, dmat[i, j], ha="center", va="center", color="w", size=16)
            if dmat[i,j] > 0.9:# and geometric_data[i,j] < 2:
                text = ax1.text(j, i, dmat[i, j], ha="center", va="center", color="k", size=16)
    ax1.set_title("Confusion Matrix for Materials A:E")
    fig1.tight_layout()
    plt.savefig('./windows/EIBII/EIBIImatrix')
    plt.show()

    return


names = ['E1','R1','B1','E2','B2']  # directory name for each microstructure 

fpath = './' #../d3samples/'             # file path to dream3d files !!!           

# samples = [fpath + 'EI.dream3d', fpath + 'EII.dream3d', fpath + 'BI.dream3d', fpath + 'BII.dream3d', fpath + 'RI.dream3d']

window_num = 50     # number of windows to generate for each distribution

samples = [fpath + 'EI.dream3d', fpath + 'RI.dream3d']


matlnames = ['EI', 'RI'] # ['EI', 'EII', 'RI', 'BI', 'BII']


'''
### set the window parameters and create the unreduced transport graph ###
'''
l = 50        # px side length of windows
m = int(l**2) # px area of windows
iwB = mwmb.intrawindow_block(l)

# Vs = generate_new_distribs(names, samples, window_num, l)


# eq0 = np.genfromtxt('./csvs/EI8.csv',delimiter=',')

EIxyz = np.zeros((3,50,50,50))
for i in range(3):
    for j in range(50):
        xyz = ['x','y','z']
        EIxyz[i,j] = np.genfromtxt('./csvs/anisotropy/EI/EI'+xyz[i]+str(j)+'.csv',delimiter=',')

RIxyz = np.zeros((3,50,50,50))
for i in range(3):
    for j in range(50):
        xyz = ['x','y','z']
        RIxyz[i,j] = np.genfromtxt('./csvs/anisotropy/RI/RI'+xyz[i]+str(j)+'.csv',delimiter=',')
        
cores = 7     # number of cores to use in mp
 
# create window distributions, U and V #
U = EIxyz[0]
V = RIxyz[2]

# compute distance between distributions   #
# and regularize by window size and number #

#d, mat = mwmb.compare_distribs(cores, U, V, iwB, l)
# write_csvs(mat, 'anisotropy/EIx-RIz')

#import scipy
#mat = np.genfromtxt('./csvs/anisotropy/EIx-RIy.csv',delimiter=',')
#rindy,cindy = scipy.optimize.linear_sum_assignment(mat)
#d = mat[rindy, cindy].sum()/125000
print(d/125000)

sweep_compare = 0
if sweep_compare:
    full_dmat = np.zeros((5,9))
    for i in range(5):
        V = Vs[i][1:]
        dmat = sweep_compare(Vs[0][0], V, 9, matlnames[0], matlnames[i])
        full_dmat[i,:] = dmat[:,0]


