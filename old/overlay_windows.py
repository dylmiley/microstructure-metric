#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 08:40:02 2025

@author: dx1
"""
import numpy as np
import matplotlib.pyplot as plt

def overlay_windows(micrograph, window, micrograph_size, window_size):
    for i in range(micrograph_size - window_size):
        for j in range(micrograph_size - window_size):
            if (micrograph[i:i+window_size,j:j+window_size] == window).all():
                print(i,j)
                break
            else:
                continue
            break
        else:
            continue
        break
    return i,j

if __name__ == '__main__':
    sl_n = 5
    sl_id = 0
    interval = 10
    fp = '/Users/dx1/Research/microstructure_data/'
    A = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')
    B = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'_windows.npy')
    p22_gbs = np.load(fp+'p22_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
    p63_gbs = np.load(fp+'p63_gbs_'+str(sl_n)+'slices_sl-id'+str(sl_id)+'_every'+str(interval)+'.npy')
   
    plt.figure()
    plt.imshow(p22_gbs[0])

    m = np.shape(p22_gbs)[1]
    ws = np.shape(A[0])[0]
    for k in range(4,5):
        for l in range(1):
            i,j = overlay_windows(p22_gbs[k], A[l+(k*4)], m, ws)    
            p22_gbs[k,i:i+200,j:j+200] += A[l+(k*4)]
    
        plt.figure()
        plt.imshow(p22_gbs[k])#,'binary')
        plt.show() 
   
    
    plt.figure()
    plt.imshow(p63_gbs[0])

    m = np.shape(p63_gbs)[1]
    ws = np.shape(A[0])[0]
    for k in range(0):
        for l in range(4):
            i,j = overlay_windows(p63_gbs[k], B[l+(k*4)], m, ws)    
            p63_gbs[k,i:i+200,j:j+200] += B[l+(k*4)]
    
        plt.figure()
        plt.imshow(p63_gbs[k],'binary')
        plt.show()