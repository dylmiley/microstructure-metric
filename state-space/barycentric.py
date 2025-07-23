#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 17:57:33 2025

@author: dx1
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import itertools

def simplex_volume(B):
    '''
    calculates the volume of an n simplex based on the edge lengths,
    using the Cayley-Menger determinent
    pairwise distances are given in B
    '''
    n = np.shape(B)[0] 
    N = n-1
    Bb = np.ones((n+1,n+1))
    Bb[n,n] = 0
    Bb [:n,:n] = B**2
    det = np.linalg.det(Bb)
    vol = (pow(-1,n)/(2**N * math.factorial(N)**2) * det)**0.5
    
    return vol

def barycentric_coordinates(dmat_vertices,distP):
    '''
    calculates the barrycentric coordinates using volume ratios
    dmat_vertices is the pairwise distance matrix between each vertex
    of the bounding triangle,
    distP is the set of distances from some point P to each of the vertices
    '''
    n = np.shape(dmat_vertices)[1]
    vol_bound = simplex_volume(dmat_vertices)
    barrys = np.zeros(n)
    for i in range(n):
        d = np.copy(dmat_vertices)
        P = np.copy(distP)
        P[i] = 0
        d[i] = P
        d[:,i] = P 
        vol_P_simplex = simplex_volume(d)
        barrys[i] = vol_P_simplex / vol_bound
        
    return barrys

def fix_unsigned_volume(barrys, table):
    '''
    the volume is unsigned due to use of Cayley-Wegner determinent to 
    calculate volume. Sign needs to be assigned to indicate whether the point
    exists beyond a given face of the simplex (-) or within a given face (+)
    ---
    this is a brute force method for assigning sign. a truth table is iterated 
    through to see which combinations of sign result in the sum of the
    the coordinates being equal to one.    
    '''
    barryscp = np.copy(barrys)   
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i][j]:
                barryscp[j] *= -1
        if np.isclose(sum(barryscp), 1):
            barrys = barryscp
            return barrys         
        barryscp = np.copy(barrys)
    return barrys
    
rng = np.random.default_rng()

n = 10
vertices = rng.random(size=(n,n+1))
#vertices = np.array([[0,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]])
print('bounding simplex: \n', vertices)

plotting = False
if plotting:
    plt.figure()
    plt.scatter(vertices[0,:], vertices[1,:])
    for i in range(3):
        plt.plot([vertices[0,i], vertices[0,(i+1)%3]], [vertices[1,i], vertices[1,(i+1)%3]])
    
B = np.zeros((n+1,n+1))
for i in range(n+1):
    for j in range(i+1,n+1):
        B[i,j] = np.linalg.norm(vertices[:,i] - vertices[:,j])

B += B.T
vol = simplex_volume(B)
print('volume of bounding simplex: ',vol)


table = list(itertools.product([False, True], repeat=n+1))
    
err = 0
coords = np.zeros(n)
count = 0
fail = 0

while fail < 1:
    P = rng.random(size = (n))
    distP = np.zeros(n+1)
    for j in range(n+1):
        distP[j] = np.linalg.norm(P - vertices[:,j])

    barrys = barycentric_coordinates(B, distP)
    if not np.isclose(sum(barrys), 1):
        barrys = fix_unsigned_volume(barrys,table)
        if not np.isclose(sum(barrys), 1):
            fail += 1
            count -= 1
        
    for i in range(n):
        coords[i] = np.dot(vertices[i,:], barrys)
    err += abs(sum(P - coords))
    # print(P - coords)
    # print('barycentric coordinates: ',barrys)

    if plotting:
        plt.scatter(P[0],P[1], c = ['black'])
        plt.scatter(coords[0],coords[1], c = ['red'], marker = '.')
    count += 1
    if count > 100:
        break
    
print('success: ', count)
print('error: ', err)

if plotting: plt.show()