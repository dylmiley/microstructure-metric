#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs 10 April 2025

@author: dylmiley
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import optimize
import master_window_generator as mwg

'''
import sys
sys.path.append('/home/dylmiley/EMS285/Python/HW6')
sys.path.append('/home/dylmiley/EMS285/Python/HW8')
import hw6_p3
import hw8_p3
'''

def read_csv(fp, n):
    """
    Parameters
    ----------
    fp : string
        the file path to the desired .csv
    n : integer 
        row length of the .csvs

    Returns
    -------
    A : float
        the csv transcribed into a variable just for us

    """
    
    # creates an (n,n) size array of zeros #
    A = np.zeros((n,n), dtype = int) 
    # opens a file at the location 'fp' and reads it #
    csv1 = open(fp, 'r')
    # loops from 0 to n, updating 'i' along the way #
    for i in range(n):
        # Updates the ith row of 'A' with the csv1, parsing the ith row into 
        # individual columns 
        A[i,:] = csv1.readline().split(',')
    # closes the csv1 file so that memory is not wasted 
    csv1.close()    
    # yields the value of A as an output of the funciton #
    return A

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
    # D[:] = int(n/2) 
    D[:] = 4*n
    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((n2+1,n2+1))
    bipartite[:n2, :n2] = G
    bipartite[:n2, n2] = D
    bipartite[n2, :n2] = D.T
    
    return bipartite

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

"""
def intrawindow_block(n):
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
    '''
    for i in range(1,n):
        G[:n, n2*i:n2*(i+1)] = G[:n, :n2] + np.ones((n,n2))*i
    ''' 
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
    ### COMMENT FOR N/2 RES ###
    # D[:] = int(n/2)     
    
    #D = D.reshape((n2,1))
    #D = D*np.ones((n2,n2))
    
    ''' combined matrix, full bipartite graph '''
    bipartite = np.zeros((n2+1,n2+1))
    bipartite[:n2, :n2] = G
    bipartite[:n2, n2] = D
    bipartite[n2, :n2] = D.T
    return bipartite

def full_bipartite(Graph, A, B):
    '''
    phase is an int which describes the phase that the graph is constructed for. 
    entries of matrices with values equal to phase are isolated by preserving those 
    entries and discarding all others?
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
    shpGraph = np.shape(Graph) - np.array([1,1])
    
    # add rest of dummy points #
    s = np.shape(Graph)
    dummyA = np.ones((s[0],s[0])) 
    
    dummyA = np.zeros((s[0],s[0]))
    dummyA += Graph[:,s[1]-1]
    Graph = np.append(Graph, dummyA.T, axis=1)
    
    dummyB = np.zeros((s[1],s[0]+s[1]))
    dummyB += Graph[s[0]-1, :]
    dummyB = np.ones((s[1],s[0]+s[1])) 

    Graph = np.append(Graph, dummyB, axis=0)
    return Graph, shpGraph
"""


def distance(A,B,n,iwB,phase,**kwargs):
    i = str(kwargs['index'])
    
    Ac = np.copy(A)
    Bc = np.copy(B)
    
    # hone in on specific phase #
    Ac[np.where(A != phase)] = 0
    Ac[np.where(A == phase)] = 1
    Bc[np.where(B != phase)] = 0
    Bc[np.where(B == phase)] = 1
    
    '''
    plt.figure()
    plt.axis('off')
    plt.imshow(Ac, cmap = 'binary')
    plt.savefig(path + 'a_phase' + i + '.svg')
    
    plt.figure()
    plt.axis('off')
    plt.imshow(Bc, cmap = 'binary')    
    plt.savefig(path + 'b_phase' + i + '.svg')
    '''
    
    # remove overlapping mass, this match is trivial and does not need to be considered #
    overlap = np.where(Ac+Bc == 2)
    Ac[overlap] = 0
    Bc[overlap] = 0
    
    #print(np.shape(overlap))

    '''
    ### EDITS ###
    Graph,shpGraph = full_bipartite(iwB, Ac, Bc)

    # number of dummy points for A and B #
    n_dA = shpGraph[1]
    n_dB = shpGraph[0]
    '''
    n_dA = 4*n
    n_dB = 4*n
    Graph = full_bipartite(iwB, Ac, Bc)
    # solve the assignment problem #    
    r,c = optimize.linear_sum_assignment(Graph)
    
    rnot = r[np.where(Graph[r,c] != 4*n)]
    cnot = c[np.where(Graph[r,c] != 4*n)]
    # rnot1 = rnot[np.where(Graph[rnot,cnot] != 0)]
    # cnot1 = cnot[np.where(Graph[rnot,cnot] != 0)]
    dWins = Graph[rnot,cnot].sum()
    

    # where rows and columns, i.e. A and B, get mapped to boundary: #
    # rAd = r[np.where(c == n_dA)] 
    # cAd = c[np.where(c == n_dA)]     
    # rBd = r[np.where(r == n_dB)]
    # cBd = c[np.where(r == n_dB)]


    # dBound = (Graph[rAd,cAd].sum() + Graph[rBd,cBd].sum()) # /n**2
    
    # full cost # 
    dist = Graph[r,c].sum() # /n**2
    
    dBound = np.around(dist - dWins,4)
    
    return dist, dBound, dWins, r, c #, shpGraph

n = 100
iwB = intrawindow_block(n)
# plt.figure()
# plt.imshow(iwB)


def phase_distance(A,B,n,iwB,Q):
    dwdb = np.zeros((Q,3))
    phases = np.arange(Q)
    
    for i in range(np.shape(phases)[0]):
        dist, dBound, dWins, r, c, shp = distance(A,B,n,iwB,phases[i],index=i)
        '''
        print('Phase: ' + str(phases[i]))
        print('total phase distance:', dist)
        print('distance of intrawindow transport:', dWins)
        print('distance of windows to boundary:', dBound)
        print('% of total distance (W, B):', '(' + str(np.around(dWins/dist*100, 2)) + ', ' + str(np.around(dBound/dist*100, 2)) + ')')
        print('-------------------')
        '''
        '''
        x,y = np.meshgrid(r,r)
        Z = np.zeros(np.shape(x), dtype=int)
        for j in range(np.shape(c)[0]): Z[j,c[j]] = 1
        fig = plt.figure()
        #plt.imshow(Z, cmap = 'seismic')
        plt.scatter(r,-c,color='k',marker='.')
        plt.plot(r,-1*np.ones(np.shape(r)[0])*shp[1], color = 'k')
        plt.plot(np.ones(np.shape(r)[0])*shp[0],-r, color = 'k')
        plt.savefig(path+'xport_map'+str(i)+'.svg')
        '''
        dwdb[i] = dWins, dBound, dist
    return dwdb.T

windowNum = 100
slices = np.arange(0,100,20)
fpath = './'
samples = [fpath + 'EI.dream3d', fpath + 'EII.dream3d', fpath + 'BI.dream3d', 
           fpath + 'BII.dream3d', fpath + 'RI.dream3d']

if False:
    windows = np.zeros((np.shape(samples)[0], windowNum, n, n))
    for i in range(2):
        windows[i] = mwg.generate_distribution(samples[i+2], windowNum, n, slices)

if False:
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","yellow"])
    avgwindist = np.zeros(10)
    for i in range(10):
        A = windows[0,i] - 1
        B =  windows[1,i] - 1
        a,b,c = phase_distance(A, B, n, iwB, 2)
        print(c)
        plt.figure()
        plt.imshow(A + 2*B, cmap = cmap)
        plt.show()
        avgwindist[i] = c[0]
        
    print(np.average(avgwindist))

""" CASE STUDIES: Micrographs obtained from online databases, phase structure """
import cv2

### Seperator data obtained from ETHZ,https://www.research-collection.ethz.ch/handle/20.500.11850/265085
image = cv2.imread('pp1615_ss.png')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshpp1615 = cv2.threshold(gray, 70, 255 , 0)
plt.figure()
plt.imshow(threshpp1615, cmap = 'binary')

image = cv2.imread('pp1615_ss_0.png')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshpp1615_0 = cv2.threshold(gray, 70, 255 , 0)
plt.figure()
plt.imshow(threshpp1615_0, cmap = 'binary')

### Graphite electrode data obtained from ETHZ, https://www.research-collection.ethz.ch/handle/20.500.11850/224851
image = cv2.imread('l1bin.png')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshl1bin = cv2.threshold(gray, 70, 255 , 0)
threshl1bin = abs(threshl1bin - 255)
plt.figure()
plt.imshow(threshl1bin, cmap = 'binary')

###  Graphite electrode data obtained from ETHZ, https://www.research-collection.ethz.ch/handle/20.500.11850/224851
image = cv2.imread('l2bin.png')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshl2bin = cv2.threshold(gray, 70, 255 , 0) 
threshl2bin = abs(threshl2bin - 255)

plt.figure()
plt.imshow(threshl2bin, cmap = 'binary')

### NMC data obtained from ETHZ, https://app.data-archive.ethz.ch/delivery/action/collectionViewer.do?collectionId=2735309&operation=viewCollection&displayType=list
image = cv2.imread('NMC_90wt_0bar_168.tif')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshNMC = cv2.threshold(gray, 70, 255 , 0)
plt.figure()
plt.imshow(threshNMC, cmap = 'binary')

### NMC data obtained from ETHZ, https://app.data-archive.ethz.ch/delivery/action/collectionViewer.do?collectionId=2735309&operation=viewCollection&displayType=list
image = cv2.imread('NMC_90wt_2000bar_028.tif')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, threshNMC90b = cv2.threshold(gray, 70, 255 , 0)
plt.figure()
plt.imshow(threshNMC90b, cmap = 'binary')

### STELLITE20 IMAGE TAKEN FROM https://www.schmitz-metallographie.de/en/gefuge/stellite-20-centrifugally-casting/
image = cv2.imread('Stellit20_001.jpg')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh20 = cv2.threshold(gray, 70, 255 , 0)
thresh20 = thresh20[50:500, :]
plt.figure()
plt.imshow(thresh20, cmap = 'binary')

### STELLITE4 IMAGE TAKEN FROM https://www.schmitz-metallographie.de/en/gefuge/stellite-4-investment-casting/
image = cv2.imread('Stellit4_001.jpg')
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh4 = cv2.threshold(gray, 70, 255 , 0)
thresh4 = thresh4[50:500, :]
plt.figure()
plt.imshow(thresh4, cmap = 'binary')


window_number = 10

### EXTRACT WINDOWS FROM STELLITE
windows_stellite20  = np.zeros((window_number,n,n))
# for i in range(3,6):    
#     windows_stellite20[i] = np.genfromtxt('macks_poster/stellite' + str(i) + '.csv', delimiter = ',')[:n,:n]

windows_stellite20A = np.zeros((window_number,n,n))
windows_stellite20B = np.zeros((window_number,n,n))

windows_stellite4 = np.zeros((window_number,n,n))

### EXTRACT WINDOWS FROM NMC and GRAPHITE ELECTRODE and SEPERATOR
windows_NMC    = np.zeros((window_number,n,n))
windows_NMC_1  = np.zeros((window_number,n,n))
windows_NMC90b = np.zeros((window_number,n,n))
windows_l1bin  = np.zeros((window_number,n,n))
windows_l2bin  = np.zeros((window_number,n,n))
windows_threshpp1615   = np.zeros((window_number,n,n))
windows_threshpp1615_0 = np.zeros((window_number,n,n))

stellite = True
if stellite:
    for i in range(window_number):
        nthresh = np.shape(thresh20)
        j  = np.random.randint((nthresh[0]-n))
        k  = np.random.randint((nthresh[1]-n))
        k1 = np.random.randint((int(nthresh[1]-n)/2))
        k2 = np.random.randint((int(nthresh[1]-n)/2),nthresh[1]-n)
        windows_stellite20[i]  = thresh20[j:j+n, k:k+n]
        windows_stellite20A[i] = thresh20[j:j+n, k1:k1+n]
        windows_stellite20B[i] = thresh20[j:j+n, k2:k2+n]    
    
    for i in range(window_number):
        nthresh = np.shape(thresh4)
        j = np.random.randint((nthresh[0]-n))
        k = np.random.randint((nthresh[1]-n))
        windows_stellite4[i] = thresh4[j:j+n, k:k+n]

NMC = True
if NMC:
    for i in range(window_number):
        nthresh = np.shape(threshNMC)
        j = np.random.randint((nthresh[0]-n))
        k = np.random.randint((nthresh[1]-n))
        j1 = np.random.randint((nthresh[0]-n))
        k1 = np.random.randint((nthresh[1]-n))
        
        windows_NMC[i] = threshNMC[j:j+n, k:k+n]
        windows_NMC_1[i] = threshNMC[j1:j1+n, k1:k1+n]
        windows_NMC90b[i] = threshNMC90b[j:j+n, k:k+n]
        plt.figure()
        plt.imshow(windows_NMC[i], 'binary')
        plt.title('NMC 0bar, window ' + str(i))
        plt.show()
        plt.figure()
        plt.imshow(windows_NMC90b[i], 'binary')
        plt.title('NMC 90bar, window ' + str(i))
        plt.show()

Graphite = True
if Graphite:
    for i in range(window_number):
        nthresh = np.shape(threshl1bin)
        j = np.random.randint((nthresh[0]-n))
        k = np.random.randint((nthresh[1]-n))
        windows_l1bin[i] = threshl1bin[j:j+n, k:k+n]
        windows_l2bin[i] = threshl2bin[j:j+n, k:k+n]
        plt.figure()
        plt.imshow(windows_l1bin[i], 'binary')
        plt.title('Graphite electrode l1, window ' + str(i))
        plt.show()
        plt.figure()
        plt.imshow(windows_l2bin[i], 'binary')
        plt.title('Graphite electrode l2, window ' + str(i))
        plt.show()

Seperator = True
if Seperator:
    for i in range(window_number):
        nthresh = np.shape(threshpp1615)
        j = np.random.randint((nthresh[0]-n))
        k = np.random.randint((nthresh[1]-n))
        windows_threshpp1615[i] = threshpp1615[j:j+n, k:k+n]
        windows_threshpp1615_0[i] = threshpp1615_0[j:j+n, k:k+n]
        plt.figure()
        plt.imshow(windows_l1bin[i], 'binary')
        plt.title('Seperator , window ' + str(i))
        plt.show()
        plt.figure()
        plt.imshow(windows_l2bin[i], 'binary')
        plt.title('Seperator 0, window ' + str(i))
        plt.show()


structs = [windows_stellite20, windows_stellite20A, windows_stellite20B,
           windows_stellite4, windows_NMC, windows_NMC_1, windows_NMC90b, 
           windows_l1bin, windows_l2bin]

names = ['stel20', 'stel20A', 'stel20B', 'stel4', 
         'NMC0', 'NMC1', 'NMC2K'
         'graph0', 'graph1']

def compare_structs(matl_A, matl_B, n, iwB, window_number):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red","blue","slategrey"])    
    avgwindist  = np.zeros((window_number,window_number))
    excessphase = np.zeros((window_number,window_number))
    for i in range(window_number):
        for j in range(i, window_number):
            A = matl_A[i]
            B = matl_B[j]
            dAB = distance(A, B, n, iwB, 0, index=i)
            avgwindist[i,j]  = dAB[2]
            avgwindist[j,i]  = dAB[2]
            excessphase[i,j] = dAB[1]
            excessphase[j,i] = dAB[1]
        
    excessphase /= (4*n)
    plt.figure()
    plt.imshow(avgwindist)
    plt.show()
    r,c  = optimize.linear_sum_assignment((avgwindist/np.max(avgwindist) + excessphase/np.max(excessphase)))
    wind = avgwindist[r,c]
    exce = excessphase[r,c]
    dist   = np.average(avgwindist[r,c])
    excess = np.average(excessphase[r,c])
    print('optimal dist.', dist, 'excess phase', excess)
    
    plt.figure()
    x = np.reshape(avgwindist,window_number**2)
    y = np.reshape(excessphase,window_number**2)
    plt.scatter(x,y)
    #plt.scatter(wind,exce)
    plt.show()
    plt.imshow(avgwindist + excessphase)
    print('distribution distance:', (dist+excess)/n**2)
    
    # for i in range(10):
    #     A = matl_A[r[i]]
    #     B = matl_B[c[i]]
    #     dAB = distance(A, B, n, iwB, 0, index=i)
    #     # print(dAB[:3])
    #     plt.figure()
    #     plt.imshow(A, cmap = 'BuGn')
    #     plt.figure()
    #     plt.imshow(B, cmap = 'BuGn')
    #     plt.figure()
    #     plt.imshow(A + 2*B, cmap = cmap)
    #     print('Ratio of total mass:', np.round(np.sum(A) / np.sum(B),3))
    
    return avgwindist, excessphase, r, c


"""
wind_exce = np.zeros((10,20))
distributions = np.zeros((8,8))
for i in range(8):
    for j in range(i+1,8):
        wind_exce[:,:10], wind_exce[:,10:20], r, c = compare_structs(structs[i], structs[j], n, iwB, window_number)
        distributions[i,j] = np.average((wind_exce[:,:10] + wind_exce[:,10:20])[r,c])/n**2
        np.savetxt('window_distances_' + names[i] + '_' + names[j] + '.csv', wind_exce)
"""   








 ### OLD CODE THAT USES PHASE FIELD AND MCGG SIMS DATA ### 

# E8[np.where(D8==4)]=8

#wins = [A4,B4,C4,D4]
   
# names = ['A','B','C','D','E']
# comps = ['ab','ac','bc','ae','bc','bd','be','cd','ce','de']
# c=0

# if 0:
#     A4[np.where(A4 == 1)] = 0
#     A4[np.where(A4 > 1)] = 1
#     B4[np.where(B4 == 1)] = 0
#     B4[np.where(B4 > 1)] = 1
#     C4[np.where(C4 == 1)] = 0
#     C4[np.where(C4 > 1)] = 1
#     D4[np.where(D4 == 1)] = 0
#     D4[np.where(D4 > 1)] = 1



# for i in range(3):
#     for j in range(3):
#         print(names[i],names[j])
#         ab = np.around(phase_distance(wins[i], wins[j], n, iwB),2)
#         print(np.average(ab[2,:]))
#         plt.scatter(np.arange(9),ab[2,:], label = comps[c])
#         plt.plot(np.arange(9),ab[2,:])
#         plt.legend()
#         c+=1

# for i in range(3):
#     plt.figure()
#     plt.axis('off')
#     plt.imshow(wins[i])

# plt.show()

# plt.figure()
# b4 = B4 + 2
# c4 = C4 + 4
# plt.imshow(A4+b4, cmap = 'binary')
# plt.figure()
# plt.imshow(A4+c4, cmap = 'binary')
# plt.figure()
# plt.imshow(b4+c4, cmap = 'binary')