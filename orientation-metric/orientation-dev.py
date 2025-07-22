import sys
sys.path.append('../state-space/')

import numpy as np
import master_wass_metric_binary as mwmb

def quatmult(q1, q2):
    ### quaternion multiplication ### 
    ind1 = np.array([0, 1, 2, 3])
    ind2 = np.array([[0, 1, 2, 3],
                     [1, 0, 3, 2],
                     [2, 3, 0, 1],
                     [3, 2, 1, 0]])
        
    sign = np.array([[1,-1,-1,-1], [1,1,-1,1], [1,1,1,-1], [1,-1,1,1]])
    prod = np.zeros(4)
    c = 0
    for row in ind2: 
        prod[c] = np.dot(q1[ind1], q2[row]*sign[c])
        c+=1
    return prod


def misorientation(q1, q2):
    ### calculation of rotation quaternion for q1 and q2 ###
    q2[1:] *= -1
    r = quatmult(q1, q2)
    return r

def texture_graph(A,B):
    ### the graph for orientation difference for two windows, A and B ###
    n = np.shape(A)[0]
    G = np.zeros((n**2, n**2))
    uniqueA = np.unique(A.reshape(-1,4), axis=0)
    uniqueB = np.unique(B.reshape(-1,4), axis=0)
    

    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    for entry in rotation_list:
                        if 
                    G[]

if '__name__' == __main__:
    rng = np.random.default_rng()
    
    ### create transport landscape for a 5x5 window ###
    l = 5         # px side length of windows
    m = int(l**2) # px area of windows
    iwB = mwmb.intrawindow_block(l)

    ### create 2 random orientations ###
    q1 = rng.random(size=4)
    q1 = q1 / np.linalg.norm(q1)
    q2 = rng.random(size=4)
    q2 = q2 / np.linalg.norm(q2)

    ### create two windows with randomly assigned q1 and q2 ###
    A = np.ones((l,l,4))
    A[:,:]*=q1
    B = np.ones((l,l,4))
    ijA = rng.integers(0, 5, size=(rng.integers(0,25), 2))
    for ij in ijA:
        A[ij[0], ij[1], :] = q2
    ijB = rng.integers(0, 5, size=(rng.integers(0,25), 2))
    for ij in ijB:
        B[ij[0], ij[1], :] = q2
    
