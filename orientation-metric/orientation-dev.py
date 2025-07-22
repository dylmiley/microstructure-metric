import sys
sys.path.append('../state-space/')

import numpy as np
import master_wass_metric_binary as mwmb

rng = np.random.default_rng(12345)


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


if '__name__' == __main__:
    l = 50        # px side length of windows
    m = int(l**2) # px area of windows
    iwB = mwmb.intrawindow_block(l)
    

