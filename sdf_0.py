import numpy as np


def uFuncL1(a, b, fij, h):
    return min(a,b) + fij*h

def uFuncL2(a, b, fij, h):
    #where a is the minimum of the x neighbor and b is the minimum of the y neighbor
    if np.abs(a - b) > fij*h:
        return min(a,b) + fij*h #if u/d/l/r motion is shorter
    else:
        return (a + b + np.sqrt(2*(fij**2)*(h**2) - (a-b)**2))/2 #if diagnols are shorter

def fastSweepSDF(X, h=1, f=lambda i,j:1, boundaryId=None, grainId=None, uFunc = 'L1', maxIter = 1e3, convThresh= 1e-2, onlyGrain=False, edgeBoundary=True, verbose=False, **kwargs):
    """A fast sweep algorithm based on Zhao 2004.
        X is a nxm 2D array of values.
        h is the length of each cell. By defualt h = 1.
        f(i,j) is the mass function of cell i,j. By default f(i,j) = 1
        boundaryId is the value in X which represents the boundaries
        grainId is the value in X which represents which grain to isolate. Then it will find the edge pixles and define those as the boundary.
        uFunc defines how shortest distances are calculated. "L1", "L2" or a function.
        edgeBoundary defines whether the edges of X are considered boundaries 
        
        
        Note, by default this assumes that the edges of X are considered boundaries.

    """
    if not isinstance(verbose, bool): raise TypeError('fastSweepSDF: verbose is a bool. Currently {}'.format(verbose))
    if not isinstance(onlyGrain, bool): raise TypeError('fastSweepSDF: onlyGrain is a bool. Currently {}'.format(onlyGrain))
    if not isinstance(edgeBoundary, bool): raise TypeError('fastSweepSDF: edgeBoundary is a bool. Currently {}'.format(edgeBoundary))
    if not isinstance(boundaryId, type(None)) and not isinstance(grainId, type(None)):
        if boundaryId == grainId or boundaryId in grainId: raise TypeError('fastSweepSDF: boundaryId cannot be the same as grainId') 

    if uFunc == 'L1': uFunc =uFuncL1
    elif uFunc == 'L2': uFunc =uFuncL2
    
    xsize,ysize = X.shape

    if edgeBoundary: U = np.zeros((xsize+2,ysize+2))+np.inf #Cut off the edges later
    else: U = np.zeros((xsize, ysize)) + np.inf

    #Create boundary array
    Xp = np.zeros([xsize+2, ysize+2])
    if isinstance(boundaryId, int): Xp = Xp + boundaryId #sets edges to boundary id
    Xp[1:xsize+1, 1:ysize+1] = X

    #Create a list of all boundary pixels
    edgePixels = set() 
    grainPixels = set()

    #If Boundaries are specified
    if isinstance(boundaryId, int):
        if edgeBoundary: Xbound = Xp
        else: Xbound = Xp[1:xsize+1, 1:ysize+1]

        xedge, yedge = np.nonzero(Xbound == boundaryId)
        edgePixels.update(set(zip(xedge,yedge)))
        if isinstance(grainId, type(None)):
            xgrain,ygrain = np.nonzero(Xbound != boundaryId)
            grainPixels.update(set(zip(xgrain, ygrain))) #-1 to get rid of the edges
    #If a certain grain or grains are specified
    if isinstance(grainId, (int, list, tuple, np.ndarray)):
        if isinstance(grainId, int): grainIds = [grainId]
        else: grainIds = grainId

        for gid in grainIds:
            tfmat = Xp == gid
            xgrain, ygrain = np.nonzero(tfmat) #get all the grain ids
            grainPixels.update(set(zip(xgrain, ygrain)))
        
        grainPix = list(grainPixels)
        for i in range(len(grainPixels)):
            gx,gy = grainPix[i]
            dij = Xp[gx, gy]
            if not edgeBoundary:
                if isinstance(boundaryId, type(None)): bid = 0
                else: bid = boundaryId
            else: bid = dij
            tf1 = Xp[gx+1, gy] in  [dij, bid]
            tf2 = Xp[gx-1, gy] in  [dij, bid] 
            tf3 = Xp[gx, gy+1] in  [dij, bid] 
            tf4 = Xp[gx, gy-1] in  [dij, bid] 
            tfedge = gx in [1,xsize] or gy in [1,ysize]
            isInternal = np.all([tf1, tf2, tf3, tf4])

            if not isInternal:
                edgePixels.add((gx,gy))
    #Default case, find boundaries for all grains
    if not isinstance(boundaryId, int) and not isinstance(grainId, (int, list,tuple, np.ndarray)):
        for gx in range(1, xsize+1):
            for gy in range(1, ysize+1):
                dij = Xp[gx, gy]
                if not edgeBoundary:
                    if isinstance(boundaryId, type(None)): bid = 0
                    else: bid = boundaryId
                else: bid = dij
                tf1 = Xp[gx+1, gy] in  [dij, bid]
                tf2 = Xp[gx-1, gy] in  [dij, bid] 
                tf3 = Xp[gx, gy+1] in  [dij, bid] 
                tf4 = Xp[gx, gy-1] in  [dij, bid] 
                tfedge = gx in [1,xsize] or gy in [1,ysize]
                isInternal = np.all([tf1, tf2, tf3, tf4])

                if not isInternal:
                    edgePixels.add((gx,gy))


    if len(edgePixels) == 0:
        raise ValueError('fastSweepSDF: No boundaries found in X')
    
    for i,j in list(edgePixels):
        if not boundaryId: i-=1;j-=1
        U[i,j] = 0 #Set boundaries
    #4 Sweeps
    xranges = [(0,U.shape[0],1), (U.shape[0]-1, -1, -1), (U.shape[0]-1, -1,-1), (0, U.shape[0], 1)]
    yranges = [(0,U.shape[1],1), (0, U.shape[1], 1), (U.shape[1]-1, -1, -1), (U.shape[1]-1, -1, -1)]
    notConverged = True
    iterNum = 0
    prevU = U.copy()
    while notConverged:
        for k in range(len(xranges)):
            xs,xe,xstep = xranges[k]
            ys,ye,ystep = yranges[k]
            for i in range(xs,xe,xstep):
                for j in range(ys,ye,ystep):
                    #If a pixel is not an edgePixel:
                    if not boundaryId: pi=i+1;pj=j+1
                    else: pi=i;pj=j
                    if (pi,pj) not in edgePixels:
                        if i <=0:
                            a = U[i+1,j]
                        elif i >= xsize-1:
                            a = U[i-1, j]
                        else:
                            a = min(U[i+1,j], U[i-1,j])

                        if j <= 0:
                            b = U[i, j+1]
                        elif j >= ysize - 1:
                            b = U[i, j-1]
                        else:
                            b = min(U[i,j+1], U[i,j-1])
                        
                        if isinstance(f, np.ndarray):
                            fij = f[i,j]
                        else:
                            fij = f(i,j)


                        U[i,j]=uFunc(a,b,fij,h)
        convCheck = np.abs(U -prevU)
        notConverged = iterNum <= maxIter and np.max(convCheck) > convThresh
        if verbose:
            print('{} of {} Convergence {} > {}'.format(iterNum, maxIter, np.max(convCheck), convThresh))
        prevU = U.copy()
        iterNum +=1
    if edgeBoundary: U = U[0:xsize, 0:ysize] #for some reason U is offset
    if onlyGrain: #Only return values from within grains. used in conjunction with grainId
        grainPixels = np.array(list(grainPixels))
        ret = np.zeros(U.shape)
        ret[grainPixels[:, 0], grainPixels[:, 1]] = U[grainPixels[:,0], grainPixels[:,1]]
        return ret
    else:
        return U

def outerEdgeSDF(X, grainId, grainBuff = 1, **kwargs):
    """where X is a numpy array of grain ids
    grainId is the specific grain to cut out and get the SDF for.
    grainBuff is the size of the buffer zone used to make the SDF since some grains are tiny.
    """
    #general proceedure, 'paintbucket' in from edges of minima boundaries this becomes a 'zero mask'. get SDF boundaries. Apply zero mask from before.
    xsize, ysize = X.shape
    grainXcoords, grainYcoords = np.nonzero(X == grainId)
    xmin = np.min(grainXcoords)
    xmax = np.max(grainXcoords)
    ymin = np.min(grainYcoords)
    ymax = np.max(grainYcoords)
    grainX = np.zeros((xmax-xmin+1+grainBuff+grainBuff, ymax-ymin+1+grainBuff+grainBuff))
    grainX[grainBuff:grainBuff + (xmax+1-xmin), grainBuff:grainBuff + (ymax+1-ymin)] = X[xmin:xmax+1, ymin:ymax+1]
    #print(grainX)
    #if ymin == ymax:
    #    grainX = grainX[:, None] #Adds singleton dimensions. Only issue in y dimension for some reason
    #    print('add y singleton')
    grainX[grainX != grainId] = 2 
    grainX[grainX == grainId] = 1
    gxmax, gymax = grainX.shape
    gxmax = gxmax - 1
    gymax = gymax - 1
    #first set window edges that aren't the same as the grain id to zero.
    grainX[0, np.nonzero(grainX[0, :] == 2)[0]] = 0
    grainX[gxmax, np.nonzero(grainX[gxmax, :] == 2)[0]] = 0
    grainX[np.nonzero(grainX[:, 0] == 2)[0], 0] = 0
    grainX[np.nonzero(grainX[:, gymax] == 2)[0], gymax] = 0
    
    #Paintbucket remove non grain pixels from the outside
    for ci in range(1, np.min([gxmax, gymax])-2):
        for i in range(ci, gxmax-ci+1):
            for j in [ci, gymax-ci]:# Squeezes in from the side.
                #print(i, j, '|', ci, gxmax-ci, '|', ci, gymax - ci, grainX.shape)
                
                uid = grainX[i+1, j]
                did = grainX[i-1, j]
                lrd = grainX[i, j+1]
                rid = grainX[i, j-1]
                quadTF = np.any(np.array([uid, did, lrd, rid])==0)
                if quadTF and grainX[i,j] != 1:
                    grainX[i, j] = 0
        for j in range(ci, gymax-ci+1):
            for i in [ci, gxmax-ci]:# Squeezes in from the side.
                #print(i, j, '|', ci, gxmax-ci, '|', ci, gymax - ci, grainX.shape)
                
                uid = grainX[i+1, j]
                did = grainX[i-1, j]
                lrd = grainX[i, j+1]
                rid = grainX[i, j-1]
                quadTF = np.any(np.array([uid, did, lrd, rid])==0)
                if quadTF and grainX[i,j] != 1:
                    grainX[i, j] = 0
    
    #If there are any remaining non-grain e.g. Enclave grains/holes, fill them in.
    grainX[grainX == 2] = 1 
    grainXSDF = fastSweepSDF(grainX, grainId = 1, **kwargs)
    #grainXSDF = grainXSDF*grainX #Masking
    
    returnX = np.zeros((xsize, ysize))
    returnX[xmin:xmax+1, ymin:ymax+1] = grainXSDF[grainBuff: grainBuff+(xmax+1-xmin), grainBuff: grainBuff+(ymax+1-ymin)]
    
    return returnX, grainXSDF, grainX

def overlapSDF(X, grainId=None, **kwargs):
    """A wrapper around fastSweepSDF which calculates each grain seperately and then pastes them in.
    """
    
    if isinstance(grainId, type(None)):
        uniqueIds = np.unique(X)
    elif isinstance(grainId, int):
        uniqueIds = [grainId]
    else:
        uniqueIds = grainId

    rX = np.zeros(X.shape)

    for i in range(len(uniqueIds)):
        gid = uniqueIds[i]
        
        if np.any(X == gid):
            grainXFull, grainSDF, grainMask = outerEdgeSDF(X, grainId = gid, **kwargs) 
            rX = rX + grainXFull
    
    return rX
            
    



