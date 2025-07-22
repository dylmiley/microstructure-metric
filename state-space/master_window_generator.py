#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 10:28:14 2024

@author: dylmiley
"""


import sys
import os
#sys.path.insert(1, '/Users/dylanmiley/Desktop/microstructure metric/library/')
#sys.path.append('../tests')
#sys.path.append('/Users/dylanmiley/Desktop/microstructure\ metric/library')
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
import csv
import sdf_0 as sdf
import h5py
import matplotlib.image as pltimg
from warnings import warn
import mputil_0 as mpu
import window_sampling_algorithm as wsa


sample_database = []
file_db = []
fpath = './'#../d3samples/'

for (directory,subdirectory,files) in os.walk(fpath):
    sample_database.extend(files)
# for i in range (len(sample_database)):
#     if sample_database[i].endswith('.dream3d'):
#         file_db.append(fpath + str(sample_database[i]))

print(file_db)
sample_database = file_db
##-#-#--#---#-----#--------#-------------#--------#-----#---#--#-#-##
"""
"""
### COMMENTED TO LOOK AT PHASES ###

# def getSlice(container, axis='x', sliceNum='half', h5pyPath=['DataContainers', 'SyntheticVolumeDataContainer', 'CellData', 'FeatureIds'], copy=False, **kwargs):
def getSlice(container, axis='x', sliceNum='half', h5pyPath=['DataContainers', 'SyntheticVolumeDataContainer', 'CellData', 'Phases'], copy=False, **kwargs):

    #def getSlice(container, axis='x', sliceNum=[0,10,20,30,40,50,60], h5pyPath=['DataContainers', 'SyntheticVolumeDataContainer', 'CellData', 'FeatureIds'], copy=False, **kwargs):
#def getSlice(container,axis='x',sliceNum='half',h5pyPath=['DataContainers','ImageDataContainer','CellData','FeatureIds'],copy=False,**kwargs):
    """
    Retrieves the sliceNum slice from fp.
    container can be:
        A file path that ends with .dream3d, 
        or .png. If so then treats each unique grayscale value, RGB or RGBA combination as a unique ID. Make sure your image isn't anti-aliased.
        A 2D or 3D numpy array.
    axis can be:
        'x' or 0
        'y' or 1
        'z' or 2
    sliceNum = 'half'
        Can be an integer within [0, axis length]
        list of integers within [0, axis length] 
        or 'half'
        or 'all'
    h5pyPath = ['DataContainers', 'SyntheticVolumeDataContainer', 'CellData', FeatureIds']
        if container is a .dream3d filepath, then points to where the slice data is. By default, it is the default synthetic generation path.
    copy = True
        if True, will create a copy of container to work on.
    
    returns
        if sliceNum is a list of integers or 'all' then returns a list of 2D numpy arrays.

    """
    if copy:
        container = container.copy() 
    if isinstance(container, str) and container.endswith('.dream3d'):
        f1d3 = h5py.File(container, 'r')
        #transverse through the path, replacing the pointer as you go.
        for path in h5pyPath:
            f1d3 = f1d3[path] 
        sampleBlock = np.array(list(f1d3))
    elif isinstance(container,str) and container.endswith('.png'):
        img = pltimg.imread(container)
        if len(img.shape) == 2: #Grayscale
            l, h= img.shape
            uniqueids = np.unique(img)
            sampleBlock = np.zeros((1, l,h))
            for i in range(1, len(uniqueids)+1): #give each configuration a unique id.
                xvals, yvals = np.nonzero(np.all(img==uniqueids[i-1]))
                sampleBlock[0, xvals,yvals]=i
        elif len(img.shape) == 3: #RGB
            l, h, w = img.shape
            uniqueids = np.unique(img.reshape((l*h, w)), axis=0)
            sampleBlock = np.zeros((1, l,h))
            for i in range(1, len(uniqueids)+1): 
                xvals, yvals = np.nonzero(np.all(img==uniqueids[i-1], axis=2))
                sampleBlock[0, xvals, yvals]=i
        else:
            raise ValueError('getSlice(): Image is not Grayscale, RGB or RGBA.')
    elif isinstance(container, np.ndarray):
        if len(container.shape) == 2:
            l, h = container.shape
            sampleBlock = container.reshape((1, l, h))
        elif len(container.shape) == 3:
            sampleBlock = container
        else:
            raise ValueError('getSlice(): container must be either 2D or 3D')
    else:
        raise ValueError('getSlice(): container must be a .dream3d filepath, a .png filepath, or a numpy array')


    if axis == 0 or axis.strip() == 'x': axis = 0
    elif axis == 1 or axis.strip() == 'y': axis = 1
    elif axis == 2 or axis.strip() == 'z': axis = 2
    else: raise ValueError('getSlice(): Unknown axis {}. axis must be 0, 1, 2, or "x", "y", "z" respectively.'.format(axis))


    #Slice or Slices
    if isinstance(sliceNum, str) and sliceNum=='half': sliceNum=[int(np.floor(sampleBlock.shape[axis]/2))]
    elif isinstance(sliceNum, str) and sliceNum == 'all': sliceNum = np.arange(sampleBlock.shape[axis])
    elif isinstance(sliceNum, int): sliceNum = [sliceNum]
    elif isinstance(sliceNum, (tuple, list, np.ndarray)): pass 
    else: raise ValueError('getSlice(): sliceNum must either be "half", "all", an integer or a list/tuple/numpy array')
    
    targetSliceIds = []
    for i in range(0, len(sliceNum)):
        sl = sliceNum[i]                
        if axis == 0:   targetSliceIds.append(sampleBlock[sl, :, :].squeeze())
        elif axis == 1: targetSliceIds.append(sampleBlock[:, sl, :].squeeze())
        elif axis == 2: targetSliceIds.append(sampleBlock[:, :, sl].squeeze())
    
    import matplotlib.pyplot as plt
    if True:
        plt.figure()
        plt.imshow(sampleBlock[:,:,np.random.randint(0, 512),0])
        plt.show()
        
    return targetSliceIds 

def _callSDF(X, sdfType, **kwargs):
    """
    """
    if sdfType == 'boundary':
        if 'boundaryId' in kwargs: boundaryId = kwargs['boundaryId']
        else: boundaryId = 0
        sdfX = sdf.fastSweepSDF(X, boundaryId=boundaryId)
    elif sdfType == 'overlap':
        sdfX = sdf.overlapSDF(X)
    else:
        sdfX = sdf.fastSweepSDF(X, **kwargs)
    return sdfX 

def section2D(s, bounds, refPoint='center', refBounds='center', **kwargs):
    """
    internal func to return a section of a 2d array. if minBound is true, ensures that it doesn't call outside of s. Assumes s is a 2d numpy array. bounds are [xlo, xhi, ylo, yhi] or [xhi, yhi] assuming xlo=ylo=0
    with relation to the ref point. 
    if refBounds == 'center': [refPoint[0]-(xlo+xhi)/2, refPoint[1]+(xlo+xhi)/2]
    if refBounds == 'bounds': [refPoint[0]-xlo, refPoint[1]+xhi]
    if refBounds == 'absolute': [refPoint[0]+xlo, refPoint[1]+xhi]

    s : 2D numpy array
        Slice to be sectioned.
    bounds : int or list
        if bounds is an integer, [xlo, xhi, ylo, yhi] = [0, bounds, 0, bounds]
        if bounds is a two element list [xhi, yhi], [xlo, xhi, ylo, yhi] = [0, xhi, 0, yhi]
        if bounds is a four element list [xlo, xhi, ylo, yhi]
    refPoint = 'center' : str or list/tuple/ndarray
        'center' - refPoint is the center of the slice
        'origin' - refPoint is (0,0)
        2 element list/tuple/ndarray (x,y)
    refBounds ='center' : str
        let refPoint = [refx, refy]
        'center' - refPoint is at the center of the section.
        'bounds' - section is [refx - xlo, refx + xhi, refy - ylo, refy + yhi]
        'absolute' - section is [refx + xlo, refx + xhi, refy + ylo, refy + yhi]
    """
    refx=0;refy=0
    if refPoint == 'center': refx, refy == np.floor(np.array(s.shape)/2)
    elif refPoint == 'origin': refx, refy = (0,0)
    elif isinstance(refPoint, (tuple, list, np.ndarray)) and len(refPoint) == 2: refx, refy = refPoint
    else: raise ValueError('_section2D(): refPoint is either "center" or "origin" or a 2-element array/list/tuple')
    
    if isinstance(bounds, int):
        xlo=ylo=0
        xhi=yhi=bounds
    elif len(bounds) == 4:
        xlo, xhi, ylo, yhi = bounds
    elif len(bounds) ==2:
        xhi, yhi = bounds
        xlo = ylo = 0
    else: raise ValueError('_section2D(): bounds needs to be either an integer or 2 or 4 long.')
    if refBounds == 'center':
        xavg = int((xlo+xhi)/2)
        yavg = int((ylo+yhi)/2)
        xlo = refx - xavg
        xhi = refx + xavg
        ylo = refy - yavg
        yhi = refy + yavg
    elif refBounds == 'bounds':
        xlo = refx - xlo 
        xhi = refx + xhi
        ylo = refy - ylo
        yhi = refy + yhi
    elif refBounds == 'absolute':
        xlo = refx + xlo 
        xhi = refx + xhi
        ylo = refy + ylo
        yhi = refy + yhi
    else: raise ValueError('_section2D(): refBounds needs to be either "center", "bounds" or "absolute"')
    #Check if bounds call outside of s, push them back if need be.
    n, m = s.shape
    if xhi-xlo > n:
        warn('section2D(): xlo:xhi cut is larger than s. Is this ok?')
        xlo = 0; xhi = n
    elif xhi >= n: xlo = xlo - (xhi - n); xhi = n
    elif xlo < 0: xhi = xhi - xlo; xlo = 0
    if yhi-ylo > n:
        warn('section2D(): ylo:yhi cut is larger than s. Is this ok?')
        ylo = 0; yhi = n
    elif yhi >= n: ylo = ylo - (yhi - n);  yhi = n 
    elif ylo < 0: yhi = yhi - ylo; ylo = 0
    return s[xlo:xhi, ylo:yhi]

def generateWindows(sample1, slices, optionsDict1={}, optionsDict2={},
        windowSize=None, 
        windowNum=1, 
        windowLoc = 'random',
        seed=None,
        sliceSDF = False,
        windowSDF = False,
        sdfType='sweep',
        sdfReverse = False,
        multiprocessing = False,
        cores = 5,
        data_sidelengths = 512,
        **kwargs):
    """
    Generates windows given a whole slew of stuff. optionsDict1 and optionsDict2 are used
    as keyword arguements for getSlice(), sliceTreatment() and windowTreatment(). For their respective samples. 
    E.g. optionsDict1 = {'sliceNum': 3, 'axis': 'x'}
         optionsDict2 = {'sliceNum': 5, 'axis': 'y'}
         Will retrieve slice 3 from axis x from the sample1 and slice5, axis='y' from the second.

        Sample generation is done in multiple steps:
        1. Calls getSlice() to retrive slices from sample1 and sample2
        2. If sliceCrop exists in kwargs or optionsDict, section2D() is called. This is useful if the sample data has bad data at the edges.
        3. If sliceSDF = True, each slice is treated with SDF
        4. Windows are taken from the slices.
        5. If windowSDF=True, each window is treated with SDF.

    sample1, sample2 : str, 2D ndarrays list of 2D ndarrays:
        Filepaths - Filepaths can be to a .dream3d file or a .png file

    kwargs:
        getSlice():
            axis = 'x' : str
                'x' or 0
                'y' or 1
                'z' or 2
                Defines which axis is the normal axis. e.g. 'x' would take a slice on the yz-plane.
            sliceNum = 'half' : str, int, list, tuple, ndarray
                'half'
                'all' - returns a list of all slices.
                integer or list of integers - returns a list of the indexed slices.

            h5pyPath =['DataContainers', 'SyntheticVolumeDataContainer', 'CellData', 'FeatureIds'] : list of str
                Denotes the h5py path to the FeatureIds container.
            copy = True : bool
               Makes a copy of the container before working.

        section2D(): 
            s : 2D numpy array
                Slice to be sectioned.
            sliceCrop : int or list. Note sliceCrop = bounds is section2D
                if sliceCrop is an integer, [xlo, xhi, ylo, yhi] = [0, sliceCrop, 0, sliceCrop]
                if sliceCrop is a two element list [xhi, yhi], [xlo, xhi, ylo, yhi] = [0, xhi, 0, yhi]
                if sliceCrop is a four element list [xlo, xhi, ylo, yhi]
            refPoint = 'center' : str or list/tuple/ndarray
                'center' - refPoint is the center of the slice
                'origin' - refPoint is (0,0)
                2 element list/tuple/ndarray (x,y)
            refBounds ='center' : str
                let refPoint = [refx, refy]
                'center' - refPoint is at the center of the section.
                'bounds' - section is [refx - xlo, refx + xhi, refy - ylo, refy + yhi]
                'absolute' - section is [refx + xlo, refx + xhi, refy + ylo, refy + yhi]

        sliceSDF=False : bool
            If sliceSDF is True, slices are replaced with their SDF histogram.
        sdfType='sweep' : str
            'sweep' - Assumes each unique value in the slice represents a seperate grain. Puts grain boundaries on the edges.
            'overlap' - Calculates the SDF for each grain seperately and superimposes them. This means that a grain within another grain creates two 'mounds' on top of each other. 
            'boundaries' - Assumes a certain grain ID are the boundaries. By default 0 is the grain boundary.
        sdfReverse=True : bool
            If True, will invert the SDF such that the mass is concentrated on the boundaries.

        windowSize = None : int
            Square window size in pixels. By default window is 20x20
        windowNum = 1 : int
            Number of windows returned. Randomly taken from the list of slices.
        windowLoc = : str 
            'random' - Randomly selects a location for each window.
            'center' - Takes the windows from the center of the slice.

        windowSDF = False : bool
            If windowsSDF is True, windows are replaced with their SDF histogram.

        multiprocessing = False : bool
            If multiprocessing = True, treating slices and windows are done using the available subprocesses.
        cores = 1 : int
            If the multiprocessing module is not running, will spawn cores subproceeses.

    returns:
        windowList1, windowList2 : list of ndarrays of each windows.


    """

    defaultKwargs = {
        'windowSize':windowSize,
        'windowNum' :windowNum,
        'windowLoc' :windowLoc,
        'seed'      :seed,
        'sliceSDF'  :sliceSDF,
        'windowSDF' :windowSDF,
        'sdfType'   :sdfType,
        }
    tempDict = {}
    tempDict.update(defaultKwargs)
    tempDict.update(kwargs)
    tempDict.update(optionsDict1)
    optionsDict1 = tempDict

    rng = np.random.RandomState(seed)
    #Get the list of slices from each sample.
    axes = ['x','y','z']
    
    ############################
    #### CHANGE MADE FEB 20 ####
    ############################
    # axes = ['z','z','z']
    
    n_sl = len(slices)
    sliceList1 = np.zeros((n_sl*3,data_sidelengths*1,data_sidelengths*1))
    for i in range(3):
        sliceList1[i*n_sl:(i+1)*n_sl] = getSlice(sample1, axis = axes[i], sliceNum = slices, **optionsDict1)
    
    #crop each microstructure if necessary. Especially for bad edge data.
    if 'sliceCrop' in optionsDict1.keys():
        for i in range(len(sliceList1)):
            sliceList1[i] = section2D(sliceList1[i], optionsDict1['sliceCrop'], **optionsDict1)
        
    ##TODO: Treat each slice e.g. do SDF
    if sliceSDF:
        if multiprocessing and not mpu.mpOn: mpu.startMP(cores=cores)

        for i in range(len(sliceList1)):
            if multiprocessing: mpu.pushJobMP(((i,1), _callSDF, [sliceList1[i], sdfType], kwargs), verbose=False)
            else: 
                sdfX=_callSDF(sliceList1[i], sdfType, **kwargs)
                if sdfReverse: sdfX = np.abs(np.max(sdfX) - sdfX)
                sliceList1[i] = sdfX
        if multiprocessing:
            mpu.waitResMP(len(sliceList1))
            resList = mpu.popResQueue()
            for tup, sdfX in resList:
                i, sample = tup
                if sdfReverse: sdfX = np.abs(np.max(sdfX) - sdfX)
                if sample == 1: sliceList1[i] = sdfX
    
    ### COMMENTED TO LOOK AT PHASES ###
    # sliceList1 = convert_sdf_binary(sliceList1)
    
    #Randomly generate points and cut out windows
    if isinstance(windowSize, type(None)): raise ValueError('generateWindows(): windowSize needs to be an integer in kwargs, optionsDict1, or optionsDict2')
    if 'windowNum' in optionsDict1: windowNum = optionsDict1['windowNum']
    if isinstance(windowNum, type(None)): raise ValueError('generateWindows(): windowNum needs to be an integer in kwargs, optionsDict1, or optionsDict2')
    if 'windowLoc' in optionsDict1: windowLoc = optionsDict1['windowLoc']
    

    windowList1 = []
    indices = wsa.find_valid_wnds(data_sidelengths, optionsDict1['windowSize'], windowNum)[0]
    print(indices)
    for i in range(windowNum):
        if 'windowSize' in optionsDict1: windowSize = optionsDict1['windowSize']
        sl=sliceList1[rng.randint(0,len(sliceList1))]
        n, m = sl.shape
        # if windowLoc == 'center': refPoint = 'center'
        # else: refPoint = [
        #         rng.randint(0,n-windowSize+1), 
        #         rng.randint(0,m-windowSize+1)
        #         ] 
        
        refPoint = [indices[0, i], indices[1, i]]
        
        windowList1.append(section2D(sl, [windowSize, windowSize], refPoint = refPoint, refBounds= 'absolute'))

    if windowSDF:
        if multiprocessing and not mpu.mpOn: mpu.startMP(cores=cores)
        for i in range(len(windowList1)):
            if multiprocessing: mpu.pushJobMP(((i,1), _callSDF, [windowList1[i], sdfType], kwargs), verbose=False)
            else: 
                sdfX = _callSDF(windowList1[i], sdfType, **kwargs)
                if sdfReverse: sdfX = np.abs(np.max(sdfX) - sdfX)
                windowList1[i] = sdfX
        
        if multiprocessing:
            mpu.waitResMP(len(windowList1))
            resList = mpu.popResQueue()
            for tup, sdfX in resList:
                i, sample = tup
                if sdfReverse: sdfX = np.abs(np.max(sdfX) - sdfX)
                if sample == 1: windowList1[i] = sdfX

    return windowList1

def convert_sdf_binary(wds):
    for i in range(len(wds)):
        w = wds[i]
        w[np.where(w > 0)] = -1
        w += 1
        wds[i] = w
    return wds

def generate_distribution(sample, window_nm, window_sz, slices):
    wd = generateWindows(sample, slices = slices, windowSize = window_sz, windowNum = window_nm)
    return wd

if __name__ == '__main__' :
    windowNum = 50
    windowSize = 50
    slices = np.arange(0,100,20)
    samples = [fpath + 'EI.dream3d', fpath + 'EII.dream3d', fpath + 'BI.dream3d', 
               fpath + 'BII.dream3d', fpath + 'RI.dream3d']
    RI = generate_distribution(samples[4], windowNum, windowSize, slices)

