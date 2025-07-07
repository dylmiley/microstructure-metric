import multiprocessing as mp
import time
import queue
import sys
#todo: add shared memory for storage intensive jobs

mpOn=False
checkInterval = 15
def setCheckInterval(interval):
    """Sets the default interval in seconds waitResMP() and waitJobMP() will print how many jobs are left in the queue. Default=15 seconds"""
    global checkInterval
    checkInterval = interval

def __runFuncMP(jobQueue, resQueue, timeout=600, mpverbose=False): 
    '''generalized processing function. For use with multiple cases so we're not killing and remaking new process all the time'''
    while True:
            try:
                job = jobQueue.get(timeout=timeout) #Each job should be (jobid, func, args[], kwargs{})
            except queue.Empty:
                break
            jobid = None
            args = []
            kwargs = {}
            if isinstance(job, str) and job == 'END':
                #print('ending', mp.current_process().name)
                break
            elif isinstance(job, (tuple, list)):
                jobid, func, args, kwargs, = job
            else:
                func = job
            if 'mpverbose' in kwargs:
                mpverbose = kwargs['mpverbose']

            if mpverbose:
                print('{}: Processing {}'.format(mp.current_process().name, jobid))
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                print('{}: Job {} | {}'.format(mp.current_process().name, jobid, repr(e)), file=sys.stderr)
                res = None
            resQueue.put((jobid, res))
    return True


def startMP(cores=1, timeout=600, mpverbose=False):
    global jobQueue, resQueue, procList, mpOn
    jobQueue = mp.Queue() #job Queue
    resQueue = mp.Queue() #results Queue
    procList = []
    addProcMP(cores, procList, timeout=timeout, mpverbose=mpverbose)
    mpOn = True

def getProcessListMP():
    return procList

def pushJobMP(job, repeat=1, verbose=True):
    """Jobs are a tuple of (jobid, func, args[], kwargs{})"""
    #Put some checking here to make sure jobs going in are good.
    if repeat == 'all': repeat = len(procList)
    for i in range(repeat):
        if verbose:
            print('Adding', job, 'to jobQueue')
        jobQueue.put(job)
    
def addProcMP(cores, procList, timeout=600, mpverbose=False):
    '''extends the procList incase you wanted MORE'''
    newProcs = [mp.Process(target=__runFuncMP, args=(jobQueue, resQueue), kwargs={'timeout':timeout, 'mpverbose':mpverbose}, daemon=False) for i in range(cores)]
    for p in newProcs:
        print('Appending', p.name)
        p.start()
    procList.extend(newProcs)

def waitJobMP(timeout=None,checkInterval=15):
    """Waits until the joblist is empty"""
    stime = time.time()
    ctime = stime 
    try:
        while jobQueue.qsize() > 0:
            if not isinstance(timeout,type(None)):
                if time.time() - stime > timeout:
                    break
            if time.time() - ctime > checkInterval:
                print('Jobs in Queue: {}, {}'.format(jobQueue.qsize(), time.asctime()))
                ctime = time.time()
            time.sleep(0.1)
    except KeyboardInterrupt:
        endMP()
        quit()

def waitResMP(doneThresh, timeout=None,checkInterval=15):
    """Waits until the reslist has a certain number of done jobs."""
    stime = time.time()
    ctime = stime 
    try:
        while resQueue.qsize() < doneThresh:
            if not isinstance(timeout,type(None)):
                if time.time() - stime > timeout:
                    break
            if time.time() - ctime > checkInterval:
                print('Jobs in Queue: {}, {}'.format(jobQueue.qsize(), time.asctime()))
                ctime = time.time()
            time.sleep(0.1)
    except KeyboardInterrupt:
        endMP()
        quit()


def popResQueue():
    resultList = []
    while True:
        try:
            jobid, result = resQueue.get(timeout=0.1)
            resultList.append((jobid, result))
            #print(jobid, result)
        except queue.Empty:
            break
    return resultList

def endMP():
    global mpOn
    if mpOn:
        for p in procList:
            print('Terminating', p.name)
            p.terminate()
        mpOn = False
        

