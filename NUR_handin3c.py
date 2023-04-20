import numpy as np
import matplotlib.pyplot as plt
import copy 
import scipy.special as spec

def readfile(filename):
    f = open(filename, 'r')
    data = f.readlines()[3:] #Skip first 3 lines 
    nhalo = int(data[0]) #number of halos
    radius = []
    
    for line in data[1:]:
        if line[:-1]!='#':
            radius.append(float(line.split()[0]))
    
    radius = np.array(radius, dtype=float)    
    f.close()
    return radius, nhalo #Return the virial radius for all the satellites in the file, and the number of halos

def romberg(left, right, m, func, params):
    """
    Numerical integration using the Romberg method.

    Parameters
    ----------
    left : int or float
        The left bound of the integration.
    right : int or float
        The right bound of the integration.
    m : int
        The order or number of points to be created.
    func : function
        The function to be integrated.
    params : list
        Any additional parameters the function requires.

    Returns
    -------
    float
        The result of the numerical integration.

    """
    h = right - left
    r = np.zeros(m)
    r[0] = 0.5 * h * (func(left,*params)+func(right,*params))
    N_p = 1
    #Create m initial approximations using trapezoid approximation
    for i in range(1,m):
        r[i] = 0
        delta = h
        h = 0.5 * h
        x = left + h
        for j in range(N_p):
            r[i] = r[i] + func(x,*params)
            x = x + delta
        r[i] = 0.5 * (r[i-1] + delta*r[i])
        N_p *= 2

    N_p = 1
    #Combine the approximations in an analogue to Neville's algorithm
    for i in range(1,m):
        N_p *= 4
        recip = 1/(N_p-1) #Calculates the denominator for the next loop for a little extra efficiency
        for j in range(m-i):
            r[j] = (N_p*r[j+1] - r[j])*recip
    return r[0]

def selection_sort(data,sortcol=None):
    for i in range(len(data[:,sortcol])-1):
        imin = i
        for j in range(i,len(data)):
            if data[j,sortcol] < data[imin,sortcol]:
                imin=j
        if imin != i:
            data[[imin, i]] = data[[i,imin]]
    return data

def simplex(N, target, start, func, maxnr, chiparams):
    counter = 0
    A_Nsat = start[:2]
    start = start[2:]
    points = np.zeros((N+1,N))
    points[0,:] = start
    #Initialise the simplex with N+1 points
    for i in range(1,N+1):
        newpoint = copy.copy(start)
        newpoint[i-1] += 0.1
        points[i,:] = newpoint
    funcvals = np.zeros((N+1,2))
    funcvals[:,0] = np.arange(N+1)
    while counter < maxnr: #makes sure there will be a cutoff
        #Find the point of the simplex with the lowest value
        funcvals[:,1], params = func(np.append(np.tile(A_Nsat,(N+1,1)),points,axis=1),*chiparams)
        #Then sort the points from low to high
        index = np.int32(selection_sort(funcvals,1)[:,0])
        points = points[index]
        values = funcvals[:,1][index]
        #Find the centroid away from the lowest value
        centroid = 1/N * np.sum(points[:N],axis=1)
        #Determine if the simplex has found a minimum
        fracrange = np.abs(values[N]-values[0])/(0.5*np.abs(values[N]+values[0]))
        if fracrange < target:
            print('Target met')
            print(values)
            return np.append(params[:2],points[0])
        #Try mirroring the simplex using the centroid
        xtry = 2*centroid - points[N]
        valtry, params = func(np.append(A_Nsat,xtry),*chiparams)
        #If the new value is better than the worst but not as good as the best
        #Accept it
        if values[0] < valtry and valtry < values[N]:
            points[N] = xtry
        #If the new value is the new best, try expanding it, otherwise accept it
        elif valtry < values[0]:
            xexp = 2*xtry - centroid
            valexp, params = func(np.append(A_Nsat,xexp),*chiparams)
            if valexp < valtry:
                points[N] = xexp
            else:
                points[N] = xtry
        #If the mirrored value is worse, shift the simplex in the direction of the
        #best value
        else:
            xtry = 0.5*(centroid+points[N])
            valtry, params = func(np.append(A_Nsat,xtry),*chiparams)
            #If the shifted value is worse, contract the simplex towards its
            #best value
            if valtry < values[N]:
                points[N] = xtry
            else:
                for i in range(1,N+1):
                    points[i] = 0.5*(points[0]+points[i])
        counter+=1
        
    print('Maximum number of iterations reached')
    return np.append(params[:2],points[0])

def N(x,A,Nsat,a,b,c):
    #The function is defined in such a way as to avoid division by 0
    return 4*np.pi*A*Nsat*((x**(a-1))/(b**(a-3)))*np.exp(-(x/b)**c)

def lnL(params, y, func, binedges):
    #Similar to chisquared above, but for the poissonian log likelyhood
    mean = np.zeros(len(y))
   
    if len(params.shape) > 1:
        chi2 = np.zeros(params.shape[0])
        for i in range((params.shape[0])):
            params[i,0] = 1
            params[i,0] = params[i,1]/romberg(0,5,10,func, params[i,:])
            for j in range(len(mean)):
                mean[j] = romberg(binedges[j],binedges[j+1],10,func,params[i,:])
            chi2[i] =-1 * np.sum(y*np.log(mean)-mean-np.log(factorial(y)))
        return chi2, params[0]
    else:
        params[0] = 1
        params[0] = params[1]/romberg(0,5,10,func, params)
        for j in range(len(mean)):
            mean[j] = romberg(binedges[j],binedges[j+1],10,func,params)
        return -1 * np.sum(y*np.log(mean)-mean-np.log(factorial(y))),params

def minpoisson(filename,init,binnum):
    params = copy.copy(init)
    radius, nhalo = readfile(filename)
    N_sat = len(radius)/nhalo
    params[1] = N_sat
    binedges=np.logspace(-4,np.log10(5),num=binnum,endpoint=False)
    
    counts,bins=np.histogram(radius,binedges)
    counts = counts * 1/nhalo
    lnLparams = [counts,N,bins]
    bestparams = simplex(3,10**-15,params,lnL ,100,lnLparams)
    print(bestparams)
    xx = np.linspace(0,5,1000)
    plt.plot(xx,N(xx,*bestparams))
    plt.stairs(counts,bins,fill=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('{} with Poisson fit'.format(filename))
    plt.xlabel('Radial measure x')
    plt.ylabel('Number of satellites')
    plt.show()
    midbins = bins[:-1]+0.5*np.diff(bins)
    G = Gtest(counts, N(midbins,*bestparams))
    print('The G-test value for the Poisson fit of {} is: {}'.format(filename,G))
    Q = Qval(binnum-1,G)
    print('The Q-value for the Poisson fit of {} is: {}'.format(filename,Q))
    
def Gtest(observed, expected):
    #Determines the G-value of a fit, ignoring empty bins
    observed = observed[np.nonzero(observed)]
    expected = expected[np.nonzero(observed)]
    ratio = observed/expected
    return 2* np.sum(observed*np.log(ratio))

def Qval(k,x):
    #Determines the Q-value of a fit using scipy library functions
    halfk = 0.5*k
    halfx = 0.5*x
    incgamma = spec.gammainc(halfk,halfx)
    gamma = spec.gamma(halfk)
    P = incgamma/gamma
    return 1-P

minpoisson('satgals_m11.txt', parameters)
minpoisson('satgals_m12.txt', parameters)
minpoisson('satgals_m13.txt', parameters)
minpoisson('satgals_m14.txt', parameters)
minpoisson('satgals_m15.txt', parameters)
