import numpy as np
import matplotlib.pyplot as plt

def N(x,A,Nsat,a,b,c):
    #The function is defined in such a way as to avoid division by 0
    return 4*np.pi*A*Nsat*((x**(a-1))/(b**(a-3)))*np.exp(-(x/b)**c)

def goldenSection(func, a, b, c, target,params = None, w=0.38197, minimum=True):
    if minimum !=True:
        tempfunc = func
        newfunc = lambda *args:-1*tempfunc(*args)
        func  = newfunc
    
    while np.abs(c-a) > target:
        if np.abs(c-b) < np.abs(b-a):
            d = b + (a-b)*w
            left = True
        else:
            d = b + (c-b)*w
            left =  False
        if func(d,*params) < func(b,*params):
            if left:
                c = b
                b = d
            else:
                a = b
                b = d
        else:
            if left:
                a = d
            else:
                c = d
        left = not left
    if func(d,*params) < func(b,*params):
        return d
    else:
        return b

parameters = [256/(np.pi**(3/2)),100,2.4,0.25,1.6]
xmax = (goldenSection(N, 0,2.5,5,0.1,params=parameters,minimum=False))
xx = np.linspace(0,5,1000)
print('The maximum of N(x) is at x={}, and N(x)={}.'.format(xmax,N(xmax,*parameters)))
plt.plot(xx, N(xx,*parameters))
plt.plot(xmax,N(xmax,*parameters),'ob')
plt.title('Maximizing N(x)')
plt.xlabel('Radius measure x')
plt.ylabel('Number of satellite galaxies N(x)')
plt.savefig('./plots/maxN.pdf')
plt.close()
