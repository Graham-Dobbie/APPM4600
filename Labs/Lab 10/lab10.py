import numpy as np
import matplotlib.pyplot as plt

def comp_trapz(a, b, f, N):
    intervals = np.linspace(a,b,N+1)
    f_evals = f(intervals)
    w = np.ones(N+1)
    w[0] = 1/2
    w[-1] = 1/2

    int_f = np.dot(f_evals, w)
    return (b-a)/(N) * int_f

def comp_simpsonz(a, b, f, N):
    points = 2*int(N/2)+1
    intervals = np.linspace(a,b,points)
    f_evals = f(intervals)
    w = np.ones(points)
    for i in range(0,points):
        if (i%2):
            w[i] = 4
        else:
            w[i] = 2
    w[0] = 1
    w[-1] = 1

    
    # print(f_evals)
    int_f = np.dot(f_evals, w)
    return (b-a)/(3*(points-1)) * int_f