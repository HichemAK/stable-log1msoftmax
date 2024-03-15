import torch
from mpmath import *
from log1msoftmax import log1m_softmax
import numpy as np

mp.prec = 2000

def log1msoftmax_pft(x : list):
    m = max(x)
    x = [i-m for i in x]
    s = sum(exp(i) for i in x)
    res = [log(1-exp(i)/s) for i in x]
    return res

def softmax_pft(x : list):
    m = max(x)
    x = [i-m for i in x]
    s = sum(exp(i) for i in x)
    res = [exp(i)/s for i in x]
    return res

def grad_log1msoftmax_pft(x : list):
    m = max(x)
    x = [i-m for i in x]
    res = [[0]*len(x) for _ in range(len(x))]
    sm_x = softmax_pft(x)
    nprint(sm_x)
    for i in range(len(x)):
        for j in range(len(x)):
            if i == j:
                # res[i][j] = -sm_x[i]
                res[i][j] = 0
            else:
                # t = softmax_pft([y for k,y in enumerate(x) if k != i])[j if j <i else j - 1]
                # res[i][j] = sm_x[i]*t
                res[i][j] = 1
    return res

def log1m_softmax_KFrank (X):                           # for one-dimensional tensor X
    xm, im = X.max (0)                                  # largest value in X is the potential problem
    X_bar = X - xm                                      # uniform shift doesn't affect softmax (except numerically)
    lse = X_bar.logsumexp (0)                           # denominator for final result
    sumexp = X_bar.exp().sum() - X_bar.exp()            # sumexp[im] potentially zero
    sumexp[im] = 1.0                                    # protect against log (0)
    log1msm = sumexp.log()                              # good for all but i = im
    X_bar = X_bar.clone()                               # to support backward pass
    X_bar[im] = -float ('inf')                          # "zero out" xm in log space
    log1msm[im] = X_bar.logsumexp (0)                   # replace bad xm term
    log1msm -= lse                                      # final result
    return  log1msm

def max_abs_dist(l1,l2):
    assert len(l1) == len(l2)
    max_error = 0
    for x,y in zip(l1,l2):
        d = abs(x - y)
        if d > max_error:
            max_error = d   
    return max_error

if __name__ == '__main__':
    g = grad_log1msoftmax_pft([-1000,-1000,0])
    nprint(g)