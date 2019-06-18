# encoding: utf-8
'''
Contains various potentials to be (possibly) used for MD.

All accept a "sigma" parameter which is the minimum of the potential,
and a "cutoff" parameter where the potential is made 0 by shifting.

Created on 1 May 2018

@author: Eugen Rožić
'''

from math import pi, exp, sqrt, cos

def vx(r, sigma):
    '''
    Hard volume exclusion potential: infinity for r<sigma, else 0
    '''
    if r < sigma:
        return float("inf")
    else:
        return 0.0

def lj_n_m(n, m, r, sigma, cutoff, eps, shift = True):
    '''
    A generalised Lennard-Jones potential with repulsive exponent "n" and
    attractive exponent "m", and a minimum at (sigma, -eps)
    
    If cutoff <= sigma the function returns infinity for r < cutoff (to
    be used as volume exclusion).
    '''
    if r >= cutoff:
        return 0.0
    if cutoff <= sigma:
        return float("inf")
    if shift: 
        co_val = m*(sigma/cutoff)**n - n*(sigma/cutoff)**m
        return eps*(m*(sigma/r)**n - n*(sigma/r)**m - co_val)/(n-m)
    else:
        return eps*(m*(sigma/r)**n - n*(sigma/r)**m)/(n-m)

def cos_sq(r, sigma, cutoff, eps, wca = False):
    '''
    A potential with a cosine-squared attractive part connecting points (sigma, -eps) and (cutoff, 0) smoothly.
    
    If wca == True then returns infinity for r < sigma (volume exclusion). 
    '''
    if cutoff < sigma:
        raise Exception("cos_sq has to have cutoff >= sigma!")
    if r >= cutoff:
        return 0.0
    elif r < sigma:
        if wca:
            return float("inf")
        else:
            return -eps
    else:
        return -eps*(cos(pi*(r-sigma)/(2*(cutoff-sigma))))**2

def gauss(std_dev, r, sigma, cutoff, eps, shift = True):
    '''
    A Gaussian potential with standard deviation "std_dev" centered at (sigma, -eps)
    '''
    if cutoff <= sigma:
        raise Exception("gauss has to have cutoff > sigma!")
    if r >= cutoff:
        return 0.0
    if shift:
        co_val = exp(-((cutoff-sigma)/(sqrt(2)*std_dev))**2)
        return -eps*(exp(-((r-sigma)/(sqrt(2)*std_dev))**2) - co_val)
    else:
        return -eps*exp(-((r-sigma)/(sqrt(2)*std_dev))**2)

def morse(a, r, sigma, cutoff, eps, shift = True):
    '''
    A Morse potential with a minimum at (sigma, -eps) with "width" given by "a".
    
    If cutoff <= sigma the function returns infinity for r < cutoff (to
    be used as volume exclusion).
    '''
    if r >= cutoff:
        return 0.0
    if cutoff <= sigma:
        return float("inf")
    if shift:
        co_val = exp(-2*a*(cutoff-sigma)) - 2*exp(-a*(cutoff-sigma))
        return eps*(exp(-2*a*(r-sigma)) - 2*exp(-a*(r-sigma)) - co_val)
    else:
        return eps*(exp(-2*a*(r-sigma)) - 2*exp(-a*(r-sigma)))
