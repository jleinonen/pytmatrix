"""
Copyright (C) 2009-2015 Jussi Leinonen, Finnish Meteorological Institute, 
California Institute of Technology

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np
from scipy.integrate import quad
import sys


def discrete_gautschi(z, w, n_iter):
    p = np.ones(z.shape)
    p /= np.sqrt(np.dot(p,p))
    p_prev = np.zeros(z.shape)
    wz = z*w
    a = np.empty(n_iter)
    b = np.empty(n_iter)

    if sys.version_info[0] > 2:
        prange = range
    else:
        prange = xrange
        
    for j in prange(n_iter):
        p_norm = np.dot(w*p,p)
        a[j] = np.dot(wz*p,p)/p_norm
        b[j] = 0.0 if j==0 else p_norm/np.dot(w*p_prev,p_prev)
        p_new = (z-a[j])*p - b[j]*p_prev 
        (p_prev, p_prev_norm) = (p, p_norm)
        p = p_new
        
    return (a, b[1:])        


def get_points_and_weights(w_func=lambda x : np.ones(x.shape), 
    left=-1.0, right=1.0, num_points=5, n=4096):
    """Quadratude points and weights for a weighting function.

    Points and weights for approximating the integral 
        I = \int_left^right f(x) w(x) dx
    given the weighting function w(x) using the approximation
        I ~ w_i f(x_i)

    Args:
        w_func: The weighting function w(x). Must be a function that takes
            one argument and is valid over the open interval (left, right).
        left: The left boundary of the interval
        right: The left boundary of the interval
        num_points: number of integration points to return
        n: the number of points to evaluate w_func at.

    Returns:
        A tuple (points, weights) where points is a sorted array of the
        points x_i and weights gives the corresponding weights w_i.
    """
    
    dx = (float(right)-left)/n
    z = np.hstack(np.linspace(left+0.5*dx, right-0.5*dx, n))
    w = dx*w_func(z)  
   
    (a, b) = discrete_gautschi(z, w, num_points)
    alpha = a
    beta = np.sqrt(b)
    
    J = np.diag(alpha)
    J += np.diag(beta, k=-1)
    J += np.diag(beta, k=1)    
    
    (points,v) = np.linalg.eigh(J)
    ind = points.argsort()
    points = points[ind]
    weights = v[0,:]**2 * w.sum()
    weights = weights[ind]
    
    return (points, weights)
