import numpy as np
from scipy.integrate import quad


def discrete_gautschi(z, w, n_iter):
    p = np.ones(z.shape)
    p /= np.sqrt(np.dot(p,p))
    p_prev = np.zeros(z.shape)
    wz = z*w
    a = np.empty(n_iter)
    b = np.empty(n_iter)
        
    for j in xrange(n_iter):
        p_norm = np.dot(w*p,p)
        a[j] = np.dot(wz*p,p)/p_norm
        b[j] = 0.0 if j==0 else p_norm/np.dot(w*p_prev,p_prev)
        p_new = (z-a[j])*p - b[j]*p_prev 
        (p_prev, p_prev_norm) = (p, p_norm)
        p = p_new
        
    return (a, b[1:])        


def get_points_and_weights(func=lambda x : np.ones(x.shape), 
    left=-1.0, right=1.0, num_points=5, n=4096):
    
    dx = (float(right)-left)/n
    z = np.hstack(np.linspace(left+0.5*dx, right-0.5*dx, n))
    w = dx*func(z)  
   
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
