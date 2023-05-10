"""
Copyright (C) 2009-2023 Jussi Leinonen, Finnish Meteorological Institute, 
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
from scipy.integrate import quad, dblquad


def gaussian_pdf(std=10.0, mean=0.0):
    """Gaussian PDF for orientation averaging.

    Args:
        std: The standard deviation in degrees of the Gaussian PDF
        mean: The mean in degrees of the Gaussian PDF.  This should be a number
          in the interval [0, 180)

    Returns:
        pdf(x), a function that returns the value of the spherical Jacobian- 
        normalized Gaussian PDF with the given STD at x (degrees). It is 
        normalized for the interval [0, 180].
    """
    norm_const = 1.0
    def pdf(x):
        return norm_const*np.exp(-0.5 * ((x-mean)/std)**2) * \
            np.sin(np.pi/180.0 * x)
    norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev 
    return pdf


def uniform_pdf():
    """Uniform PDF for orientation averaging.

    Returns:
        pdf(x), a function that returns the value of the spherical Jacobian-
        normalized uniform PDF. It is normalized for the interval [0, 180].
    """
    norm_const = 1.0
    def pdf(x):
        return norm_const * np.sin(np.pi/180.0 * x)
    norm_dev = quad(pdf, 0.0, 180.0)[0]
    # ensure that the integral over the distribution equals 1
    norm_const /= norm_dev 
    return pdf


def orient_single(tm):
    """Compute the T-matrix using a single orientation scatterer.

    Args:
        tm: Scatterer (or descendant) instance

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    return tm.get_SZ_single()


def orient_averaged_adaptive(tm):
    """Compute the T-matrix using variable orientation scatterers.
    
    This method uses a very slow adaptive routine and should mainly be used
    for reference purposes. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        tm: Scatterer (or descendant) instance

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2,2), dtype=complex)
    Z = np.zeros((4,4))

    def Sfunc(beta, alpha, i, j, real):
        (S_ang, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
        s = S_ang[i,j].real if real else S_ang[i,j].imag            
        return s * tm.or_pdf(beta)

    ind = range(2)
    for i in ind:
        for j in ind:
            S.real[i,j] = dblquad(Sfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j,True))[0]/360.0        
            S.imag[i,j] = dblquad(Sfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j,False))[0]/360.0

    def Zfunc(beta, alpha, i, j):
        (S_and, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
        return Z_ang[i,j] * tm.or_pdf(beta)

    ind = range(4)
    for i in ind:
        for j in ind:
            Z[i,j] = dblquad(Zfunc, 0.0, 360.0, 
                lambda x: 0.0, lambda x: 180.0, (i,j))[0]/360.0

    return (S, Z)


def orient_averaged_fixed(tm):
    """Compute the T-matrix using variable orientation scatterers.
    
    This method uses a fast Gaussian quadrature and is suitable
    for most use. Uses the set particle orientation PDF, ignoring
    the alpha and beta attributes.

    Args:
        tm: Scatterer (or descendant) instance.

    Returns:
        The amplitude (S) and phase (Z) matrices.
    """
    S = np.zeros((2,2), dtype=complex)
    Z = np.zeros((4,4))
    ap = np.linspace(0, 360, tm.n_alpha+1)[:-1]
    aw = 1.0/tm.n_alpha

    for alpha in ap:
        for (beta, w) in zip(tm.beta_p, tm.beta_w):
            (S_ang, Z_ang) = tm.get_SZ_single(alpha=alpha, beta=beta)
            S += w * S_ang
            Z += w * Z_ang

    sw = tm.beta_w.sum()
    #normalize to get a proper average
    S *= aw/sw
    Z *= aw/sw

    return (S, Z)
