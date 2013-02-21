"""
Copyright (C) 2009-2013 Jussi Leinonen

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

from fortran_tm import pytmatrix
from gaussqr import pygaussqr
import scatter
import orientation


class TMatrix(object):
    """T-Matrix scattering from nonspherical particles.

    Class for simulating scattering from nonspherical particles with the 
    T-matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

    Usage instructions:

    First, the class should be be initialized. Any attributes (see below)
    can be passed as keyword arguments to the constructor. For example:
    tm = tmatrix.TMatrix(lam=2.0, m=complex(0,2))

    The properties of the scattering and the radiation should then be set 
    as attributes of this object. The naming of the attributes is identical to 
    the Fortran code.

    The functions for computing the various scattering properties can then be 
    called. The TMatrix object will automatically recompute the T-matrix 
    and/or the amplitude and phase matrices when needed.

    Attributes: 
        axi: Equivalent radius.
        rat: If rat==1 (default), axi is the equivalent volume radius.
        lam: The wavelength of incident light (same units as axi).
        m: The complex refractive index.
        eps: The horizontal-to-rotational axis ratio.
        np: Particle shape. -1: spheroids; -2: cylinders; 
            n > 0: nth degree Chebyshev particles (not fully supported).
        alpha, beta: The Euler angles of the particle orientation (degrees).
        thet0, thet: The zenith angles of incident and scattered radiation 
            (degrees).
        phi0, phi: The azimuth angles of incident and scattered radiation 
            (degrees).
        Kw_sqr: The squared reference water dielectric factor for computing 
            radar reflectivity.
        scatter: The function to use to compute the scattering properties.
            Should be one of the TMatrix class methods (scatter_single, 
            scatter_averaged_adaptive, scatter_averaged_fixed).
        or_pdf: Particle orientation PDF for orientational averaging.
        n_alpha: Number of integration points in the alpha Euler angle.
        n_beta: Number of integration points in the beta Euler angle.
    """

    def __init__(self, **kwargs):
        self.axi = 1.0
        self.rat = 1.0
        self.lam = 1.0
        self.m = complex(2,0)
        self.eps = 1.000001
        self.np = -1
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = 0.0
        self.beta = 0.0
        self.thet0 = 90.0
        self.thet = 90.0
        self.phi0 = 0.0
        self.phi = 180.0
        self.Kw_sqr = 0.93
        self.scatter = scatter.scatter_single
        self.or_pdf = orientation.gaussian_pdf()
        self.n_alpha = 5
        self.n_beta = 10

        self._tm_signature = ()
        self._orient_signature = ()
        self._scatter_signature = ()        

        attrs = ("axi", "rat", "lam", "m", "eps", "np", "ddelt", "ndgs", 
            "alpha", "beta", "thet0", "thet", "phi0", "phi", "Kw_sqr",
            "scatter", "or_pdf", "n_alpha", "n_beta")
        for k in kwargs:
            if k in attrs:
                self.__dict__[k] = kwargs[k]


    def set_geometry(self, geom):
        """A convenience function to set the geometry variables.

        Args:
            geom: A tuple containing (thet0, thet, phi0, phi, alpha, beta).
            See the TMatrix class documentation for a description of these
            angles.
        """
        (self.thet0, self.thet, self.phi0, self.phi, self.alpha, 
            self.beta) = geom   
        

    def _init_tmatrix(self):
        """Initialize the T-matrix.
        """
        self.nmax = pytmatrix.calctmat(self.axi, self.rat, self.lam, 
            self.m.real, self.m.imag, self.eps, self.np, self.ddelt, 
            self.ndgs)
        self._tm_signature = (self.axi, self.rat, self.lam, self.m,
            self.eps, self.np, self.ddelt, self.ndgs)        


    def _init_orient(self):
        """Retrieve the quadrature points and weights.
        """
        (self.beta_p, self.beta_w) = pygaussqr.get_points_and_weights(
            self.or_pdf, 0, 180, self.n_beta)
        self._orient_signature = (self.or_pdf, self.n_alpha, self.n_beta)


    def _set_scatter_signature(self):
        """Mark the amplitude and scattering matrices as up to date.
        """
        self._scatter_signature = (self.thet0, self.thet, self.phi0, self.phi,
                                self.alpha, self.beta, self.scatter)
        self._orient_signature = (self.or_pdf, self.n_alpha, self.n_beta)


    def get_SZ(self):
        tm_outdated = self._tm_signature != (self.axi, self.rat, self.lam, 
            self.m, self.eps, self.np, self.ddelt, self.ndgs)
        if tm_outdated:
            self._init_tmatrix()

        orient_outdated = self._orient_signature != \
            (self.or_pdf, self.n_alpha, self.n_beta)
        if orient_outdated:
            self._init_orient()

        scatter_outdated = self._scatter_signature != (self.thet0, self.thet, 
            self.phi0, self.phi, self.alpha, self.beta, self.scatter)

        outdated = tm_outdated or orient_outdated or scatter_outdated
        if outdated:
            (self._S, self._Z) = self.scatter(self)
            self._set_scatter_signature()

        return (self._S, self._Z)


    def get_S(self):
        return self.get_SZ()[0]


    def get_Z(self):
        return self.get_SZ()[1]


    

    






