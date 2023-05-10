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

import warnings
import numpy as np
from pytmatrix.fortran_tm import pytmatrix
from pytmatrix.quadrature import quadrature
import pytmatrix.orientation as orientation


class Scatterer(object):
    """T-Matrix scattering from nonspherical particles.

    Class for simulating scattering from nonspherical particles with the 
    T-Matrix method. Uses a wrapper to the Fortran code by M. Mishchenko.

    Usage instructions:

    First, the class should be be initialized. Any attributes (see below)
    can be passed as keyword arguments to the constructor. For example:
    sca = tmatrix.Scatterer(wavelength=2.0, m=complex(0,2))

    The properties of the scattering and the radiation should then be set 
    as attributes of this object. 

    The functions for computing the various scattering properties can then be 
    called. The Scatterer object will automatically recompute the T-matrix 
    and/or the amplitude and phase matrices when needed.

    Attributes: 
        radius: Equivalent radius.
        radius_type: If radius_type==Scatterer.RADIUS_EQUAL_VOLUME (default),
            radius is the equivalent volume radius.
            If radius_type==Scatterer.RADIUS_MAXIMUM, radius is the maximum 
            radius.
            If radius_type==Scatterer.RADIUS_EQUAL_AREA, 
            radius is the equivalent area radius.
        wavelength: The wavelength of incident light (same units as axi).
        m: The complex refractive index.
        axis_ratio: The horizontal-to-rotational axis ratio.
        shape: Particle shape. 
            Scatterer.SHAPE_SPHEROID: spheroid
            Scatterer.SHAPE_CYLINDER: cylinders; 
            Scatterer.SHAPE_CHEBYSHEV: Chebyshev particles (not yet 
                supported).
        alpha, beta: The Euler angles of the particle orientation (degrees).
        thet0, thet: The zenith angles of incident and scattered radiation 
            (degrees).
        phi0, phi: The azimuth angles of incident and scattered radiation 
            (degrees).
        Kw_sqr: The squared reference water dielectric factor for computing 
            radar reflectivity.
        orient: The function to use to compute the scattering properties.
            Should be one of the orientation module methods (orient_single, 
            orient_averaged_adaptive, orient_averaged_fixed).
        or_pdf: Particle orientation PDF for orientational averaging.
        n_alpha: Number of integration points in the alpha Euler angle.
        n_beta: Number of integration points in the beta Euler angle.
        psd_integrator: Set this to a PSDIntegrator instance to enable size
            distribution integration. If this is None (default), size 
            distribution integration is not used. See the PSDIntegrator
            documentation for more information.
        psd: Set to a callable object giving the PSD value for a given 
            diameter (for example a GammaPSD instance); default None. Has no
            effect if psd_integrator is None.
    """

    _attr_list = set(["radius", "radius_type", "wavelength", "m",
        "axis_ratio", "shape", "np", "ddelt", "ndgs", "alpha",
        "beta", "thet0", "thet", "phi0", "phi", "Kw_sqr", "orient",
        "scatter", "or_pdf", "n_alpha", "n_beta", "psd_integrator",
        "psd"])

    _deprecated_aliases = {"axi": "radius",
        "lam": "wavelength",
        "eps": "axis_ratio",
        "rat": "radius_type",
        "np": "shape",
        "scatter": "orient"
    }

    RADIUS_EQUAL_VOLUME = 1.0
    RADIUS_EQUAL_AREA = 0.0
    RADIUS_MAXIMUM = 2.0

    SHAPE_SPHEROID = -1
    SHAPE_CYLINDER = -2
    SHAPE_CHEBYSHEV = 1

    def __init__(self, **kwargs):
        self.radius = 1.0
        self.radius_type = Scatterer.RADIUS_EQUAL_VOLUME
        self.wavelength = 1.0
        self.m = complex(2,0)
        self.axis_ratio = 1.0
        self.shape = Scatterer.SHAPE_SPHEROID
        self.ddelt = 1e-3
        self.ndgs = 2
        self.alpha = 0.0
        self.beta = 0.0
        self.thet0 = 90.0
        self.thet = 90.0
        self.phi0 = 0.0
        self.phi = 180.0
        self.Kw_sqr = 0.93
        self.orient = orientation.orient_single
        self.or_pdf = orientation.gaussian_pdf()
        self.n_alpha = 5
        self.n_beta = 10

        self._tm_signature = ()        
        self._scatter_signature = ()
        self._orient_signature = ()
        self._psd_signature = ()
        
        self.psd_integrator = None
        self.psd = None    

        self.suppress_warning = kwargs["suppress_warning"] if \
            "suppress_warning" in kwargs else False

        for attr in self.__class__._deprecated_aliases:            
            if attr in kwargs:
                self._warn_deprecation(attr)
                self.__dict__[self._deprecated_aliases[attr]] = kwargs[attr]
        for attr in self._attr_list:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]


    def set_geometry(self, geom):
        """A convenience function to set the geometry variables.

        Args:
            geom: A tuple containing (thet0, thet, phi0, phi, alpha, beta).
            See the Scatterer class documentation for a description of these
            angles.
        """
        (self.thet0, self.thet, self.phi0, self.phi, self.alpha, 
            self.beta) = geom


    def get_geometry(self):
        """A convenience function to get the geometry variables.

        Returns:
            A tuple containing (thet0, thet, phi0, phi, alpha, beta).
            See the Scatterer class documentation for a description of these
            angles.
        """
        return (self.thet0, self.thet, self.phi0, self.phi, self.alpha, 
            self.beta)


    def _warn_deprecation(self, attr):
        if not self.suppress_warning:
            replacement = self._deprecated_aliases[attr]           
            warnings.simplefilter("always")
            warnings.warn(("The attribute '{attr}' is deprecated and may " + \
                "be removed in a future version. It has been renamed to " + \
                "'{replacement}'.").format(attr=attr, 
                replacement=replacement), DeprecationWarning)
            warnings.filters.pop(0)


    def __getattr__(self, name):
        if name == "_aliases":
            raise AttributeError
        if name in self._deprecated_aliases:
            self._warn_deprecation(name)
        name = self._deprecated_aliases.get(name, name)  
        return object.__getattribute__(self, name)


    def __setattr__(self, name, value):
        if name in self._deprecated_aliases:
            self._warn_deprecation(name)
        name = self._deprecated_aliases.get(name, name)
        object.__setattr__(self, name, value)  


    def _init_tmatrix(self):
        """Initialize the T-matrix.
        """

        if self.radius_type == Scatterer.RADIUS_MAXIMUM:
            # Maximum radius is not directly supported in the original
            # so we convert it to equal volume radius
            radius_type = Scatterer.RADIUS_EQUAL_VOLUME
            radius = self.equal_volume_from_maximum()
        else:
            radius_type = self.radius_type
            radius = self.radius

        self.nmax = pytmatrix.calctmat(radius, radius_type,
            self.wavelength, self.m.real, self.m.imag, self.axis_ratio,
            self.shape, self.ddelt, self.ndgs)
        self._tm_signature = (self.radius, self.radius_type, self.wavelength,
            self.m, self.axis_ratio, self.shape, self.ddelt, self.ndgs)        


    def _init_orient(self):
        """Retrieve the quadrature points and weights if needed.
        """
        if self.orient == orientation.orient_averaged_fixed:
            (self.beta_p, self.beta_w) = quadrature.get_points_and_weights(
                self.or_pdf, 0, 180, self.n_beta)
        self._set_orient_signature()


    def _set_scatter_signature(self):
        """Mark the amplitude and scattering matrices as up to date.
        """
        self._scatter_signature = (self.thet0, self.thet, self.phi0, self.phi,
            self.alpha, self.beta, self.orient)


    def _set_orient_signature(self):
        self._orient_signature = (self.orient, self.or_pdf, self.n_alpha, 
            self.n_beta)   
    

    def _set_psd_signature(self):
        self._psd_signature = (self.psd,)


    def equal_volume_from_maximum(self):
        if self.shape == Scatterer.SHAPE_SPHEROID:
            if self.axis_ratio > 1.0: # oblate
                r_eq = self.radius/self.axis_ratio**(1.0/3.0)
            else: # prolate
                r_eq = self.radius*self.axis_ratio**(2.0/3.0)
        elif self.shape == Scatterer.SHAPE_CYLINDER:
            if self.axis_ratio > 1.0: # oblate
                r_eq = self.radius*(1.5/self.axis_ratio)**(1.0/3.0)
            else: # prolate
                r_eq = self.radius*(1.5*self.axis_ratio**2)**(1.0/3.0)
        else:
            raise AttributeError("Unsupported shape for maximum radius.")
        return r_eq


    def get_SZ_single(self, alpha=None, beta=None):
        """Get the S and Z matrices for a single orientation.
        """
        if alpha == None:
            alpha = self.alpha
        if beta == None:
            beta = self.beta

        tm_outdated = self._tm_signature != (self.radius, self.radius_type, 
            self.wavelength, self.m, self.axis_ratio, self.shape, self.ddelt, 
            self.ndgs)
        if tm_outdated:
            self._init_tmatrix()

        scatter_outdated = self._scatter_signature != (self.thet0, self.thet, 
            self.phi0, self.phi, alpha, beta, self.orient)

        outdated = tm_outdated or scatter_outdated

        if outdated:
            (self._S_single, self._Z_single) = pytmatrix.calcampl(self.nmax, 
                self.wavelength, self.thet0, self.thet, self.phi0, self.phi, 
                alpha, beta)
            self._set_scatter_signature()

        return (self._S_single, self._Z_single)


    def get_SZ_orient(self):
        """Get the S and Z matrices using the specified orientation averaging.
        """

        tm_outdated = self._tm_signature != (self.radius, self.radius_type, 
            self.wavelength, self.m, self.axis_ratio, self.shape, self.ddelt, 
            self.ndgs)
        scatter_outdated = self._scatter_signature != (self.thet0, self.thet, 
            self.phi0, self.phi, self.alpha, self.beta, self.orient)

        orient_outdated = self._orient_signature != \
            (self.orient, self.or_pdf, self.n_alpha, self.n_beta)
        if orient_outdated:
            self._init_orient()
        
        outdated = tm_outdated or scatter_outdated or orient_outdated

        if outdated:
            (self._S_orient, self._Z_orient) = self.orient(self)
            self._set_scatter_signature()

        return (self._S_orient, self._Z_orient)


    def get_SZ(self):
        """Get the S and Z matrices using the current parameters.
        """
        if self.psd_integrator is None:
            (self._S, self._Z) = self.get_SZ_orient()
        else:
            scatter_outdated = self._scatter_signature != (self.thet0, 
                self.thet, self.phi0, self.phi, self.alpha, self.beta, 
                self.orient)            
            psd_outdated = self._psd_signature != (self.psd,)
            outdated = scatter_outdated or psd_outdated

            if outdated:
                (self._S, self._Z) = self.psd_integrator(self.psd, 
                    self.get_geometry())
                self._set_scatter_signature()
                self._set_psd_signature()

        return (self._S, self._Z)


    def get_S(self):
        return self.get_SZ()[0]


    def get_Z(self):
        return self.get_SZ()[1]



# Alias with a warning
class TMatrix(Scatterer):
    def __init__(self, **kwargs):
        if not kwargs.get("suppress_warning", False):
            warnings.simplefilter("always")
            warnings.warn("'TMatrix' is deprecated and may be removed in " +
                "a future version. It has been renamed to 'Scatterer'.", 
                DeprecationWarning)
            warnings.filters.pop(0)
        super(TMatrix, self).__init__(**kwargs)
