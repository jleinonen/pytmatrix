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
from pytmatrix.tmatrix import Scatterer
from pytmatrix.psd import PSDIntegrator, GammaPSD, BinnedPSD
import pytmatrix.tmatrix_aux as tmatrix_aux


class TMatrixPSD(Scatterer):
    """T-matrix class to perform computations over PSDs.

    This class derives from TMatrix to perform computations on particle
    ensembles given by a particle size distribution (PSD).

    Most class attributes can be set as in the TMatrix class, and there
    are several additional attributes. However, note that unlike with TMatrix,
    the results are based on lookup tables that are initialized with
    init_scatter_table. This means that after initializing 
    (init_scatter_table) or loading (load_scatter_table) the lookup tables,
    only changes to the PSD will have an effect on the results. To change
    other properties of the scattering problem (e.g. the refractive index m),
    you must rerun init_scatter_table.

    Initialize the class and set the attributes as described below. Call
    init_scatter_table to compute the lookup table for scattering values at
    different scatterer geometries. 

    The scattering properties can now be retrieved multiple times without
    re-initializing. Use set_geometry to select the geometry you need. 
    After this, you can call the same functions as with the TMatrix class to
    retrieve the scattering/radar parameters.

    Attributes:
        psd: set to a callable object giving the PSD value for a given 
            diameter (for example a GammaPSD instance); 
            default GammaPSD(D0=1.0)
        n_psd: the number of points for which to sample the PSD and scattering 
            properties for; default n_psd=500 should be good for most purposes
        psd_m_func: set to a callable object giving the refractive index as a
            function of diameter, or None to use the "m" attribute for all
            sizes; default None
        psd_eps_func: set to a callable object giving the aspect ratio
            (horizontal to rotational) as a function of diameter, or None to 
            use the "eps" attribute for all sizes; default None
        D_max: set to the maximum single scatterer size that is desired to be
            used (usually the D_max corresponding to the largest PSD you 
            intend to use)
        geoms: tuple containing the scattering geometry tuples that are 
            initialized (thet0, thet, phi0, phi, alpha, beta); 
            default horizontal backscatter
    """

    _psd_attr_list = set(["num_points", "m_func", "axis_ratio_func", "D_max", 
        "geometries"])

    _aliases = {"n_psd": "num_points",
        "psd_eps_func": "axis_ratio_func",
        "psd_m_func": "m_func"
        }


    def __init__(self, **kwargs):
        super(TMatrixPSD, self).__init__(**kwargs)

        if not kwargs.get("suppress_warning", False):
            warnings.simplefilter("always")
            warnings.warn("TMatrixPSD is deprecated and may be removed in " +
                "a future version. Use the PSDIntegrator class and the " + \
                "psd_integrator property of the Scatterer class instead.", 
                DeprecationWarning)
            warnings.filters.pop(0)

        self.num_points = 500
        self.m_func = None
        self.axis_ratio_func = None
        self.D_max = None
        self.geometries = (tmatrix_aux.geom_horiz_back,)

        self.psd_integrator = PSDIntegrator()        

        for attr in self._aliases:
            if attr in kwargs:
                self.__dict__[self._aliases[attr]] = kwargs[attr]
        for attr in self.__class__._psd_attr_list:
            if attr in kwargs:
                self.__dict__[attr] = kwargs[attr]

        self.set_geometry(self.geometries[0])


    def __setattr__(self, name, value):
        name = self._aliases.get(name, name)
        super(TMatrixPSD, self).__setattr__(name, value)


    def __getattr__(self, name):
        if name == "_aliases":
            raise AttributeError
        name = self._aliases.get(name, name)  
        return super(TMatrixPSD, self).__getattr__(name)


    def _copy_attrs(self):        
        for attr in self._psd_attr_list:
            if attr in self.__dict__:
                self.psd_integrator.__dict__[attr] = self.__dict__[attr]


    def init_scatter_table(self):
        self._copy_attrs()
        self.psd_integrator.init_scatter_table(self)


    def save_scatter_table(self, fn, description=""):
        self._copy_attrs()
        self.psd_integrator.save_scatter_table(fn, description=description)


    def load_scatter_table(self, fn):
        self.psd_integrator.load_scatter_table(fn)


   




    


    
