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

import numpy as np
from scipy.integrate import trapz
from scipy.special import gamma
try:
    import cPickle as pickle
except ImportError:
    import pickle
from datetime import datetime
from tmatrix import TMatrix
import tmatrix_aux


class GammaPSD(object):
    """Normalized gamma particle size distribution (PSD).
    
    Callable class to provide a normalized gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    Attributes:
        D0: the median volume diameter.
        Nw: the intercept parameter.
        mu: the shape parameter.
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            the class is initialized but must be manually changed afterwards)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, D0=1.0, Nw=1.0, mu=0.0):
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0 * D0
        self.Nw = float(Nw)
        self.nf = Nw * 6.0/3.67**4 * (3.67+mu)**(mu+4)/gamma(mu+4)

    def __call__(self, D):
        d = (D/self.D0)
        psd = self.nf * d**self.mu * np.exp(-(3.67+self.mu)*d)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, GammaPSD) and (self.D0 == other.D0) and \
                (self.Nw == other.Nw) and (self.mu == other.mu) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


class BinnedPSD(object):
    """Binned gamma particle size distribution (PSD).
    
    Callable class to provide a binned PSD with the given bin edges and PSD
    values.

    Args (constructor):
        The first argument to the constructor should specify n+1 bin edges, 
        and the second should specify n bin_psd values.        
        
    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters outside the bins.
    """
    
    def __init__(self, bin_edges, bin_psd):
        if len(bin_edges) != len(bin_psd)+1:
            raise ValueError("There must be n+1 bin edges for n bins.")
        
        self.bin_edges = bin_edges
        self.bin_psd = bin_psd
        
    def psd_for_D(self, D):       
        if not (self.bin_edges[0] < D <= self.bin_edges[-1]):
            return 0.0
        
        # binary search for the right bin
        start = 0
        end = len(self.bin_edges)
        while end-start > 1:
            half = (start+end)//2
            if self.bin_edges[start] < D <= self.bin_edges[half]:
                end = half
            else:
                start = half
                                
        return bin_psd[start]                    
        
    def __call__(self, D):
        if np.shape(D) == (): # D is a scalar
            return self.psd_for_D(D)
        else:
            return np.array([self.psd_for_D(d) for d in D])
    
    def __eq__(self, other):
        return len(self.bin_edges) == len(other.bin_edges) and \
            (self.bin_edges == other.bin_edges).all() and \
            (self.bin_psd == other.bin_psd).all()
                
                


class TMatrixPSD(TMatrix):
    """T-matrix class to perform computations over PSDs.

    This class derives from tmatrix to perform computations on particle
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

    def __init__(self, **kwargs):
        super(TMatrixPSD, self).__init__(**kwargs)
        self.psd = GammaPSD(1.0)
        self.n_psd = 500
        self.psd_m_func = None
        self.psd_eps_func = None
        self.D_max = self.psd.D_max
        self.geometries = (tmatrix_aux.geom_horiz_back,)
        self.set_geometry(self.geometries[0])
        self._S_table = None
        self._Z_table = None
        self._previous_psd = None
        attrs = ("psd", "n_psd", "psd_m_func", "psd_eps_func", "D_max",
            "geoms")
        for k in kwargs:
            if k in attrs:
                self.__dict__[k] = kwargs[k]


    def get_SZ(self):
        """
        Compute the scattering matrices for the given PSD and geometries.

        Returns:
            The new amplitude (S) and phase (Z) matrices.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError(
                "Initialize or load the scattering table first.")

        if self._previous_psd != self.psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = self.psd(self._psd_D)

            for geom in self.geometries:
                self._S_dict[geom] = \
                    trapz(self._S_table[geom] * psd_w, self._psd_D)
                self._Z_dict[geom] = \
                    trapz(self._Z_table[geom] * psd_w, self._psd_D)

            self._previous_psd = self.psd
        
        current_geom = (self.thet0, self.thet, self.phi0, self.phi, 
            self.alpha, self.beta)

        return (self._S_dict[current_geom], self._Z_dict[current_geom])


    def init_scatter_table(self):
        """Initialize the scattering lookup tables.
        
        Initialize the scattering lookup tables for the different geometries.
        Before calling this, the following attributes must be set:
           n_psd, psd_m_func, psd_eps_func, D_max, geoms
        and additionally, all the desired attributes of the tmatrix class
        (e.g. wavelength, aspect ratio).
        """
        self._psd_D = \
            np.linspace(self.D_max/self.n_psd, self.D_max, self.n_psd)

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None

        self._m_table = np.ndarray(self.n_psd, dtype=complex)
        for geom in self.geometries:
            self._S_table[geom] = \
                np.ndarray((2,2,self.n_psd), dtype=complex)
            self._Z_table[geom] = np.ndarray((4,4,self.n_psd))

        for (i,D) in enumerate(self._psd_D):
            if self.psd_m_func != None:
                self.m = self.psd_m_func(D)
            if self.psd_eps_func != None:
                self.eps = self.psd_eps_func(D)
            self._m_table[i] = self.m
            self.axi = D/2.0
            for geom in self.geometries:
                self.set_geometry(geom)
                (S, Z) = super(TMatrixPSD, self).get_SZ()
                self._S_table[geom][:,:,i] = S
                self._Z_table[geom][:,:,i] = Z

        self.set_geometry(self.geometries[0])


    def save_scatter_table(self, fn, description=""):
        """Save the scattering lookup tables.
        
        Save the state of the scattering lookup tables to a file.
        This can be loaded later with load_scatter_table.

        Other variables will not be saved, but this does not matter because
        the results of the computations are based only on the contents
        of the table.

        Args:
           fn: The name of the scattering table file. 
           description (optional): A description of the table.
        """
        data = {
           "description": description,
           "time": datetime.now(),
           "psd_scatter": (self.n_psd, self.D_max, self._psd_D, self._S_table,
                _self.Z_table, self._m_table, self.lam, self.eps, 
                self.geometries)
           }
        pickle.dump(data, file(fn, 'w'), pickle.HIGHEST_PROTOCOL)


    def load_scatter_table(self, fn):
        """Load the scattering lookup tables.
        
        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.            
        """
        data = pickle.load(file(fn))
        (self.n_psd, self.D_max, self._psd_D, self._S_table, self._Z_table,
            self._m_table, self.lam, self.eps, self.geometries) = \
            data["psd_scatter"]
        return (data["time"], data["description"])


    
