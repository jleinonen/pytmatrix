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

from datetime import datetime
try:
    import cPickle as pickle
except ImportError:
    import pickle
import warnings
import numpy as np
from scipy.integrate import trapz
from scipy.special import gamma
import pytmatrix.scatter as scatter
import pytmatrix.tmatrix_aux as tmatrix_aux


class PSD(object):
    def __call__(self, D):
        if np.shape(D) == ():
            return 0.0
        else:
            return np.zeros_like(D)

    def __eq__(self, other):
        return False


class ExponentialPSD(PSD):
    """Exponential particle size distribution (PSD).
    
    Callable class to provide an exponential PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * exp(-Lambda*D)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter        
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, N0=1.0, Lambda=1.0, D_max=None):
        self.N0 = float(N0)
        self.Lambda = float(Lambda)
        self.D_max = 11.0/Lambda if D_max is None else D_max

    def __call__(self, D):
        psd = self.N0 * np.exp(-self.Lambda*D)
        if np.shape(D) == ():
            if D > self.D_max:
                return 0.0
        else:
            psd[D > self.D_max] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, ExponentialPSD) and \
                (self.N0 == other.N0) and (self.Lambda == other.Lambda) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


class UnnormalizedGammaPSD(ExponentialPSD):
    """Gamma particle size distribution (PSD).
    
    Callable class to provide an gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = N0 * D**mu * exp(-Lambda*D)

    Attributes:
        N0: the intercept parameter.
        Lambda: the inverse scale parameter
        mu: the shape parameter
        D_max: the maximum diameter to consider (defaults to 11/Lambda,
            i.e. approx. 3*D0, if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """
    
    def __init__(self, N0=1.0, Lambda=1.0, mu=0.0, D_max=None):
        super(UnnormalizedGammaPSD, self).__init__(N0=N0, Lambda=Lambda, 
            D_max=D_max)
        self.mu = mu

    def __call__(self, D):
        # For large mu, this is better numerically than multiplying by D**mu
        psd = self.N0 * np.exp(self.mu*np.log(D)-self.Lambda*D)
        if np.shape(D) == ():
            if (D > self.D_max) or (D==0):
                return 0.0
        else:
            psd[(D > self.D_max) | (D == 0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return super(UnnormalizedGammaPSD, self).__eq__(other) and \
                self.mu == other.mu
        except AttributeError:
            return False
        


class GammaPSD(PSD):
    """Normalized gamma particle size distribution (PSD).
    
    Callable class to provide a normalized gamma PSD with the given 
    parameters. The attributes can also be given as arguments to the 
    constructor.

    The PSD form is:
    N(D) = Nw * f(mu) * (D/D0)**mu * exp(-(3.67+mu)*D/D0)
    f(mu) = 6/(3.67**4) * (3.67+mu)**(mu+4)/Gamma(mu+4)

    Attributes:
        D0: the median volume diameter.
        Nw: the intercept parameter.
        mu: the shape parameter.
        D_max: the maximum diameter to consider (defaults to 3*D0 when
            if None)

    Args (call):
        D: the particle diameter.

    Returns (call):
        The PSD value for the given diameter.    
        Returns 0 for all diameters larger than D_max.
    """

    def __init__(self, D0=1.0, Nw=1.0, mu=0.0, D_max=None):
        self.D0 = float(D0)
        self.mu = float(mu)
        self.D_max = 3.0*D0 if D_max is None else D_max
        self.Nw = float(Nw)
        self.nf = Nw * 6.0/3.67**4 * (3.67+mu)**(mu+4)/gamma(mu+4)

    def __call__(self, D):
        d = (D/self.D0)
        psd = self.nf * np.exp(self.mu*np.log(d)-(3.67+self.mu)*d)
        if np.shape(D) == ():
            if (D > self.D_max) or (D==0.0):
                return 0.0
        else:
            psd[(D > self.D_max) | (D==0.0)] = 0.0
        return psd

    def __eq__(self, other):
        try:
            return isinstance(other, GammaPSD) and (self.D0 == other.D0) and \
                (self.Nw == other.Nw) and (self.mu == other.mu) and \
                (self.D_max == other.D_max)
        except AttributeError:
            return False


class BinnedPSD(PSD):
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
                                
        return self.bin_psd[start]                    
        
    def __call__(self, D):
        if np.shape(D) == (): # D is a scalar
            return self.psd_for_D(D)
        else:
            return np.array([self.psd_for_D(d) for d in D])
    
    def __eq__(self, other):
        if other is None:
            return False
        return len(self.bin_edges) == len(other.bin_edges) and \
            (self.bin_edges == other.bin_edges).all() and \
            (self.bin_psd == other.bin_psd).all()


class PSDIntegrator(object):
    """A class used to perform computations over PSDs.

    This class can be used to integrate scattering properties over particle
    size distributions.

    Initialize an instance of the class and set the attributes as described
    below. Call init_scatter_table to compute the lookup table for scattering
    values at different scatterer geometries. Set the class instance as the
    psd_integrator attribute of a Scatterer object to enable PSD averaging for
    that object.

    After a call to init_scatter_table, the scattering properties can be
    retrieved multiple times without re-initializing. However, the geometry of
    the Scatterer instance must be set to one of those specified in the
    "geometries" attribute.

    Attributes:
        
        num_points: the number of points for which to sample the PSD and 
            scattering properties for; default num_points=1024 should be good
            for most purposes
        m_func: set to a callable object giving the refractive index as a
            function of diameter, or None to use the "m" attribute of the
            Scatterer for all sizes; default None
        axis_ratio_func: set to a callable object giving the aspect ratio
            (horizontal to rotational) as a function of diameter, or None to 
            use the "axis_ratio" attribute for all sizes; default None
        D_max: set to the maximum single scatterer size that is desired to be
            used (usually the D_max corresponding to the largest PSD you 
            intend to use)
        geometries: tuple containing the scattering geometry tuples that are 
            initialized (thet0, thet, phi0, phi, alpha, beta); 
            default horizontal backscatter
    """

    attrs = set(["num_points", "m_func", "axis_ratio_func", "D_max", 
        "geometries"])

    def __init__(self, **kwargs):      
        self.num_points = 1024
        self.m_func = None
        self.axis_ratio_func = None
        self.D_max = None
        self.geometries = (tmatrix_aux.geom_horiz_back,)

        for k in kwargs:
            if k in self.__class__.attrs:
                self.__dict__[k] = kwargs[k]

        self._S_table = None
        self._Z_table = None
        self._angular_table = None
        self._previous_psd = None


    def __call__(self, psd, geometry):
        return self.get_SZ(psd, geometry)


    def get_SZ(self, psd, geometry):
        """
        Compute the scattering matrices for the given PSD and geometries.

        Returns:
            The new amplitude (S) and phase (Z) matrices.
        """
        if (self._S_table is None) or (self._Z_table is None):
            raise AttributeError(
                "Initialize or load the scattering table first.")

        if (not isinstance(psd, PSD)) or self._previous_psd != psd:
            self._S_dict = {}
            self._Z_dict = {}
            psd_w = psd(self._psd_D)

            for geom in self.geometries:
                self._S_dict[geom] = \
                    trapz(self._S_table[geom] * psd_w, self._psd_D)
                self._Z_dict[geom] = \
                    trapz(self._Z_table[geom] * psd_w, self._psd_D)

            self._previous_psd = psd

        return (self._S_dict[geometry], self._Z_dict[geometry])


    def get_angular_integrated(self, psd, geometry, property_name):
        if self._angular_table is None:
            raise AttributeError(
                "Initialize or load the table of angular-integrated " + 
                "quantities first."
            )

        psd_w = psd(self._psd_D)

        def sca_xsect(geom):
            return trapz(
                self._angular_table["sca_xsect"][geom] * psd_w, 
                self._psd_D
            )
    
        if property_name == "sca_xsect":
            sca_prop = sca_xsect(geometry)
        elif property_name == "ext_xsect":
            sca_prop = trapz(
                self._angular_table["ext_xsect"][geometry] * psd_w, 
                self._psd_D
            )
        elif property_name == "asym":
            sca_xsect_int = sca_xsect(geometry)
            if sca_xsect_int > 0:
                sca_prop = trapz(
                    self._angular_table["asym"][geometry] * \
                    self._angular_table["sca_xsect"][geometry] * psd_w,  
                    self._psd_D
                )
                sca_prop /= sca_xsect_int
            else:
                sca_prop = 0.0

        return sca_prop


    def init_scatter_table(self, tm, angular_integration=False, verbose=False):
        """Initialize the scattering lookup tables.
        
        Initialize the scattering lookup tables for the different geometries.
        Before calling this, the following attributes must be set:
           num_points, m_func, axis_ratio_func, D_max, geometries
        and additionally, all the desired attributes of the Scatterer class
        (e.g. wavelength, aspect ratio).

        Args:
            tm: a Scatterer instance.
            angular_integration: If True, also calculate the 
                angle-integrated quantities (scattering cross section, 
                extinction cross section, asymmetry parameter). These are 
                needed to call the corresponding functions in the scatter 
                module when PSD integration is active. The default is False.
            verbose: if True, print information about the progress of the 
                calculation (which may take a while). If False (default), 
                run silently.
        """
        self._psd_D = np.linspace(self.D_max/self.num_points, self.D_max, 
            self.num_points)

        self._S_table = {}
        self._Z_table = {}
        self._previous_psd = None
        self._m_table = np.empty(self.num_points, dtype=complex)
        if angular_integration:
            self._angular_table = {"sca_xsect": {}, "ext_xsect": {}, 
                "asym": {}}
        else:
            self._angular_table = None
        
        (old_m, old_axis_ratio, old_radius, old_geom, old_psd_integrator) = \
            (tm.m, tm.axis_ratio, tm.radius, tm.get_geometry(), 
                tm.psd_integrator)
        
        try:
            # temporarily disable PSD integration to avoid recursion
            tm.psd_integrator = None 

            for geom in self.geometries:
                self._S_table[geom] = \
                    np.empty((2,2,self.num_points), dtype=complex)
                self._Z_table[geom] = np.empty((4,4,self.num_points))

                if angular_integration:
                    for int_var in ["sca_xsect", "ext_xsect", "asym"]:
                        self._angular_table[int_var][geom] = \
                            np.empty(self.num_points)

            for (i,D) in enumerate(self._psd_D):
                if verbose:
                    print("Computing point {i} at D={D}...".format(i=i, D=D))
                if self.m_func != None:
                    tm.m = self.m_func(D)
                if self.axis_ratio_func != None:
                    tm.axis_ratio = self.axis_ratio_func(D)
                self._m_table[i] = tm.m
                tm.radius = D/2.0
                for geom in self.geometries:
                    tm.set_geometry(geom)
                    (S, Z) = tm.get_SZ_orient()
                    self._S_table[geom][:,:,i] = S
                    self._Z_table[geom][:,:,i] = Z

                    if angular_integration:
                        self._angular_table["sca_xsect"][geom][i] = \
                            scatter.sca_xsect(tm)
                        self._angular_table["ext_xsect"][geom][i] = \
                            scatter.ext_xsect(tm)
                        self._angular_table["asym"][geom][i] = \
                            scatter.asym(tm)
        finally:
            #restore old values
            (tm.m, tm.axis_ratio, tm.radius, tm.psd_integrator) = \
                (old_m, old_axis_ratio, old_radius, old_psd_integrator) 
            tm.set_geometry(old_geom)



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
           "psd_scatter": (self.num_points, self.D_max, self._psd_D, 
                self._S_table, self._Z_table, self._angular_table, 
                self._m_table, self.geometries),
           "version": tmatrix_aux.VERSION
           }
        pickle.dump(data, file(fn, 'w'), pickle.HIGHEST_PROTOCOL)


    def load_scatter_table(self, fn):
        """Load the scattering lookup tables.
        
        Load the scattering lookup tables saved with save_scatter_table.

        Args:
            fn: The name of the scattering table file.            
        """
        data = pickle.load(file(fn))

        if ("version" not in data) or (data["version"]!=tmatrix_aux.VERSION):
            warnings.warn("Loading data saved with another version.", Warning)

        (self.num_points, self.D_max, self._psd_D, self._S_table, 
            self._Z_table, self._angular_table, self._m_table, 
            self.geometries) = data["psd_scatter"]
        return (data["time"], data["description"])

