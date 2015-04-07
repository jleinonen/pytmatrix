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
from pytmatrix.scatter import ldr, ext_xsect


def radar_xsect(scatterer, h_pol=True):
    """Radar cross section for the current setup.    

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The radar cross section.
    """
    Z = scatterer.get_Z()
    if h_pol:
        return 2 * np.pi * \
            (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    else:
        return 2 * np.pi * \
            (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])


def refl(scatterer, h_pol=True):
    """Reflectivity (with number concentration N=1) for the current setup.

    Args:
        scatterer: a Scatterer instance.
        h_pol: If True (default), use horizontal polarization.
        If False, use vertical polarization.

    Returns:
        The reflectivity.

    NOTE: To compute reflectivity in dBZ, give the particle diameter and
    wavelength in [mm], then take 10*log10(Zi).
    """
    return scatterer.wavelength**4/(np.pi**5*scatterer.Kw_sqr) * \
        radar_xsect(scatterer, h_pol)

#alias for compatibility
Zi = refl


def Zdr(scatterer):
    """
    Differential reflectivity (Z_dr) for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
        The Z_dr.
    """
    return radar_xsect(scatterer, True)/radar_xsect(scatterer, False)


def delta_hv(scatterer):
    """
    Delta_hv for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
        Delta_hv [rad].
    """
    Z = scatterer.get_Z()
    return np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])


def rho_hv(scatterer):
    """
    Copolarized correlation (rho_hv) for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
       rho_hv.
    """
    Z = scatterer.get_Z()
    a = (Z[2,2] + Z[3,3])**2 + (Z[3,2] - Z[2,3])**2
    b = (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    c = (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])
    return np.sqrt(a / (b*c))


def Kdp(scatterer):
    """
    Specific differential phase (K_dp) for the current setup.

    Args:
        scatterer: a Scatterer instance.

    Returns:
        K_dp [deg/km].

    NOTE: This only returns the correct value if the particle diameter and
    wavelength are given in [mm]. The scatterer object should be set to 
    forward scattering geometry before calling this function.
    """
    if (scatterer.thet0 != scatterer.thet) or \
        (scatterer.phi0 != scatterer.phi):
        
        raise ValueError("A forward scattering geometry is needed to " + \
            "compute the specific differential phase.")

    S = scatterer.get_S()
    return 1e-3 * (180.0/np.pi) * scatterer.wavelength * (S[1,1]-S[0,0]).real


def Ai(scatterer, h_pol=True):
    """
    Specific attenuation (A) for the current setup.

    Parameters:
        h_pol (default True): compute attenuation for the horizontal 
        polarization. If False, use vertical polarization.

    Returns:
        A [dB/km].

    NOTE: This only returns the correct value if the particle diameter and
    wavelength are given in [mm]. The scatterer object should be set to 
    forward scattering geometry before calling this function.
    """
    return 4.343e-3 * ext_xsect(scatterer, h_pol=h_pol)


