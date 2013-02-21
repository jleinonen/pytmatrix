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


def radar_xsect(tm, h_pol=True):
    """Radar cross section for the current setup.    

    Args:
        h_pol: If true (default), use horizontal polarization.
        If false, use vertical polarization.

    Returns:
        The scattering cross section.
    """
    Z = tm.get_Z()
    if h_pol:
        return 2 * np.pi * \
            (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    else:
        return 2 * np.pi * \
            (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])


def refl(tm, h_pol=True):
    """Reflectivity (with number concentration N=1) for the current setup.

    Args:
        h_pol: If true (default), use horizontal polarization.
        If false, use vertical polarization.

    Returns:
        The reflectivity.

    NOTE: To compute reflectivity in dBZ, give the particle diameter and
    wavelength in [mm], then take 10*log10(Zi).
    """
    return tm.lam**4/(np.pi**5*tm.Kw_sqr) * radar_xsect(tm, h_pol)


def Zi(tm, h_pol=True):
    return refl(tm, h_pol)


def ldr(tm, h_pol=True):
    """
    Linear depolarizarion ratio (LDR) for the current setup

    Args:
        h_pol: If true (default), return LDR_h.
        If false, return LDR_v.

    Returs:
       The LDR.
    """
    Z = tm.get_Z()
    if h_pol:
        return (Z[0,0] - Z[0,1] + Z[1,0] - Z[1,1]) / \
               (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    else:
        return (Z[0,0] + Z[0,1] - Z[1,0] - Z[1,1]) / \
               (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])


def Zdr(tm):
    """
    Differential reflectivity (Z_dr) for the current setup

    Returs:
       The Z_dr.
    """
    return radar_xsect(tm, True)/radar_xsect(tm, False)


def delta_hv(tm):
    """
    Delta_hv for the current setup

    Returs:
       Delta_hv [rad].
    """
    Z = tm.get_Z()
    return np.arctan2(Z[2,3] - Z[3,2], -Z[2,2] - Z[3,3])


def rho_hv(tm):
    """
    Copolarized correlation (rho_hv) for the current setup

    Returs:
       rho_hv.
    """
    Z = tm.get_Z()
    a = (Z[2,2] + Z[3,3])**2 + (Z[3,2] - Z[2,3])**2
    b = (Z[0,0] - Z[0,1] - Z[1,0] + Z[1,1])
    c = (Z[0,0] + Z[0,1] + Z[1,0] + Z[1,1])
    return np.sqrt(a / (b*c))


def Kdp(tm):
    """
    Specific differential phase (K_dp) for the current setup

    Returs:
       K_dp [deg/km].

    NOTE: This only returns the correct value if the particle diameter and
    wavelength are given in [mm].
    """
    S = tm.get_S()
    return 1e-3 * (180.0/np.pi) * tm.lam * (S[1,1]-S[0,0]).real


def Ai(tm, h_pol=True):
    """
    Specific attenuation (A) for the current setup

    Parameters:
       h_pol (default True): compute attenuation for the horizontal polarization.
          If False, use vertical polarization.

    Returs:
       A [dB/km].

    NOTE: This only returns the correct value if the particle diameter and
    wavelength are given in [mm].
    """
    S = tm.get_S()
    if h_pol:
        return 8.686e-3 * tm.lam * S[1,1].imag
    else:
        return 8.686e-3 * tm.lam * S[0,0].imag