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

#current version
VERSION = "0.3.3"

#typical wavelengths [mm] at different bands
wl_S = 111.0
wl_C = 53.5
wl_X = 33.3
wl_Ku = 22.0
wl_Ka = 8.43
wl_W = 3.19

#typical values of K_w_sqr at different bands
K_w_sqr = {wl_S: 0.93, wl_C: 0.93, wl_X: 0.93, wl_Ku: 0.93, wl_Ka: 0.92, 
  wl_W: 0.75}

#preset geometries
geom_horiz_back = (90.0, 90.0, 0.0, 180.0, 0.0, 0.0) #horiz. backscatter
geom_horiz_forw = (90.0, 90.0, 0.0, 0.0, 0.0, 0.0) #horiz. forward scatter
geom_vert_back = (0.0, 180.0, 0.0, 0.0, 0.0, 0.0) #vert. backscatter
geom_vert_forw = (180.0, 180.0, 0.0, 0.0, 0.0, 0.0) #vert. forward scatter

#Drop Shape Relationship Functions


def dsr_thurai_2007(D_eq):
    """
    Drop shape relationship function from Thurai2007
    (http://dx.doi.org/10.1175/JTECH2051.1) paper.
    Arguments:
        D_eq: Drop volume-equivalent diameter (mm)

    Returns:
        r: The vertical-to-horizontal drop axis ratio. Note: the Scatterer class
        expects horizontal to vertical, so you should pass 1/dsr_thurai_2007
    """

    if D_eq < 0.7:
        return 1.0
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
            8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
            4.095e-5*D_eq**4


def dsr_pb(D_eq):
    """
    Pruppacher and Beard drop shape relationship function.

    Arguments:
        D_eq: Drop volume-equivalent diameter (mm)
    Returns:
        r: The vertical-to-horizontal drop axis ratio. Note: the Scatterer class
        expects horizontal to vertical, so you should pass 1/dsr_pb
    """
    return 1.03 - 0.062*D_eq


def dsr_bc(D_eq):
    """
    Beard and Chuang drop shape relationship function.
    Arguments:
        D_eq: Drop volume-equivalent diameter (mm)
    Returns:
        r: The vertical-to-horizontal drop axis ratio. Note: the Scatterer class
        expects horizontal to vertical, so you should pass 1/dsr_bc
    """

    return 1.0048 + 5.7e-04*D_eq - 2.628e-02 * D_eq**2 + \
        3.682e-03*D_eq**3 - 1.677e-04 * D_eq**4
