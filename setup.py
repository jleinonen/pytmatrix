#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copyright (C) 2009-2017 Jussi Leinonen, Finnish Meteorological Institute, 
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

long_description = """A Python code for computing the scattering properties
of homogeneous nonspherical scatterers with the T-Matrix method.

Requires NumPy and SciPy.
"""

import sys

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pytmatrix', parent_package, top_path,
        version = '0.3.2',
        author  = "Jussi Leinonen",
        author_email = "jsleinonen@gmail.com",
        description = "T-matrix scattering computations",
        license = "MIT",
        url = 'https://github.com/jleinonen/pytmatrix',
        download_url = \
            'https://github.com/jleinonen/pytmatrix/releases/download/0.3.1/pytmatrix-0.3.1.zip',
        long_description = long_description,
        classifiers = [
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Fortran",
            "Programming Language :: Python",
            "Topic :: Scientific/Engineering :: Physics",
        ]
    )

    kw = {}
    if sys.platform == 'darwin':
        kw['extra_link_args'] = ['-undefined dynamic_lookup', '-bundle']
    config.add_extension('fortran_tm.pytmatrix',
        sources=['pytmatrix/fortran_tm/pytmatrix.pyf',
            'pytmatrix/fortran_tm/ampld.lp.f',
            'pytmatrix/fortran_tm/lpd.f'],
        **kw)

    return config


if __name__ == "__main__":

    from numpy.distutils.core import setup
    setup(configuration=configuration,
        packages = ['pytmatrix','pytmatrix.test','pytmatrix.quadrature',
            'pytmatrix.fortran_tm'],        
        package_data = {
            'pytmatrix': ['ice_refr.dat'],
            'pytmatrix.fortran_tm': ['ampld.par.f']
        },
        platforms = ['any'],
        requires = ['numpy', 'scipy'])
