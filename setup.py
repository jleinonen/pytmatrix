#!/usr/bin/env python
# -*- coding: utf-8 -*-

long_description = """A Python code for computing the scattering properties
of homogeneous nonspherical scatterers with the T-Matrix method.

Requires NumPy and SciPy.
"""

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('pytmatrix', parent_package, top_path,
        version = '0.1.3',
        author  = "Jussi Leinonen",
        author_email = "jsleinonen@gmail.com",
        description = "T-matrix scattering computations",
        license = "MIT",
        url = 'http://code.google.com/p/pytmatrix/',
        download_url = \
            'http://pytmatrix.googlecode.com/files/pytmatrix-0.1.3.zip',
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

    config.add_extension('fortran_tm.pytmatrix',
        sources=['pytmatrix/fortran_tm/pytmatrix.pyf',
            'pytmatrix/fortran_tm/ampld.lp.f',
            'pytmatrix/fortran_tm/lpd.f'],
        )

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
