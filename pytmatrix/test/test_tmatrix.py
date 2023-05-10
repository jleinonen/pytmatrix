"""
Copyright (C) 2009-2023 Jussi Leinonen

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

import unittest
import numpy as np
from pytmatrix.tmatrix import TMatrix, Scatterer
from pytmatrix.tmatrix_psd import TMatrixPSD
from pytmatrix import orientation
from pytmatrix import radar
from pytmatrix import refractive
from pytmatrix import tmatrix_aux
from pytmatrix import psd
from pytmatrix import scatter


#some allowance for rounding errors etc
epsilon = 1e-7


def run_tests():
    """Tests for the T-matrix code.

       Runs several tests that test the code. All tests should return ok.
       If they don't, please contact the author.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(TMatrixTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    
def test_backend():
    """Replicate the test run of the backend T-Matrix code.
       
       Replicates the results of the default test run of the backend code
       by Mishchenko. The user can use the function to manually check that 
       the results match. Small errors may be present due to different compiler
       optimizations.
    """    
    scatterer = Scatterer(radius=10.0, rat=0.1, wavelength=2*np.pi, 
        m=complex(1.5,0.02), axis_ratio=0.5, ddelt=1e-3, ndgs=2, np=-1)
    scatterer.thet0 = 56.0
    scatterer.thet = 65.0
    scatterer.phi0 = 114.0
    scatterer.phi = 128.0
    scatterer.alpha = 145.0
    scatterer.beta = 52.0
        
    print("Amplitude matrix S:")
    print(scatterer.get_S())
    print("Phase matrix Z:")
    print(scatterer.get_Z())


class TMatrixTests(unittest.TestCase):

    def test_single(self):
        """Test a single-orientation case
        """
        tm = TMatrix(axi=2.0, lam=6.5, m=complex(1.5,0.5), eps=1.0/0.6,
            suppress_warning=True)
        (S, Z) = tm.get_SZ()

        S_ref = np.array(
            [[complex(3.89338755e-02, -2.43467777e-01),
              complex(-1.11474042e-24, -3.75103868e-24)],
             [complex(1.11461702e-24, 3.75030914e-24),
              complex(-8.38637654e-02, 3.10409912e-01)]])

        Z_ref = np.array(
            [[8.20899248e-02, -2.12975199e-02, -1.94051304e-24,
              2.43057373e-25],
            [-2.12975199e-02, 8.20899248e-02, 2.00801268e-25,
             -1.07794906e-24],
            [1.94055633e-24, -2.01190190e-25, -7.88399525e-02,
             8.33266362e-03],
            [2.43215306e-25,  -1.07799010e-24,  -8.33266362e-03,
             -7.88399525e-02]])

        test_relative(self, S, S_ref)
        test_relative(self, Z, Z_ref)


    def test_adaptive_orient(self):
        """Test an adaptive orientation averaging case
        """
        tm = TMatrix(axi=2.0, lam=6.5, m=complex(1.5,0.5), eps=1.0/0.6,
            suppress_warning=True)
        tm.or_pdf = orientation.gaussian_pdf(20.0)
        tm.orient = orientation.orient_averaged_adaptive
        (S, Z) = tm.get_SZ()
        
        S_ref = np.array(
            [[complex(6.49005717e-02, -2.42488000e-01), 
              complex(-6.12697676e-16, -4.10602248e-15)],
             [complex(-1.50048180e-14, -1.64195485e-15), 
              complex(-9.54176591e-02, 2.84758322e-01)]])

        Z_ref = np.array(
            [[7.89677648e-02, -1.37631854e-02, -7.45412599e-15, 
             -9.23979111e-20],
            [-1.37631854e-02, 7.82165256e-02, 5.61975938e-15,
             -1.32888054e-15],
            [8.68047418e-15, 3.52110917e-15, -7.73358177e-02,
             5.14571155e-03],
            [1.31977116e-19, -3.38136420e-15, -5.14571155e-03, 
             -7.65845784e-02]])

        test_relative(self, S, S_ref)
        test_relative(self, Z, Z_ref)               


    def test_fixed_orient(self):
        """Test a fixed-point orientation averaging case
        """
        tm = TMatrix(axi=2.0, lam=6.5, m=complex(1.5,0.5), eps=1.0/0.6,
            suppress_warning=True)
        tm.or_pdf = orientation.gaussian_pdf(20.0)
        tm.orient = orientation.orient_averaged_fixed
        (S, Z) = tm.get_SZ()
        
        S_ref = np.array(
            [[complex(6.49006090e-02, -2.42487917e-01),
              complex(1.20257317e-11, -5.23022168e-11)],
             [complex(6.21754594e-12, 2.95662844e-11),
              complex(-9.54177082e-02, 2.84758158e-01)]])

        Z_ref = np.array(
            [[7.89748116e-02, -1.37649947e-02, -1.58053610e-11, 
                -4.56295798e-12],
             [-1.37649947e-02, 7.82237468e-02, -2.85105399e-11,
                -3.43475094e-12],
             [2.42108565e-11, -3.92054806e-11, -7.73426425e-02,
                5.14654926e-03],
             [4.56792369e-12, -3.77838854e-12, -5.14654926e-03,
                -7.65915776e-02]])

        test_relative(self, S, S_ref)
        test_relative(self, Z, Z_ref)


    def test_psd(self):
        """Test a case that integrates over a particle size distribution
        """
        tm = Scatterer(wavelength=6.5, m=complex(1.5,0.5), axis_ratio=1.0/0.6)
        tm.psd_integrator = psd.PSDIntegrator()
        tm.psd_integrator.num_points = 500
        tm.psd = psd.GammaPSD(D0=1.0, Nw=1e3, mu=4)
        tm.psd_integrator.D_max = 10.0
        tm.psd_integrator.init_scatter_table(tm)
        (S, Z) = tm.get_SZ()

        S_ref = np.array(
            [[complex(1.02521928e+00, 6.76066598e-01), 
              complex(6.71933838e-24, 6.83819665e-24)],
             [complex(-6.71933678e-24, -6.83813546e-24), 
              complex(-1.10464413e+00, -1.05571494e+00)]])

        Z_ref = np.array(
            [[7.20540295e-02, -1.54020475e-02, -9.96222107e-25,
             8.34246458e-26],
            [-1.54020475e-02, 7.20540295e-02, 1.23279391e-25,
             1.40049088e-25],
            [9.96224596e-25, -1.23291269e-25, -6.89739108e-02,
             1.38873290e-02],
            [8.34137617e-26, 1.40048866e-25, -1.38873290e-02,
             -6.89739108e-02]])
        
        test_relative(self, S, S_ref)
        test_relative(self, Z, Z_ref)


    def test_radar(self):
        """Test that the radar properties are computed correctly
        """
        tm = TMatrixPSD(lam=tmatrix_aux.wl_C, 
            m=refractive.m_w_10C[tmatrix_aux.wl_C], suppress_warning=True)
        tm.psd = psd.GammaPSD(D0=2.0, Nw=1e3, mu=4)        
        tm.psd_eps_func = lambda D: 1.0/drop_ar(D)
        tm.D_max = 10.0
        tm.or_pdf = orientation.gaussian_pdf(20.0)
        tm.orient = orientation.orient_averaged_fixed
        tm.geometries = (tmatrix_aux.geom_horiz_back, 
            tmatrix_aux.geom_horiz_forw)
        tm.init_scatter_table()

        radar_xsect_h = radar.radar_xsect(tm)
        Z_h = radar.refl(tm)
        Z_v = radar.refl(tm, False)
        ldr = radar.ldr(tm)
        Zdr = radar.Zdr(tm)
        delta_hv = radar.delta_hv(tm)
        rho_hv = radar.rho_hv(tm)
        tm.set_geometry(tmatrix_aux.geom_horiz_forw)
        Kdp = radar.Kdp(tm)
        A_h = radar.Ai(tm)
        A_v = radar.Ai(tm, False)

        radar_xsect_h_ref = 0.22176446239750278
        Z_h_ref = 6383.7337897299258
        Z_v_ref = 5066.721040036321
        ldr_ref = 0.0021960626647629547
        Zdr_ref = 1.2599339374097778
        delta_hv_ref = -0.00021227778705544846
        rho_hv_ref = 0.99603080460983828
        Kdp_ref = 0.19334678024367824
        A_h_ref = 0.018923976733777458
        A_v_ref = 0.016366340549483317

        for (val, ref) in zip(
            (radar_xsect_h, Z_h, Z_v, ldr, Zdr, delta_hv, rho_hv, Kdp, A_h, 
                A_v),
            (radar_xsect_h_ref, Z_h_ref, Z_v_ref, ldr_ref, Zdr_ref, 
                delta_hv_ref, rho_hv_ref, Kdp_ref, A_h_ref, A_v_ref)):
            test_relative(self, val, ref)


    def test_rayleigh(self):
        """Test match with Rayleigh scattering for small spheres
        """
        wl = 100.0
        r = 1.0
        eps = 1.0
        m = complex(1.5,0.5)
        tm = Scatterer(wavelength=wl, radius=r, axis_ratio=eps, m=m)
        S = tm.get_S()

        k = 2*np.pi/wl
        S_ray = k**2 * (m**2-1)/(m**2+2) * r
        
        test_relative(self, S[0,0], S_ray, limit=1e-3)
        test_relative(self, S[1,1], -S_ray, limit=1e-3)        
        test_less(self, abs(S[0,1]), 1e-25)
        test_less(self, abs(S[1,0]), 1e-25)


    def test_optical_theorem(self):
        """Optical theorem: test that for a lossless particle, Csca=Cext
        """
        tm = Scatterer(radius=4.0, wavelength=6.5, m=complex(1.5,0.0), 
            axis_ratio=1.0/0.6)
        tm.set_geometry(tmatrix_aux.geom_horiz_forw)
        ssa_h = scatter.ssa(tm, True)
        ssa_v = scatter.ssa(tm, False)

        test_less(self, abs(1.0-ssa_h), 1e-6)
        test_less(self, abs(1.0-ssa_v), 1e-6)


    def test_asymmetry(self):
        """Test calculation of the asymmetry parameter
        """
        tm = Scatterer(radius=4.0, wavelength=6.5, m=complex(1.5,0.5), 
            axis_ratio=1.0)
        tm.set_geometry(tmatrix_aux.geom_horiz_forw)
        asym_horiz = scatter.asym(tm)
        tm.set_geometry(tmatrix_aux.geom_vert_forw)
        asym_vert = scatter.asym(tm)
        # Is the asymmetry parameter the same for a sphere in two directions?
        test_less(self, abs(1-asym_horiz/asym_vert), 1e-6)

        # Is the asymmetry parameter zero for small particles?
        tm.radius = 0.0004
        tm.set_geometry(tmatrix_aux.geom_horiz_forw)
        asym_horiz = scatter.asym(tm)
        test_less(self, abs(asym_horiz), 1e-8)


    def test_against_mie(self):
        """Test scattering parameters against Mie results
        """
        # Reference values computed with the Mie code of Maetzler
        sca_xsect_ref = 4.4471684294079958
        ext_xsect_ref = 7.8419745883848435
        asym_ref = 0.76146646088675629

        sca = Scatterer(wavelength=1, radius=1, m=complex(3.0,0.5))
        sca_xsect = scatter.sca_xsect(sca)
        ext_xsect = scatter.ext_xsect(sca)
        asym = scatter.asym(sca)

        test_less(self, abs(1-sca_xsect/sca_xsect_ref), 1e-6)
        test_less(self, abs(1-ext_xsect/ext_xsect_ref), 1e-6)
        test_less(self, abs(1-asym/asym_ref), 1e-6)


    def test_integrated_x_sca(self):
        """Test Rayleigh scattering cross section integrated over sizes.
        """

        m = complex(3.0,0.5)
        K = (m**2-1)/(m**2+2)
        N0 = 10
        Lambda = 1e4

        sca = Scatterer(wavelength=1, m=m)
        sca.psd_integrator = psd.PSDIntegrator()        
        sca.psd = psd.ExponentialPSD(N0=N0, Lambda=Lambda)
        sca.psd.D_max = 0.002
        sca.psd_integrator.D_max = sca.psd.D_max
        # 256 is quite low, but we want to run the test reasonably fast
        sca.psd_integrator.num_points = 256
        sca.psd_integrator.init_scatter_table(sca, angular_integration=True)

        # This size-integrated scattering cross section has an analytical value.
        # Check that we can reproduce it.
        sca_xsect_ref = 480*N0*np.pi**5*abs(K)**2/Lambda**7
        sca_xsect = scatter.sca_xsect(sca)
        test_less(self, abs(1-sca_xsect/sca_xsect_ref), 1e-3)

    def test_attn_polarization(self):
        """Test attenuation calculation for multiple polarization with PSD.
        """

        wavelength = tmatrix_aux.wl_C
        m = refractive.m_w_20C[wavelength]
        K = (m**2-1)/(m**2+2)
        diam_max = 0.75
        gamma_psd = psd.GammaPSD(D0=0.25, Nw=10e3, mu=0, D_max=diam_max)

        sca = Scatterer(wavelength=wavelength, m=m)
        sca.axis_ratio = 1.0/0.6
        sca.psd_integrator = psd.PSDIntegrator()        
        sca.psd = gamma_psd
        sca.psd_integrator.D_max = sca.psd.D_max
        # 64 is quite low, but we want to run the test reasonably fast
        sca.psd_integrator.num_points = 64
        sca.psd_integrator.init_scatter_table(sca, angular_integration=True)

        self.assertEqual(radar.Ai(sca), radar.Ai(sca, h_pol=True))
        test_less(self, 0, radar.Ai(sca, h_pol=True))
        test_less(self, 0, radar.Ai(sca, h_pol=False))
        # Check that we have differential attenuation
        test_less(self, radar.Ai(sca, h_pol=False), radar.Ai(sca, h_pol=True))
        

def test_relative(tests, x, x_ref, limit=epsilon):
    abs_diff = abs(x-x_ref)
    rel_diff = abs_diff/abs(x_ref)
    try:
        shape = x.shape
        if not shape:
            raise AttributeError()
        for i in range(shape[0]):
            for j in range(shape[1]):
                tests.assertTrue(abs_diff[i,j] < 1e-15 or \
                    rel_diff[i,j] < epsilon)            
    except AttributeError:
        test_less(tests, rel_diff, limit)


def test_less(tests, value, limit):
    try:
        tests.assertLess(value, limit)
    except AttributeError: # in Python 2.6 which has no assertLess
        tests.assertTrue(value < limit)


# For testing variable aspect ratio
def drop_ar(D_eq):
    if D_eq < 0.7:
        return 1.0;
    elif D_eq < 1.5:
        return 1.173 - 0.5165*D_eq + 0.4698*D_eq**2 - 0.1317*D_eq**3 - \
            8.5e-3*D_eq**4
    else:
        return 1.065 - 6.25e-2*D_eq - 3.99e-3*D_eq**2 + 7.66e-4*D_eq**3 - \
            4.095e-5*D_eq**4 


if __name__ == '__main__':
    unittest.main()
