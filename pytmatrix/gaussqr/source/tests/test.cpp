/*cd 
 * Copyright (c) 2005, Andrew Fernandes (andrew@fernandes.org);
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 * - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 
 * - Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * - Neither the name of the North Carolina State University nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include "gaussqr.h"

#include <algorithm>
using std::min;
using std::max;

#include <iostream>
using std::cout;
using std::endl;

#include <cmath>
using std::fabs;

#include <float.h>

// ========================================================
// Helper Functions

static real_t pow_ru( real_t x , unsigned n )
{
	real_t y = 1.0;
	
	while ( n > 0 ) {
		if ( n & 1 ) y *= x;
		if ( n >>= 1 ) x *= x;
	}
	
	return(y);
}

static real_t p0 ( const real_t &x ) { return( (0.74e2 + (0.6e1 + (-0.92e2 + (0.75e2 + 0.23e2 * pow_ru(x,3)) * pow_ru(x,4)) * x) * x) * x * x ); } // degree 11
static real_t p1 ( const real_t &x ) { return( (-0.8e1 + (-0.61e2 + (0.10e2 + (-0.23e2 + 0.98e2 * pow_ru(x,8)) * pow_ru(x,3)) * x) * pow_ru(x,3)) * pow_ru(x,6) ); } // degree 21
static real_t p2 ( const real_t &x ) { return( (0.68e2 + (0.91e2 + (-0.81e2 + (0.40e2 - 0.47e2 * pow_ru(x,4)) * pow_ru(x,7)) * x) * pow_ru(x,4)) * pow_ru(x,3) ); } // degree 19
static real_t p3 ( const real_t &x ) { return( (-0.15e2 + (-0.27e2 + (0.30e2 + (0.16e2 - 0.28e2 * x) * x * x) * pow_ru(x,5)) * pow_ru(x,7)) * pow_ru(x,5) ); } // degree 20
static real_t p4 ( const real_t &x ) { return( (-0.91e2 + (0.92e2 + (0.43e2 + (-0.90e2 + 0.47e2 * pow_ru(x,3)) * x * x) * pow_ru(x,3)) * x) * pow_ru(x,9) ); } // degree 18

real_t absolute_error( const integer_t n , const real_t *estimated , const real_t *actual )
{
	if ( n < 1 || estimated == 0 || actual == 0 )
		return(FLT_MAX);
	
	real_t err = 0.0;
	for ( integer_t i = 0; i < n; i++ ) {
		if ( actual[i] == 0.0 ) return(FLT_MAX);
		err = max( err , std::fabs( (estimated[i]-actual[i])/actual[i] ) );
	}
	
	return(err);
}

static real_t abs_err( const real_t &approx , const real_t &exact )
{
	return ( fabs((approx-exact)/exact) );
}



// ========================================================
// Test Functions

gaussqr_result test_fejer2() {

	gaussqr_result gqr = gaussqr_success;

	const integer_t np = 5;
	real_t (*p[np])( const real_t &x ) = { p0, p1 , p2 , p3 , p4 };
	real_t integral_p[np] = { 0.146e3/0.5e1 , -0.36e2/0.77e2 , -18.0 , -0.266e3/0.39e2 , 0.4530e4/0.209e3 };
	
	const integer_t min_fejer2_degree = 5 , max_fejer2_degree = 13;
	
	cout << "n_abscissae poly approx exact %abs_err" << endl << endl;
	
	for ( integer_t m = 0; m < np; m++ ) {
		for ( integer_t n = min_fejer2_degree; n < max_fejer2_degree; n++ ) {
			
			real_t z[max_fejer2_degree], q[max_fejer2_degree], f[max_fejer2_degree];
			
			gaussqr_result gqr = fejer2_abscissae(n,z,q);
			if ( gqr != gaussqr_success ) goto done;
			
			for ( integer_t i = 0; i < n; i++ ) {
				f[i] = p[m](z[i]);
			}

			real_t sum = 0.0;
			for ( integer_t i = 0; i < n; i++ ) {
				sum += q[i] * f[i];
			}
			
			cout << n << ' ' << m << ' ' << sum << ' ' << integral_p[m] << ' ' << abs_err(sum,integral_p[m])*100.0 << '%' << endl;
			
		}
		cout << endl;
	}

done:
	return(gqr);
}


gaussqr_result test_distribution( distribution_type distribution, const real_t left , const real_t right , const domain_type domain , integer_t n_rcoefs , const real_t *p ) {
	
	gaussqr_result gqr = gaussqr_success;
	
	const integer_t n = 1023;
	real_t z[n], q[n], x[n], dx[n], f[n], w[n], a0[n], b0[n], a1[n], b1[n];
	real_t err = FLT_MAX;
	
	gqr = fejer2_abscissae(n,z,q);
	if ( gqr != gaussqr_success ) goto done;
	
	gqr = map_fejer2_domain(left,right,domain,n,z,x,dx);
	if ( gqr != gaussqr_success ) goto done;
	
	for ( integer_t i = 0; i < n; i++ ) {
		gqr = standard_distribution_pdf(distribution,x[i],&f[i],p);
		if ( gqr != gaussqr_success ) goto done;
		w[i] = f[i] * q[i] * dx[i];
	}
	
	// calculate the recursion coefficients a and b from the quadrature scheme
	gqr = lanczos_tridiagonalize(n,x,w,a0,b0);
	if ( gqr != gaussqr_success ) goto done;
	
	// calculate the recursion coefficients a and b from the known recurrences
	gqr = standard_distribution_rcoeffs(distribution,n,a1,b1,p);
	if ( gqr != gaussqr_success ) goto done;
	
	// calculate abscissa and weights
	gqr = gaussqr_from_rcoeffs(n_rcoefs,a0,b0,x,w);
	if ( gqr != gaussqr_success ) goto done;
		
	// compare recurrence coefficients, theoretical vs derived
	cout << "[i]\ta0\ta1\tb0\tb1" << endl << endl;
	for ( integer_t i = 0; i < min(n,n_rcoefs); i++ ) {
		cout << '[' << i << ']' << '\t' << a0[i] << '\t' << a1[i] << '\t' << b0[i] << '\t' << b1[i] << '\t' << endl;
	}
	cout << endl;	
	err = absolute_error(n_rcoefs,b0,b1);
	if ( gqr != gaussqr_success ) goto done;
	cout << "max(b_err) => " << err*100.0 << '%' << endl;
	if ( err > 0.01 ) {
		cout << "WARNING: large error detected; check the first few" << endl;
		cout << "         entries for convergence..." << endl;
	}
	cout << endl;
	
	// print abscissa and weights
	cout << "[i]\tabscissa\tweight" << endl << endl;
	for ( integer_t i = 0; i < min(n,n_rcoefs); i++ ) {
		cout << '[' << i << ']' << '\t' << x[i] << '\t' << w[i] << endl;
	}
	cout << endl;
	
done:
	return(gqr);
}

// ========================================================
// Main Function

int main( int argc , char *argv[] )
{
	gaussqr_result gqr = gaussqr_success;
	real_t parameter[2] = { 0.0 , 0.0 };
	
	cout << "============= Polynomial Integration Tests" << endl << endl;

	gqr = test_fejer2();
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	cout << "============= Normal Coefficient Tests" << endl << endl;
	
	parameter[0] = 0.0;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_normal,-FLT_MAX,FLT_MAX,domain_infinite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 1.0;
	parameter[1] = 0.5;
	gqr = test_distribution(distribution_normal,-FLT_MAX,FLT_MAX,domain_infinite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;

	cout << "============= Gamma Coefficient Tests" << endl << endl;
	
	parameter[0] = 1.0;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_gamma,0,FLT_MAX,domain_right_infinite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 2.0;
	parameter[1] = 2.0;
	gqr = test_distribution(distribution_gamma,0,FLT_MAX,domain_right_infinite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;

	cout << "============= Log-Normal Coefficient Tests" << endl << endl;
	
	parameter[0] = 1.0;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_log_normal,0,FLT_MAX,domain_right_infinite,6,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 2.0;
	parameter[1] = 0.5;
	gqr = test_distribution(distribution_log_normal,0,FLT_MAX,domain_right_infinite,15,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;

	cout << "============= Student's T Coefficient Tests" << endl << endl;
	
	parameter[0] = 18.0;
	gqr = test_distribution(distribution_students_t,-FLT_MAX,FLT_MAX,domain_infinite,9,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 25.0;
	gqr = test_distribution(distribution_students_t,-FLT_MAX,FLT_MAX,domain_infinite,13,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;

	cout << "============= Inverse-Gamma Coefficient Tests" << endl << endl;
	
	parameter[0] = 12.0;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_inverse_gamma,0,FLT_MAX,domain_right_infinite,6,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 25.0;
	parameter[1] = 2.0;
	gqr = test_distribution(distribution_inverse_gamma,0,FLT_MAX,domain_right_infinite,12,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	cout << "============= Beta Coefficient Tests" << endl << endl;
	
	parameter[0] = 1.0;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_beta,0,1,domain_finite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 1.0;
	parameter[1] = 2.25;
	gqr = test_distribution(distribution_beta,0,1,domain_finite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 2.25;
	parameter[1] = 1.0;
	gqr = test_distribution(distribution_beta,0,1,domain_finite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 1.5;
	parameter[1] = 3.5;
	gqr = test_distribution(distribution_beta,0,1,domain_finite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 2.0;
	parameter[1] = 2.0;
	gqr = test_distribution(distribution_beta,0,1,domain_finite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	cout << "============= Fisher's F Coefficient Tests" << endl << endl;
	
	parameter[0] = 37.0;
	parameter[1] = 121.0;
	gqr = test_distribution(distribution_fishers_f,0,FLT_MAX,domain_right_infinite,16,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
	parameter[0] = 25.0;
	parameter[1] = 32.0;
	gqr = test_distribution(distribution_fishers_f,0,FLT_MAX,domain_right_infinite,8,parameter);
	cout << endl << endl;
	if ( gqr != gaussqr_success ) goto done;
	
done:
	if ( gqr != gaussqr_success ) {
		cout << "FAILED: code " << static_cast<int>(gqr) << endl;
	}
	return static_cast<int>(gqr);
}
