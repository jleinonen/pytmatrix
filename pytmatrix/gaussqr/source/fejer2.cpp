/*
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
#include "util/linalg.hpp"
#include "fft/fftpack.h"

#include <new>
using std::nothrow;

#include <cmath>
using std::cos;

gaussqr_result fejer2_abscissae( const integer_t n , real_t *z , real_t *q )
/*
	function [z,q] = fejer2(n)

	Weights q and nodes z for the n-point Fejer type-2 quadrature rules,
	calculated via the inverse (fast) real_t discrete fourier transform.

	The knots are the same knots as the Clenshaw-Curtis knots, implying that
	by doubling the number of evalulation points, you can re-use the function
	evaluations at the old points, since the new set of knots will contain the
	old set of knots. In other words, the knots for an n-point fejer2 rule
	will be contained in the (2*n+1)-point fejer2 rule.
 
	Although capable of computing knots and weights for arbitrary 'n', the
	computation will be much faster and efficient if (n+1) has factors only
	from the set from the integer_t set {2,3,4,5}.
	
	Therefore one recommended sequence of n is {3,7,15,31,63,127,255,511,1023,...}.
	Three (or five) times this set would also be acceptable.
  
*/
{
	if ( n < 3 || z == 0 || q == 0 )
		return gaussqr_illegal_argument;
	
	gaussqr_result rv = gaussqr_success;

	integer_t n_fft = n + 1;
	const real_t pi = 3.1415926535897932384626433832795028841971693993751058209749445923;

	// allocate memory
	real_t *v = new(nothrow) real_t[n_fft];
	real_t *work = new(nothrow) real_t[2*n_fft];
	integer_t *ifac = new(nothrow) integer_t[sizeof(integer_t)*8];
	if ( v == 0 || work == 0 || ifac == 0 ) {
		rv = gaussqr_memory_allocation_error;
		goto done;
	}
		
	// calculate the knots
	for ( integer_t k = 1; k < n_fft; k++ ) {
		z[k-1] = -cos((k*pi)/n_fft);
	}
	
	// calculate the transformed weights
	v[0] = 2.0;
	for ( integer_t k = 1; k < n_fft; k++ ) v[k] = 0.0;
	for ( integer_t k = 1; k < n_fft/2; k++ ) {
		v[2*k-1] = 2.0/(1.0-4.0*(k*k));
	}
	v[n_fft-1-(n_fft&1)] = (n_fft-3.0)/(2.0*(n_fft/2)-1.0)-1.0;
	
    // take the fourier transform, and scale
	rffti(&n_fft,work,ifac);
	rfftb(&n_fft,v,work,ifac);
	for ( integer_t i = 0; i < n_fft; i++ ) {
		v[i] /= n_fft;
	}
	
	// transfer u to q, removing first (zero) element
	for ( integer_t i = 1; i < n_fft; i++ ) {
		q[i-1] = v[i];
	}

done:
	delete[](ifac);
    delete[](work);
	delete[](v);
	return(rv);
}


gaussqr_result map_fejer2_domain( const real_t a, const real_t b , const domain_type type , const integer_t n , const real_t *x , real_t *y , real_t *dy )
/*
	Given an interval [a,b] and a domain_type, this subroutine maps the interval [-1,1] to [a,b],
	an interval that may be closed, left-infinite, right-infinite, or totally infinite.
 
	The input points are x, and should be enclosed in the open interval (-1:1). This restriction is NOT checked for.
 
	On output, the points y are the mapped points to the new interval, and the dy array gives the derivative of the transformation.
 
	Note that the change of integration variables is $\int_{-1}^{+1} f(x) \, dx = \int_{a}^{b} f(y(x)) dy \, dx$, and this is
	used in the quadrature approximations to integration.
 
*/
{
	if ( a >= b || n < 3 || x == 0 || y == 0 || dy == 0 )
		return(gaussqr_illegal_argument);
	
	switch(type) {
		
		case domain_finite :
			for ( integer_t i = 0; i < n; i++ ) {
				const real_t bpa = b + a;
				const real_t bma = b - a;
				y[i] = 0.5 * ( x[i]*bma + bpa );
				dy[i] = 0.5 * bma;
			}
			break;
	
		case domain_left_infinite :
			for ( integer_t i = 0; i < n; i++ ) {
				const real_t x_m1 = (1.0-x[i]);
				const real_t x_p1 = (1.0+x[i]);
				y[i] = x_m1/x_p1 + b;
				dy[i] = 2.0/(x_p1*x_p1);
			}
			break;
			
		case domain_right_infinite :
			for ( integer_t i = 0; i < n; i++ ) {
				const real_t x_m1 = (1.0-x[i]);
				const real_t x_p1 = (1.0+x[i]);
				y[i] = x_p1/x_m1 + a;
				dy[i] = 2.0/(x_m1*x_m1);
			}
			break;
			
		case domain_infinite :
			for ( integer_t i = 0; i < n; i++ ) {
				const real_t x_m1 = (1.0-x[i]);
				const real_t x_p1 = (1.0+x[i]);
				y[i] = (2.0*x[i])/(x_m1*x_p1);
				dy[i] = (2.0*(1.0+(x[i]*x[i])))/(x_m1*x_m1*x_p1*x_p1);
			}
			break;
			
		default :
			return(gaussqr_illegal_argument);
		
	}
	
	return(gaussqr_success);
}

