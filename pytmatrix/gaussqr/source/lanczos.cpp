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

gaussqr_result lanczos_tridiagonalize( const integer_t n , const real_t *x_in , const real_t *w_in , real_t *a , real_t *b )
/*
	[a,b] = lanczos_tridiagonalize(x,w)
	
	The lanczos method for tridiagonalization.
 
	n := length of a, b, x, and w
 	x,w := vectors of absciscae and weights
	a,b := the computed polynomial recursion coefficients
	
	Note that the first element of 'b' will be equal
	to sum(w) and can be used as a normalization constant
	for generated quadrature rules.
*/	
{
	if ( n < 3 || x_in == 0 || w_in == 0 || a == 0 || b == 0 )
		return gaussqr_illegal_argument;
	
	Vector x; Assign(x,n,x_in);
	Vector w; Assign(w,n,w_in);
	
	Vector p0(x);
	Vector p1(n,0.0);
	p1[0] = w[0];
	
	for ( integer_t i = 0; i < (n-1); i++ ) {
		real_t pi = w[i+1];
		real_t gamma = 1.0;
		real_t sigma = 0.0;
		real_t t = 0.0;
		for ( integer_t k = 0; k <= (i+1); k++ ) {
			real_t rho = p1[k] + pi;
			real_t zeta = gamma * rho;
			real_t old_sigma = sigma;
			 if ( rho <= 0.0 ) {
				gamma = 1.0;
				sigma = 0.0;
			} else {
				gamma = p1[k] / rho;
				sigma = pi / rho;
		 	}
		 	real_t tk = sigma * ( p0[k] - x[i+1] ) - gamma * t;
		 	p0[k] -= ( tk - t);
		 	t = tk;
		 	if ( sigma <= 0.0 ) {
				pi = old_sigma * p1[k];
		 	} else {
				pi = t * t / sigma;
		 }
		 p1[k] = zeta;
		}
	 }

	Assign(a,p0);
	Assign(b,p1);

	return gaussqr_success;
}
