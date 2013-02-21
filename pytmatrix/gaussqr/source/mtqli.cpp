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

#include <cmath>
using std::sqrt;
using std::fabs;

static void heapsort_eig( Vector &v , Matrix &m );

gaussqr_result gaussqr_from_rcoeffs( const integer_t n , const real_t *a , const real_t *b , real_t *x , real_t *w )
/*
	[x,w] = gaussqr_from_rcoeffs(a,b)
	
	Calculates the sorted eigenvlues and normalized eigenvectors
	of a (modified) symmetric tri-diagonal matrix, via the implicit-
	shift QL algorithm. The 'modified' comes from the fact that the
	off-diagonals are mapped to their square-roots before decomposition.
	
	Then, the eigendecomposition is used to compute the gaussian quadrature
	rule for the given recursion coefficients. The first element of 'b' is
	used to normalize the quadrature scheme, so set the first element
	of 'b' to be equal to the desired $\integer_t w(x) \, dx$ on input.
 
	The method is based on the SLATEC subroutine IMTQLV.
	
	n := length of a, b, x, and w
 
	On input:
	   a := a vector representing the matrix diagonal
	   b := a vector whose elements are the square of the matrix
			sub- & super-diagonal; it has the same length as a,
			and the first element is not part of the matrix
	
	On output:
	    x := gaussian quadrature abscissae
	    w := gaussian quadrature weights
*/
{
	if ( n < 3 || a == 0 || b == 0 || x == 0 || w == 0 )
		return gaussqr_illegal_argument;
	
	Vector d; Assign(d,n,a);
	Vector e(n,0.0);
	
	// note that this routine will work perfectly well
	// for arbitrary off-diagonals if we omit the sqrt
	for ( integer_t i = 0; i < n-1; i++ ) {
		e[i] = sqrt(b[i+1]);
	}
	
	const integer_t TQLI_MAX_ITER = 32;
	
	real_t bb, c, dd, f, g, p, r, s;
	
	Matrix z; AssignIdentity(z,n);
	
	for ( integer_t l = 0; l < n; l++ ) {
		integer_t m = 0, iter = 0;
		do {
			for ( m = l; m < n-1; m++ ) {
				dd = fabs( d[m] ) + fabs( d[m+1] );
				if ( fabs( e[m] ) + dd == dd )
					break;
			}
			if ( m != l ) {
				if ( iter++ >= TQLI_MAX_ITER )
					return gaussqr_calculation_failed;
				g = ( d[l+1] - d[l] ) / ( 2.0 * e[l] );
				r = sqrt( ( g * g ) + 1.0 );
				g = d[m] - d[l] + e[l] / ( g + copysign(r,g) );
				s = c = 1.0;
				p = 0.0;
				for ( integer_t i = m-1; i >= l; i-- ) {
					f = s * e[i];
					bb = c * e[i];
					if ( fabs( f ) >= fabs( g ) ) {
						c = g / f;
						r = sqrt( ( c * c ) + 1.0 );
						e[i+1] = f * r;
						s = 1.0 / r;
						c *= s;
					} else {
						s = f / g;
						r = sqrt( ( s * s ) + 1.0 );
						e[i+1] = g * r;
						c = 1.0 / r;
						s *= c;
					}
					g = d[i+1] - p;
					r = ( d[i] - g ) * s + 2.0 * c * bb;
					p = s * r;
					d[i+1] = g + p;
					g = c * r - bb;
					for ( integer_t k = 0; k < n; k++ ) {
						f = z[k][i+1];
						z[k][i+1] = s * z[k][i] + c * f;
						z[k][i]   = c * z[k][i] - s * f;
					}
				}
				d[l] = d[l] - p;
				e[l] = g;
				e[m] = 0.0;
			}
		} while ( m != l );
	}
	
	heapsort_eig(d,z); // it is not strictly necessary to sort the eigenvalues
	
	for ( integer_t i = 0; i < n; i++ ) {
		x[i] = d[i]; // the abscissae are just the eigenvalues
		w[i] = b[0] * ( z[0][i] * z[0][i] ); // first element of each column eigenvector
	}
	
	return gaussqr_success;
}


static void heapsort_eig( Vector &v , Matrix &m )
{
	// given a vector of eigenvalues and a matrix of
	// corresponding eigenvector columns, sort the 
	// eigenvalues (and vectors) in ascending order
	// using the heapsort algorithm

	// ASSUMES: length(v) == rows(m) == columns(m)
	//          and that this size is > 0,
	//          and the first index of the matrix
	//          denotes the row
	
	const integer_t N = v.size();

	integer_t n = N;
	integer_t i = n/2;
	while ( 1 ) {
		real_t t;
		Vector tv(N,0.0);

		if ( i > 0 ) {
			i--;
			t = v[i]; for ( integer_t k=0; k < N; k++ ) tv[k] = m[k][i];
		} else {
			n--;
			if ( n == 0 ) return;
			t = v[n]; for ( integer_t k=0; k < N; k++ ) tv[k] = m[k][n];
			v[n] = v[0]; for ( integer_t k=0; k < N; k++ ) m[k][n] = m[k][0];
		}

		integer_t j = i;
		integer_t w = i * 2 + 1;

		while ( w < n ) {
			if ( w+1 < n && v[w+1] > v[w] ) w++;
			if ( v[w] > t ) {
				v[j] = v[w]; for ( integer_t k=0; k < N; k++ ) m[k][j] = m[k][w];
				j = w;
				w = j * 2 + 1;
			} else {
				break;
			}
		}
		v[j] = t; for ( integer_t k=0; k < N; k++ ) m[k][j] = tv[k];
	}

}
