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

#include <cmath>
#include <algorithm>

const integer_t gaussqr_version_major = 1;
const integer_t gaussqr_version_minor = 1;

gaussqr_result relative_error( const integer_t n , const real_t *w0 , const real_t *w1 , real_t *err )
/*
	Returns the maximum relative error difference between two vectors, w0 and w1.
 
*/
{
	if ( n < 3 || w0 == 0 || w1 == 0 || err == 0 )
		return(gaussqr_illegal_argument);

	*err = 0.0;
	for ( integer_t i = 0; i < n; i++ ) {
		*err = std::max( *err , std::fabs( (w0[i]-w1[i])/(0.5*(w0[i]+w1[i])) ) );
	}
	
	return(gaussqr_success);
}

namespace {
	
	// raise a real_t number to an integer_t power
	inline real_t pow_ri( real_t x , integer_t n )
	{
		real_t y = 1.0;
		
		if ( n < 0 ) {
			x = 1.0/x;
			n = -n;
		}
		
		while ( n > 0 ) {
			if ( n & 1 ) y *= x;
			if ( n >>= 1 ) x *= x;
		}
		
		return(y);
	}

	inline real_t squared( const real_t &x )
	// square the argument
	{
		return( x * x );
	}
	
}

gaussqr_result standard_distribution_rcoeffs( const distribution_type type , const integer_t n , real_t *a , real_t *b , const real_t *p )
/*
	Given one of the enumerated distriution types, and the number of desired recursion coefficients n,
	this routine returns the desired recursion coefficients a and b. The parameters of the distribution
	are passed in the array p, whose length is implied by the distribution type. See the cases below
	for the number and meaning of the parameters (they are quite standard).
 
	WARNING: No error checking is done on the input parameters p for validity!
 
*/
{
	if ( n < 3 || a == 0 || b == 0 )
		return(gaussqr_illegal_argument);
	
	using std::exp;
	
	real_t lambda = 0.0;
	
	switch(type) {
		
		case distribution_normal : {
			const real_t &mu = p[0];
			const real_t sigma2 = squared(p[1]);
			lambda = 1.0;
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = mu;
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = i * sigma2;
			}
		} break;
			
		case distribution_gamma : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			lambda = beta;
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = alpha + 2*i;
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = i*(alpha+i-1);
			}
		} break;
			
		case distribution_log_normal : {
			const real_t &m = p[0];
			const real_t s2 = squared(p[1]);
			lambda = exp(m);
			const real_t zeta = exp(s2);
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = pow(zeta,0.5*(2*i-1))*(pow_ri(zeta,i)*(zeta+1.0)-1.0);
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = pow_ri(zeta,3*i-2)*(pow_ri(zeta,i)-1.0);
			}
		} break;
			
		case distribution_students_t : {
			const real_t &nu = p[0];
			lambda = 1.0;
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = 0.0;
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = i*nu*(nu-i+1)/((nu-2*i)*(nu-2*i+2));
			}
		} break;
			
		case distribution_inverse_gamma : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			lambda = beta;
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = (alpha+1.0)/((alpha-2*i+1)*(alpha-2*i-1));
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = i*(alpha-i+1)/((alpha-2*i)*squared(alpha-2*i+1)*(alpha-2*i+2));
			}
		} break;
			
		case distribution_beta : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			const real_t gamma = alpha + beta;
			lambda = 1.0;
			for ( integer_t i = 0; i < n; i++ ) {
				if ( i == 0 && alpha == 1.0 && beta == 1.0 ) {
					a[i] = 0.5;
				} else {
					a[i] = (alpha*gamma+(2*i-2)*alpha+2*i*beta+i*(2*i-2))/((gamma+2*i)*(gamma+2*i-2));
				}
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = (i*(gamma+i-2)*(alpha+i-1)*(beta+i-1))/((gamma+2*i-1)*squared(gamma+2*i-2)*(gamma+2*i-3));
			}
		} break;
			
		case distribution_fishers_f : {
			const real_t &nu1 = p[0];
			const real_t &nu2 = p[1];
			lambda = nu2/nu1;
			for ( integer_t i = 0; i < n; i++ ) {
				a[i] = (nu1*nu2+2*nu1+4*i*nu2-8*i*i)/((nu2-4*i-2)*(nu2-4*i+2));
			}
			for ( integer_t i = 1; i < n; i++ ) {
				b[i] = (2*i*(nu1+2*i-2)*(nu2-2*i+2)*(nu1+nu2-2*i))/((nu2-4*i)*squared(nu2-4*i+2)*(nu2-4*i+4));
			}
		} break;
			
		default :
			return(gaussqr_illegal_argument);
			
	}
	
	const real_t lambda2 = squared(lambda);
	for ( integer_t i = 0; i < n; i++ ) {
		a[i] *= lambda;
		b[i] *= lambda2;
	}
	b[0] = 1.0;
	
	return(gaussqr_success);
}

namespace {	

	// Define a polymorphic gamma function, since C++ does not provide one.
	// Note that the 'tgamma' variants are C99 standard and are probably
	// called something else on Windows.
	inline       float gamma_function( const      float  &x ) { return tgammaf(x); }
	inline      double gamma_function( const      double &x ) { return tgamma (x); }
	inline long double gamma_function( const long double &x ) { return tgammal(x); }

	// Unfortunately, no standard library defines the beta function; we do so here.
	inline real_t beta_function( const real_t &alpha , const real_t &beta ) {
		return( gamma_function(alpha)*gamma_function(beta)/gamma_function(alpha+beta) );
	}

}

gaussqr_result standard_distribution_pdf( const distribution_type type , const real_t x , real_t *y , const real_t *p )
/*
	Given one of the enumerated distriution types and an abscissa x,
	this routine returns the ordinate y of the desired density function. The parameters of the distribution
	are passed in the array p, whose length is implied by the distribution type. See the cases below
	for the number and meaning of the parameters (they are quite standard).
 
	WARNING: No error checking is done on the input parameters p or abscissa x for validity!
 
 */
{
	if ( y == 0 )
		return(gaussqr_illegal_argument);
	
	using std::exp;
	using std::pow;
	using std::log;
	using std::sqrt;
	
	switch(type) {
		
		case distribution_normal : {
			const real_t &mu = p[0];
			const real_t sigma2 = squared(p[1]);
			const real_t z = x - mu;
			*y = exp( -0.5 * squared(z) / sigma2 ) / sqrt( 2.0 * sigma2 * M_PI );
		} break;
			
		case distribution_gamma : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			*y = pow(x,alpha-1.0) * exp(-x/beta) / ( gamma_function(alpha) * pow(beta,alpha) );
		} break;			
			
		case distribution_log_normal : {
			const real_t &m = p[0];
			const real_t s2 = squared(p[1]);
			const real_t z = log(x) - m;
			*y = exp( -0.5 * squared(z) / s2 ) / ( x * sqrt( 2.0 * s2 * M_PI ) );
		} break;			
			
		case distribution_students_t : {
			const real_t &nu = p[0];
			const real_t tmp = 0.5*( nu + 1.0 );
			*y = gamma_function(tmp)/( sqrt(nu*M_PI) * gamma_function(0.5*nu) * pow(1.0+x*x/nu,tmp) );
		} break;			
			
		case distribution_inverse_gamma : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			*y = pow(x,-(alpha+1.0)) * pow(beta,alpha) * exp(-beta/x) / gamma_function(alpha);
		} break;			
			
		case distribution_beta : {
			const real_t &alpha = p[0];
			const real_t &beta  = p[1];
			*y = pow(x,alpha-1.0) * pow(1.0-x,beta-1.0) / beta_function(alpha,beta);
		} break;			
			
		case distribution_fishers_f : {
			const real_t &nu1 = p[0];
			const real_t &nu2 = p[1];
			const real_t nu1by2 = 0.5 * nu1;
			const real_t nu2by2 = 0.5 * nu2;
			*y = pow(nu1,nu1by2) * pow(nu2,nu2by2) * pow(x,nu1by2-1.0) / ( beta_function(nu1by2,nu2by2) * pow(nu1*x+nu2,nu1by2+nu2by2) );
		} break;			
			
		default :
			return(gaussqr_illegal_argument);
			
	}
	
	return(gaussqr_success);
}
