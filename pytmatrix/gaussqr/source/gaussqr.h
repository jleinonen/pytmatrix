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
#ifndef GAUSSQR_H_INCLUDED
#define GAUSSQR_H_INCLUDED

/*
 * Note that the C++ modules here can safely be compiled
 * to NOT use exception handling or run-time type information.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "util/precision.h"
	
extern const integer_t gaussqr_version_major;
extern const integer_t gaussqr_version_minor;
	
typedef int gaussqr_result;
typedef int domain_type;
typedef int distribution_type;
	
enum {
	/* function return values */
	gaussqr_success = 0,
	gaussqr_illegal_argument = 1,
	gaussqr_calculation_failed = -1,
	gaussqr_memory_allocation_error = -2
};

enum {
	/* for the 'map_fejer2_domain' function */
	domain_finite = 0,
	domain_left_infinite = 1,
	domain_right_infinite = 2,
	domain_infinite = 3
};

enum {
	/* for the 'standard_distribution_rcoeffs' function */
	distribution_normal = 0,
	distribution_gamma = 1,
	distribution_log_normal = 2,
	distribution_students_t = 3,
	distribution_inverse_gamma = 4,
	distribution_beta = 5,
	distribution_fishers_f = 6
};

/*
 * WARNING: To calculate n recursion coefficients, at least the first 2*n moments of the target density function must exist!
 *
 * Each function below is documented in the corresponding implementation file.
 *
 * For an quick example of how to use these functions for your own distribution, read the paper and take
 * a look at the 'test_distribution' function in the 'test.cpp' file included in this distribution.
 *
 */

gaussqr_result fejer2_abscissae( const integer_t n , real_t *z , real_t *q );

gaussqr_result map_fejer2_domain( const real_t a, const real_t b , const domain_type type , const integer_t n , const real_t *x , real_t *y , real_t *dy );

gaussqr_result lanczos_tridiagonalize( const integer_t n , const real_t *x , const real_t *w , real_t *a , real_t *b );

gaussqr_result standard_distribution_rcoeffs( const distribution_type type , const integer_t n , real_t *a , real_t *b , const real_t *p );

gaussqr_result standard_distribution_pdf( const distribution_type type , const real_t x , real_t *y , const real_t *p );

gaussqr_result gaussqr_from_rcoeffs( const integer_t n , const real_t *a , const real_t *b , real_t *x , real_t *w );

gaussqr_result relative_error( const integer_t n , const real_t *w0 , const real_t *w1 , real_t *err );

#ifdef __cplusplus
}
#endif
	
#endif /* ! GAUSSQR_H_INCLUDED */
    
