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
#ifndef LINALG_HPP_INCLUDED
#define LINALG_HPP_INCLUDED

#include "precision.h"

/*
 
 Vectors are indexed starting from zero.
 
 Matrices are also indexed from zero. Further, the first
 index denotes the row, the second the column.
 
 You can change the typedefs and inline functions here
 to use any vector/matrix class you wish.
 
 */

#include <vector>

typedef std::vector<real_t> Vector;
typedef std::vector<Vector> Matrix;

// v <- a
inline void Assign( Vector &v , const integer_t n , const real_t *a ) {
	v.resize(n);
	for ( integer_t i = 0; i < n; i++ )
		v[i] = a[i];
}

// a <- v
inline void Assign( real_t *a , const Vector &v ) {
	for ( unsigned i = 0; i < v.size(); i++ )
		a[i] = v[i];
}

// identity matrix
inline void AssignIdentity( Matrix &m , const integer_t n ) {
	m.resize(n);
	for ( integer_t i = 0; i < n; i++ ) {
		m[i].resize(n);
		for ( integer_t j = 0; j < n; j ++ ) {
			m[i][j]  = 0.0;
		}
		m[i][i] = 1.0;
	}
}

#endif /* ! LINALG_HPP_INCLUDED */
