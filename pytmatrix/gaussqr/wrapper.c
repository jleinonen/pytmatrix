#include <stdlib.h>
#include "source/gaussqr.h"
#include <Python.h>
#include <numpy/arrayobject.h>

static PyObject *get_points_and_weights(PyObject *self, PyObject *args) {

   PyObject *distfunc;
   PyArrayObject *xarr = NULL, *warr = NULL, *farr = NULL;
   double left = -1.0, right = 1.0;
   int np = 1024, nc = 5;
   double *z, *q, *x, *dx, *f, *w, *a, *b;
   npy_intp dims[1];
   int i;   
   
   /* The parameters are the weight function, the integration limits, 
      the number of function points and weights and the number of points. */
   if (!PyArg_ParseTuple(args, "O|ddii", &distfunc, &left, &right, &nc, &np) || (nc > np))
      return NULL;  

   /* Allocate everything in a big chunk */
   z = malloc(8 * np * sizeof(double));
   q = z + np;
   x = q + np;
   dx = x + np;
   f = dx + np;
   w = f + np;
   a = w + np;
   b = a + np;
   
   /* Initialize the GaussQR code for this domain*/
   fejer2_abscissae(np,z,q);
   map_fejer2_domain(left,right,domain_finite,np,z,x,dx);   
   
   /* Create an ndarray from the evaluation points call the user-defined weight function */
   dims[0] = np;
   xarr = (PyArrayObject *)PyArray_SimpleNew(1, dims, PyArray_DOUBLE);     
   memcpy(xarr->data, x, np*sizeof(double));
   farr = (PyArrayObject *)PyArray_ContiguousFromObject(
      PyEval_CallFunction(distfunc, "(O)", (PyObject *)xarr), PyArray_DOUBLE, 1, 1);      
   memcpy(f, farr->data, np*sizeof(double)); 
   
   /* Solve the tridiagonalization problem with GaussQR to get the weights */
   for (i = 0; i < np; i++)
      w[i] = f[i] * q[i] * dx[i];
   lanczos_tridiagonalize(np,x,w,a,b);
   gaussqr_from_rcoeffs(nc,a,b,x,w);
   
   dims[0] = nc;
   xarr = (PyArrayObject *)PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
   warr = (PyArrayObject *)PyArray_SimpleNew(1, dims, PyArray_DOUBLE);
   memcpy(xarr->data, x, nc*sizeof(double));
   memcpy(warr->data, w, nc*sizeof(double));
   
   free(z);
   
   return Py_BuildValue("(NN)", xarr, warr);
   
}

static PyMethodDef PyGaussQRMethods[] = {
    {"get_points_and_weights",  get_points_and_weights, METH_VARARGS,
     "Calculate the quadrature points and weights."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initpygaussqr(void) {
    (void) Py_InitModule("pygaussqr", PyGaussQRMethods);
    import_array();
}
