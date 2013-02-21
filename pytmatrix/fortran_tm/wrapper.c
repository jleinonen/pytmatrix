#include <Python.h>

typedef struct {double dr; double di;} dcplx;

/* Fortran functions defined in ampld.lp.f */
void calctmat_(double *axi, double *rat, double *lam, double *mrr, double *mri,
	       double *eps, int *np, double *ddelt, int *ndgs, int *nmax);
void calcampl_(int *nmax, double *lam, double* thet0, double *thet,
	       double *phi0, double *phi, double* alpha, double *beta,
	       dcplx **S, double **Z);

/* Convert from Fortran complex format to the Python format */
void convert_complex(dcplx *from, Py_complex *to) {
	to->real = from->dr;
	to->imag = from->di; 
}

/* Compute the T-matrix for later use at different orientations */
static PyObject *calc_tmatrix(PyObject *self, PyObject *args) {
   
   double axi, rat, lam, mrr, mri, eps, ddelt;
   int np, ndgs, nmax;
   
   int ok = PyArg_ParseTuple(args, "ddddddidi", &axi, &rat, &lam, &mrr,
			     &mri, &eps, &np, &ddelt, &ndgs);
   if (!ok)
      return NULL;
   
   calctmat_(&axi,&rat,&lam,&mrr,&mri,&eps,&np,&ddelt,&ndgs,&nmax);

   return Py_BuildValue("di", lam, nmax);
   
}

/* Compute the amplitude and phase matrices once the T-matrix has been computed */
static PyObject *get_ampl(PyObject *self, PyObject *args) {
   
   double lam, thet0, thet, phi0, phi, alpha, beta;
   int nmax;
   dcplx S[2][2];
   Py_complex S_py[2][2];
   double Z[4][4];
   
   PyArg_ParseTuple(args, "(di)dddddd", &lam, &nmax, &thet0,
		  &thet, &phi0, &phi, &alpha, &beta);
   
   calcampl_(&nmax,&lam,&thet0,&thet,&phi0,&phi,&alpha,&beta,(dcplx **)S,(double **)Z);
   
   convert_complex(&(S[0][0]), &(S_py[0][0]));
   convert_complex(&(S[1][0]), &(S_py[1][0]));
   convert_complex(&(S[0][1]), &(S_py[0][1]));
   convert_complex(&(S[1][1]), &(S_py[1][1]));
   
   /* Transpose the matrices here due to the different storage orders in C and Fortran */
   return Py_BuildValue("[[DD][DD]][[dddd][dddd][dddd][dddd]]",
			&(S_py[0][0]), &(S_py[1][0]),
			&(S_py[0][1]), &(S_py[1][1]),
			Z[0][0], Z[1][0], Z[2][0], Z[3][0], 
			Z[0][1], Z[1][1], Z[2][1], Z[3][1], 
			Z[0][2], Z[1][2], Z[2][2], Z[3][2], 
			Z[0][3], Z[1][3], Z[2][3], Z[3][3]);   
   
}

static PyMethodDef TMatrixMethods[] = {
    {"calc_tmatrix",  calc_tmatrix, METH_VARARGS,
     "Calculate the T-matrix."},
    {"get_ampl",  get_ampl, METH_VARARGS,
     "Calculate the amplitude and phase matrices."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initpytmatrix(void) {
    (void) Py_InitModule("pytmatrix", TMatrixMethods);
}
