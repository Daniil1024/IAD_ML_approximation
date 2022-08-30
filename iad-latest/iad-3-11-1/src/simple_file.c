#include <Python.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <ctype.h>

#include "ad_globl.h"
#include "ad_prime.h"
#include "iad_type.h"
#include "iad_pub.h"
#include "iad_io.h"
#include "iad_calc.h"
#include "iad_util.h"
#include "mygetopt.h"
#include "version.h"
#include "mc_lost.h"
#include "ad_frsnl.h"

double return_to_python(double a, double b)
{
  return Cos_Critical_Angle(a, b);
}

static PyObject *py_primecounter(PyObject *self, PyObject *args)
{
  double n_frm, n_til = 0;
  if (!PyArg_ParseTuple(args, "dd", &n_frm, &n_til))
  {
    return NULL;
  }
  // fprintf (stderr, "%f\n", n_frm);
  double result = return_to_python(n_frm, n_til);

  return PyFloat_FromDouble(result);
}

static PyMethodDef CountingMethods[] = {
    {"primecounter", py_primecounter, METH_VARARGS, "Function for counting primes in a range in c"},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef fastcountmodule = {
    PyModuleDef_HEAD_INIT,
    "Fastcount",
    "C library for counting fast",
    -1,
    CountingMethods};
PyMODINIT_FUNC PyInit_Fastcount(void)
{
  return PyModule_Create(&fastcountmodule);
};
