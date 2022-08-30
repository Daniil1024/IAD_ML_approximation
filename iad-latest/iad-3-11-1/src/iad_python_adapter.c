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

static void initialise_experiment(struct measure_type *m, struct invert_type *r)
{
    Initialize_Measure(m);
    Initialize_Result(*m, r);
}

static void
my_print_results_header(FILE *fp)
{
    fprintf(fp,
            "#     \tMeasured \t   M_R   \tMeasured \t   M_T   \tEstimated\tEstimated\tEstimated");
    if (Debug(DEBUG_LOST_LIGHT))
        fprintf(fp,
                "\t  Lost   \t  Lost   \t  Lost   \t  Lost   \t   MC    \t   IAD   \t  Error  ");
    fprintf(fp, "\n");

    fprintf(fp,
            "##wave\t   M_R   \t   fit   \t   M_T   \t   fit   \t  mu_a   \t  mu_s'  \t    g    ");
    if (Debug(DEBUG_LOST_LIGHT))
        fprintf(fp,
                "\t   UR1   \t   URU   \t   UT1   \t   UTU   \t    #    \t    #    \t  State  ");
    fprintf(fp, "\n");

    fprintf(fp,
            "# [nm]\t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  1/mm   \t  1/mm   \t  [---]  ");
    if (Debug(DEBUG_LOST_LIGHT))
        fprintf(fp,
                "\t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  \t  [---]  ");
    fprintf(fp, "\n");
}

void my_print_optical_property_result(FILE *fp,
                                      struct measure_type m,
                                      struct invert_type r,
                                      double LR,
                                      double LT,
                                      double mu_a,
                                      double mu_sp, int mc_iter, int line)
{
    if (m.lambda != 0)
        fprintf(fp, "%6.1f\t", m.lambda);
    else
        fprintf(fp, "%6d\t", line);

    if (mu_a > 200)
        mu_a = 199.9999;
    if (mu_sp > 1000)
        mu_sp = 999.9999;

    fprintf(fp, "%0.3e\t%0.3e\t", m.m_r, LR);
    fprintf(fp, "%0.3e\t%0.3e\t", m.m_t, LT);
    fprintf(fp, "%0.3e\t", mu_a);
    fprintf(fp, "%0.3e\t", mu_sp);
    fprintf(fp, "%0.3e\t", r.g);

    if (Debug(DEBUG_LOST_LIGHT))
    {
        fprintf(fp, "%0.3e\t%0.3e\t", m.ur1_lost, m.uru_lost);
        fprintf(fp, "%0.3e\t%0.3e\t", m.ut1_lost, m.utu_lost);
        fprintf(fp, " %2d  \t", mc_iter);
        fprintf(fp, " %4d\t", r.iterations);
    }
    // fprintf(fp, "# %c \n", what_char(r.error));
    fflush(fp);
}
static void
my_Calculate_Mua_Musp(struct measure_type m,
                      struct invert_type r, double *musp, double *mua)
{
    if (r.default_b == HUGE_VAL || r.b == HUGE_VAL)
    {
        if (r.a == 0)
        {
            *musp = 0.0;
            *mua = 1.0;
            return;
        }
        *musp = 1.0 - r.g;
        *mua = (1.0 - r.a) / r.a;
        return;
    }

    *musp = r.a * r.b / m.slab_thickness * (1.0 - r.g);
    *mua = (1 - r.a) * r.b / m.slab_thickness;
}
/*static PyObject *forward_calculation(PyObject *self, PyObject *args)
{
    double mua, musp, thickness = 0;
    // fprintf (stderr, "%p\n", args);
    if (!PyArg_ParseTuple(args, "ddd", &mua, &musp, &thickness))
    {
        return NULL;
    }
    struct measure_type m;
    struct invert_type r;
    initialise_experiment(&m, &r);
    m.method = SUBSTITUTION;
    r.method.quad_pts = 12;
    double g = 0.7;
    double mus = musp / (1.0-g);
    r.default_mus = mus;
    r.default_mua = mua;
    r.a = mus / (mus+mua);
    r.b = (mus + mua) * thickness;
    r.g = g;
    r.slab.a = r.a;
    r.slab.b = r.b;
    r.slab.g = r.g;
    m.slab_thickness = thickness;
    int MC_iterations = 19;
  //  Set_Debugging(255);
    double mu_sp, mu_a, m_r, m_t;
    Calculate_MR_MT(m, r, MC_iterations, &m_r, &m_t);
//	my_Calculate_Mua_Musp(m, r, &mu_sp, &mu_a);
 //   Write_Header(m, r, -1);
 //   my_print_results_header(stdout);
  //  my_print_optical_property_result(stdout, m, r, m_r, m_t, mu_a, mu_sp, 0,
//										  0);
    return Py_BuildValue("dd", m_r, m_t);
}*/
static PyObject *forward_calculation(PyObject *self, PyObject *args)
{
    // fprintf (stderr, "%p\n", args);
    PyObject *seq, *musp_buf_obj, *thic_buf_obj;
    Py_buffer mua_buf, musp_buf, thic_buf;
    if (!PyArg_ParseTuple(args, "OOO", &seq, &musp_buf_obj, &thic_buf_obj))
    {
        return NULL;
    }
    seq = PySequence_Fast(seq, "argument must be iterable");

    /* prepare data as an array of doubles */
    int seqlen = PySequence_Fast_GET_SIZE(seq);
    double* dbar = malloc(seqlen*sizeof(double));
    if(!dbar) {
        Py_DECREF(seq);
        return PyErr_NoMemory(  );
    }
    for(int i=0; i < seqlen; i++) {
        PyObject *fitem;
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        if(!item) {
            Py_DECREF(seq);
            free(dbar);
            return 0;
        }
        fitem = PyNumber_Float(item);
        if(!fitem) {
            Py_DECREF(seq);
            free(dbar);
            PyErr_SetString(PyExc_TypeError, "all items must be numbers");
            return 0;
        }
        dbar[i] = PyFloat_AS_DOUBLE(fitem);
        Py_DECREF(fitem);
    }    

    for (int i = 0; i < seqlen; i++)
    {
        double mua = dbar[i];
        double musp = dbar[i];
        double thickness = dbar[i];
        struct measure_type m;
        struct invert_type r;
        initialise_experiment(&m, &r);
        m.method = SUBSTITUTION;
        r.method.quad_pts = 12;
        double g = 0.7;
        double mus = musp / (1.0 - g);
        r.default_mus = mus;
        r.default_mua = mua;
        r.a = mus / (mus + mua);
        r.b = (mus + mua) * thickness;
        r.g = g;
        r.slab.a = r.a;
        r.slab.b = r.b;
        r.slab.g = r.g;
        m.slab_thickness = thickness;
        int MC_iterations = 19;
        double mu_sp, mu_a, m_r, m_t;
        Calculate_MR_MT(m, r, 0, &m_r, &m_t);
        fprintf(stdout, "%f\n", mua);
    }
    return Py_BuildValue("dd", 0, 0);
}

static PyMethodDef CountingMethods[] = {
    {"forward_calculation", forward_calculation, METH_VARARGS, "Description"},
    {NULL, NULL, 0, NULL}};
static struct PyModuleDef iad = {
    PyModuleDef_HEAD_INIT,
    "IAD",
    "you know it",
    -1,
    CountingMethods};
PyMODINIT_FUNC PyInit_iad(void)
{
    return PyModule_Create(&iad);
};