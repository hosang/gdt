

/*
 * Ideas taken from:
 * http://www.cs.brown.edu/~pff/dt/
 * http://www.cs.auckland.ac.nz/~rklette/TeachAuckland.html/mm/MI30slides.pdf
 */


#include "Python.h"
#include "arrayobject.h"
#include "C_gdt.h"

#include <math.h>

#define INF 1e20


static PyMethodDef _C_gdtMethods[] = {
    {"gdt", gdt, METH_VARARGS},
    {NULL, NULL}
};

void init_C_gdt() {
    (void) Py_InitModule("_C_gdt", _C_gdtMethods);
    import_array();
}

static void dt1d(float *f, float *out, int len, int *v, float *z) {
    int k = 0;
    v[0] = 0;
    z[0] = 0;
    int q;
    for (q = 1; q < len; q++) {
        while (k >= 0 && f[v[k]] + (v[k] - z[k]) * (v[k] - z[k]) >
                f[q] + (q - z[k]) * (q - z[k]))
            k--;
        if (k < 0) {
            k = 0;
            v[0] = q;
        }
        float s = 1 + ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2 * (q - v[k]));
        if (s < len) {
            k++;
            v[k] = q;
            z[k] = s;
        }
    }

    for (q = len - 1; q >= 0; q--) {
        out[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
        if (q <= z[k]) k--;
    }
}

static inline void strided_load(float *dest, float *src, size_t num, size_t stride) {
    int i;
    for (i = 0; i < num; i++) {
        dest[i] = src[stride * i];
    }
}

static inline void strided_store(float *dest, float *src, size_t num, size_t stride) {
    int i;
    for (i = 0; i < num; i++) {
        dest[stride * i] = src[i];
    }
}

static void dt2d(float *images, int depth, int height, int width) {
    int stacksize = height;
    if (width > height) stacksize = width;
    int *intstack = malloc(stacksize * sizeof(int));
    float *floatstack = malloc(stacksize * sizeof(float));
    int q, d;

    float *line_in = malloc(stacksize * sizeof(float));
    float *line_out = malloc(stacksize * sizeof(float));
    for (d = 0; d < depth; d++) {
        float *f = images + d * height * width;
        // transform rows
        for (q = 0; q < height; q++) {
            memcpy(line_in, f + (q * width), width * sizeof(float));
            dt1d(line_in, f + (q * width), width, intstack, floatstack);
        }

        // transform cols
        for (q = 0; q < width; q++) {
            strided_load(line_in, f + q, height, width);
            memcpy(line_out, line_in, height * sizeof(float));
            dt1d(line_in, line_out, height, intstack, floatstack);
            strided_store(f + q, line_out, height, width);
        }
    }
    free(line_in);
    free(line_out);

    free(intstack);
    free(floatstack);
}

int not_floatmatrix(PyArrayObject *mat) { 
    if (mat->descr->type_num != NPY_FLOAT32 || mat->nd != 3) {
        PyErr_SetString(PyExc_ValueError,
            "In not_floatmatrix: array must be of type Float and 3 dimensional (d x n x m).");
        return 1; }
    return 0;
}

static PyObject *gdt(PyObject *self, PyObject *args) {
    PyArrayObject *mat;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &mat))
        return NULL;

    if (mat == NULL) return NULL;
    if (not_floatmatrix(mat)) return NULL;
    int d = mat->dimensions[0];
    int n = mat->dimensions[1];
    int m = mat->dimensions[2];
    float *cout = (float *) mat->data;
    dt2d(cout, d, n, m);
    Py_INCREF(Py_None);
    return Py_None;
}

