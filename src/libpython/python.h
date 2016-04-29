#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "docstr.h"
#include <mitsuba/core/object.h>

PYBIND11_DECLARE_HOLDER_TYPE(T, mitsuba::ref<T>);

#define D(...) DOC(__VA_ARGS__)
#define DM(...) DOC(mitsuba, __VA_ARGS__)

#define MTS_PY_DECLARE(name) \
    extern void python_export_##name(py::module &)

#define MTS_PY_IMPORT(name) \
    python_export_##name(m)

#define MTS_PY_EXPORT(name) \
    void python_export_##name(py::module &m)

#define MTS_PY_CLASS(Name, ...) \
    py::class_<Name, ref<Name>>(m, #Name, DM(Name), ##__VA_ARGS__)

#define qdef(Class, Function, ...) \
    def(#Function, &Class::Function, DM(Class, Function), ##__VA_ARGS__)

using namespace mitsuba;

namespace py = pybind11;
