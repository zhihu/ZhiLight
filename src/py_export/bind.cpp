#include "py_export/bind.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(C, m) {
    bind::define_model_config(m);
    bind::define_quant_config(m);
    bind::define_dist_config(m)
    bind::define_engine(m);

    // models
    bind::define_cpm_base(m);
    bind::define_llama(m);

    bind::define_dynamic_batch(m);
}
