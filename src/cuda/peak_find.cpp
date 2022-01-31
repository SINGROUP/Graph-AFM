
#include <torch/extension.h>

#include <defs.h>

torch::Tensor match_template(torch::Tensor array, torch::Tensor temp, TMMethod method) {
    CHECK_INPUT(array);
    CHECK_INPUT(temp);
    CHECK4D(array);
    CHECK3D(temp);
    return match_template_cuda(array, temp, method);
}

torch::Tensor ccl(torch::Tensor array) {
    CHECK_INPUT(array);
    CHECK4D(array);
    return ccl_cuda(array);
}

std::vector<torch::Tensor> find_label_min(torch::Tensor array, torch::Tensor labels) {
    CHECK_INPUT(array);
    CHECK_INPUT(labels);
    CHECK4D(array);
    CHECK4D(labels);
    return find_label_min_cuda(array, labels);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("match_template", &match_template, "Match template");
  m.def("ccl", &ccl, "Connected-component labelling");
  m.def("find_label_min", &find_label_min, "Find min indices in labelled regions");
  py::enum_<TMMethod>(m, "TMMethod")
    .value("mad", TMMethod::TM_MAD)
    .value("msd", TMMethod::TM_MSD)
    .value("mad_norm", TMMethod::TM_MAD_NORM)
    .value("msd_norm", TMMethod::TM_MSD_NORM)
    .export_values();
  m.attr("MAX_LABELS") = MAX_LABELS;
}
