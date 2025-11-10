#include <torch/extension.h>
#include <pybind11/pybind11.h>

torch::Tensor conv2d(const torch::Tensor& inArray, const torch::Tensor& filter, int radius);
torch::Tensor conv2dConstMem(const torch::Tensor& inArray, const torch::Tensor& filter, int radius);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "conv2d", &conv2d, "2D convolution (CUDA)",
        py::arg("inArray"), py::arg("filter"), py::arg("radius")
    );
    m.def(
        "conv2dConstMem", &conv2dConstMem, "2D convolution using constant memory (CUDA)",
        py::arg("inArray"), py::arg("filter"), py::arg("radius")
    );
}

