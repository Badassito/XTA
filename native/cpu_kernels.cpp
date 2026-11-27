#include "cpu_kernels.hpp"

#if defined(__FAST_MATH__) || defined(_M_FP_FAST)
#error "XTA native CPU kernels require strict floating-point compilation"
#endif

#define XTA_STRINGIFY_IMPL(token) #token
#define XTA_STRINGIFY(token) XTA_STRINGIFY_IMPL(token)
#if defined(__clang__)
#define XTA_CPU_COMPILER "Clang " __clang_version__
#elif defined(_MSC_VER)
#define XTA_CPU_COMPILER "MSVC " XTA_STRINGIFY(_MSC_FULL_VER)
#elif defined(__GNUC__)
#define XTA_CPU_COMPILER "GCC " __VERSION__
#else
#define XTA_CPU_COMPILER "unidentified C++17 compiler"
#endif

namespace xta::cpu {

namespace {

Object allocate(const char* operation, std::initializer_list<Py_ssize_t> shape, const char* dtype) {
    Object numpy(PyImport_ImportModule("numpy"));
    Object function(PyObject_GetAttrString(numpy.get(), operation));
    Object dimensions(PyTuple_New(static_cast<Py_ssize_t>(shape.size())));
    Py_ssize_t axis = 0;
    for (auto dimension : shape) {
        Object length(PyLong_FromSsize_t(dimension));
        PyTuple_SET_ITEM(dimensions.get(), axis++, length.release());
    }
    return Object(PyObject_CallFunction(function.get(), "Os", dimensions.get(), dtype));
}

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT, "_xta_cpu_kernels", "Ahead-of-time native XTA CPU kernels.",
    -1, nullptr, nullptr, nullptr, nullptr, nullptr
};

} // namespace

Object empty(std::initializer_list<Py_ssize_t> shape, const char* dtype) {
    return allocate("empty", shape, dtype);
}

Object zeros(std::initializer_list<Py_ssize_t> shape, const char* dtype) {
    return allocate("zeros", shape, dtype);
}

} // namespace xta::cpu

PyMODINIT_FUNC PyInit__xta_cpu_kernels() {
    using namespace xta::cpu;
    return protect([]() -> PyObject* {
        Object module(PyModule_Create(&module_def));
        if (PyModule_AddIntConstant(module.get(), "ABI_VERSION", 1) < 0
            || PyModule_AddStringConstant(module.get(), "SOURCE_SHA256", XTA_CPU_SOURCE_SHA256) < 0
            || PyModule_AddStringConstant(module.get(), "COMPILER", XTA_CPU_COMPILER) < 0
            || PyModule_AddStringConstant(module.get(), "FP_CONTRACT", "explicit_fma_only") < 0) {
            return nullptr;
        }
#define XTA_CPU_DOMAIN(domain) \
        if (register_##domain(module.get()) < 0 \
            || PyModule_AddStringConstant(module.get(), #domain "_SOURCE_SHA256", XTA_CPU_DIGEST_##domain) < 0) { \
            return nullptr; \
        }
#include "cpu_domains.inc"
#undef XTA_CPU_DOMAIN
        return module.release();
    });
}
