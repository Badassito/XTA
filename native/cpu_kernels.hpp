#ifndef XTA_CPU_KERNELS_HPP
#define XTA_CPU_KERNELS_HPP

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace xta::cpu {

// All errors in the Python bridge are raised with the GIL held. A native loop
// can use a C++ exception to unwind its GilRelease before setting a Python error.
struct PythonError {};

[[noreturn]] inline void value_error(const char* message) {
    PyErr_SetString(PyExc_ValueError, message);
    throw PythonError{};
}

[[noreturn]] inline void type_error(const char* message) {
    PyErr_SetString(PyExc_TypeError, message);
    throw PythonError{};
}

[[noreturn]] inline void overflow_error(const char* message) {
    PyErr_SetString(PyExc_OverflowError, message);
    throw PythonError{};
}

template<class Function>
PyObject* protect(Function&& call) noexcept {
    try {
        return std::forward<Function>(call)();
    } catch (const PythonError&) {
        return nullptr;
    } catch (const std::bad_alloc&) {
        return PyErr_NoMemory();
    } catch (const std::length_error& error) {
        PyErr_SetString(PyExc_OverflowError, error.what());
    } catch (const std::overflow_error& error) {
        PyErr_SetString(PyExc_OverflowError, error.what());
    } catch (const std::invalid_argument& error) {
        PyErr_SetString(PyExc_ValueError, error.what());
    } catch (const std::exception& error) {
        PyErr_SetString(PyExc_RuntimeError, error.what());
    } catch (...) {
        PyErr_SetString(PyExc_RuntimeError, "Unexpected native CPU kernel failure");
    }
    return nullptr;
}

class Object {
public:
    Object() noexcept = default;
    // Takes a new reference. A null reference preserves the current Python error.
    explicit Object(PyObject* object) : object_(object) {
        if (!object_) {
            throw PythonError{};
        }
    }
    Object(const Object&) = delete;
    Object& operator=(const Object&) = delete;
    Object(Object&& other) noexcept : object_(other.release()) {}
    Object& operator=(Object&& other) noexcept {
        if (this != &other) {
            Py_XDECREF(object_);
            object_ = other.release();
        }
        return *this;
    }
    ~Object() { Py_XDECREF(object_); }

    PyObject* get() const noexcept { return object_; }
    PyObject* release() noexcept { return std::exchange(object_, nullptr); }

private:
    PyObject* object_ = nullptr;
};

class GilRelease {
public:
    GilRelease() : state_(PyEval_SaveThread()) {}
    GilRelease(const GilRelease&) = delete;
    GilRelease& operator=(const GilRelease&) = delete;
    ~GilRelease() { PyEval_RestoreThread(state_); }

private:
    PyThreadState* state_;
};

inline Py_ssize_t checked_add(Py_ssize_t left, Py_ssize_t right) {
    if ((right > 0 && left > PY_SSIZE_T_MAX - right)
        || (right < 0 && left < PY_SSIZE_T_MIN - right)) {
        throw std::overflow_error("Native CPU index addition overflow");
    }
    return left + right;
}

// The dimension/count is nonnegative. The stride can have either sign.
inline Py_ssize_t checked_extent(Py_ssize_t count, Py_ssize_t stride) {
    if (count < 0) {
        throw std::invalid_argument("Negative native CPU buffer dimension");
    }
    if (count && (stride > PY_SSIZE_T_MAX / count || stride < PY_SSIZE_T_MIN / count)) {
        throw std::overflow_error("Native CPU buffer extent overflow");
    }
    return count * stride;
}

// Read a fixed-width little-endian field from a possibly unaligned wire record.
template<class T>
T load_little_endian(const void* address) noexcept {
    static_assert(std::is_unsigned_v<T>);
#if defined(_WIN32) || (defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    T result;
    std::memcpy(&result, address, sizeof(result));
    return result;
#else
    const auto* bytes = static_cast<const unsigned char*>(address);
    T result = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) result |= static_cast<T>(bytes[i]) << (8 * i);
    return result;
#endif
}

class Buffer {
public:
    // ndim=-1 accepts any rank. The exporter stays pinned until destruction.
    Buffer(PyObject* object, const char* label, int ndim = -1, bool writable = false)
        : label_(label) {
        const int flags = PyBUF_STRIDES | PyBUF_FORMAT | (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(object, &view_, flags) < 0) {
            throw PythonError{};
        }
        try {
            validate(ndim);
        } catch (...) {
            PyBuffer_Release(&view_);
            throw;
        }
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { PyBuffer_Release(&view_); }

    int ndim() const noexcept { return view_.ndim; }
    Py_ssize_t shape(int axis) const noexcept { return view_.shape[axis]; }
    Py_ssize_t stride(int axis) const noexcept { return view_.strides[axis]; }
    Py_ssize_t itemsize() const noexcept { return view_.itemsize; }
    Py_ssize_t nbytes() const noexcept { return view_.len; }
    const char* format() const noexcept { return view_.format; }
    bool readonly() const noexcept { return view_.readonly != 0; }
    bool c_contiguous() const noexcept { return PyBuffer_IsContiguous(&view_, 'C') != 0; }
    char* data() const noexcept { return static_cast<char*>(view_.buf); }

    template<class T>
    bool is() const noexcept {
        if (itemsize() != static_cast<Py_ssize_t>(sizeof(T))) {
            return false;
        }
        const char* code = format();
        if (*code == '@' || *code == '=') {
            ++code;
        } else if (*code == '<' || *code == '>' || *code == '!') {
            const std::uint16_t one = 1;
            const bool little = *reinterpret_cast<const unsigned char*>(&one) == 1;
            if (sizeof(T) > 1 && little != (*code == '<')) {
                return false;
            }
            ++code;
        }
        if (!code[0] || code[1]) {
            return false;
        }
        if constexpr (std::is_same_v<T, bool>) {
            return *code == '?';
        } else if constexpr (std::is_same_v<T, float>) {
            return *code == 'f';
        } else if constexpr (std::is_same_v<T, double>) {
            return *code == 'd';
        } else if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
            return std::strchr("bhilqn", *code) != nullptr;
        } else if constexpr (std::is_integral_v<T>) {
            return std::strchr("BHILQN", *code) != nullptr;
        } else {
            return false;
        }
    }

    template<class T>
    const Buffer& require() const {
        if (!is<T>()) {
            PyErr_Format(PyExc_TypeError, "%s has an unsupported buffer dtype: %s", label_, format());
            throw PythonError{};
        }
        return *this;
    }

    void require_contiguous() const {
        if (!c_contiguous()) {
            PyErr_Format(PyExc_ValueError, "%s must be C contiguous", label_);
            throw PythonError{};
        }
    }

    // The caller validates the rank and bounds for the loop computing indices.
    template<class... Index>
    char* ptr(Index... index) const noexcept {
        const std::array<Py_ssize_t, sizeof...(Index)> indices{static_cast<Py_ssize_t>(index)...};
        Py_ssize_t offset = 0;
        for (std::size_t axis = 0; axis < indices.size(); ++axis) {
            offset += indices[axis] * view_.strides[axis];
        }
        return data() + offset;
    }

    template<class T, class... Index>
    T load(Index... index) const noexcept {
        if constexpr (std::is_same_v<T, bool>) {
            return *reinterpret_cast<const unsigned char*>(ptr(index...)) != 0;
        } else {
            T value;
            std::memcpy(&value, ptr(index...), sizeof(value));
            return value;
        }
    }

    template<class T, class... Index>
    void store(T value, Index... index) const noexcept {
        if constexpr (std::is_same_v<T, bool>) {
            *reinterpret_cast<unsigned char*>(ptr(index...)) = static_cast<unsigned char>(value);
        } else {
            std::memcpy(ptr(index...), &value, sizeof(value));
        }
    }

private:
    void validate(int expected_ndim) const {
        if (view_.ndim < 0 || (expected_ndim >= 0 && view_.ndim != expected_ndim)) {
            PyErr_Format(PyExc_ValueError, "%s has an unsupported buffer rank: %d", label_, view_.ndim);
            throw PythonError{};
        }
        if (view_.itemsize <= 0 || view_.len < 0 || view_.format == nullptr
            || (!view_.buf && view_.len) || (view_.ndim && (!view_.shape || !view_.strides))
            || view_.suboffsets) {
            PyErr_Format(PyExc_BufferError, "%s must export a direct strided buffer", label_);
            throw PythonError{};
        }
        bool empty = false;
        for (int axis = 0; axis < view_.ndim; ++axis) {
            if (shape(axis) < 0) {
                value_error("Negative native CPU buffer dimension");
            }
            empty |= shape(axis) == 0;
        }
        Py_ssize_t count_bytes = empty ? 0 : view_.itemsize;
        Py_ssize_t lowest = 0;
        Py_ssize_t highest = 0;
        for (int axis = 0; !empty && axis < view_.ndim; ++axis) {
            const auto extent = checked_extent(shape(axis) - 1, stride(axis));
            lowest = checked_add(lowest, extent < 0 ? extent : 0);
            highest = checked_add(highest, extent > 0 ? extent : 0);
            count_bytes = checked_extent(shape(axis), count_bytes);
        }
        if (!empty) {
            checked_add(highest, view_.itemsize - 1);
        }
        if (count_bytes != view_.len) {
            PyErr_Format(PyExc_BufferError, "%s has an inconsistent buffer length", label_);
            throw PythonError{};
        }
    }

    const char* label_;
    Py_buffer view_{};
};

// NumPy sees only a normal Python call made under the GIL. Native code acquires
// the returned array as a Buffer for its bulk loop. No NumPy headers or ABI.
Object empty(std::initializer_list<Py_ssize_t> shape, const char* dtype);
Object zeros(std::initializer_list<Py_ssize_t> shape, const char* dtype);

// Domain methods can define their registration inside this namespace or use
// the fully qualified name after the declaration below.
#define XTA_CPU_DOMAIN(domain) int register_##domain(PyObject*);
#include "cpu_domains.inc"
#undef XTA_CPU_DOMAIN

} // namespace xta::cpu

#endif
