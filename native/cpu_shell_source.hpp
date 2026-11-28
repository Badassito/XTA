#ifndef XTA_CPU_SHELL_SOURCE_HPP
#define XTA_CPU_SHELL_SOURCE_HPP

#include "cpu_kernels.hpp"

#include <memory>

namespace xta::cpu {

// Borrow the dense array or CVOL payload. Encoded access decodes one pixel.
class ShellSource {
public:
    explicit ShellSource(PyObject* source) {
        if (!PyTuple_Check(source)) {
            dense_ = std::make_unique<Buffer>(source, "Shell source", 3);
            dense_bool_ = dense_->is<bool>();
            for (int axis = 0; axis < 3; ++axis) {
                shape_[axis] = dense_->shape(axis);
            }
        } else {
            if (PyTuple_GET_SIZE(source) != 4) {
                value_error("Invalid encoded shell source descriptor");
            }
            auto* shape = PyTuple_GET_ITEM(source, 0);
            if (!PyTuple_Check(shape) || PyTuple_GET_SIZE(shape) != 3) {
                value_error("Encoded shell shape must have three dimensions");
            }
            for (int axis = 0; axis < 3; ++axis) {
                shape_[axis] = PyLong_AsSsize_t(PyTuple_GET_ITEM(shape, axis));
                if (PyErr_Occurred()) {
                    throw PythonError{};
                }
            }
            encoded_ = std::make_unique<Buffer>(PyTuple_GET_ITEM(source, 1), "Shell payload", 1);
            index_ = std::make_unique<Buffer>(PyTuple_GET_ITEM(source, 2), "Shell index", 2);
            encoded_->require<std::uint8_t>().require_contiguous();
            index_->require<std::uint8_t>().require_contiguous();
            if (index_->shape(0) != shape_[0] || index_->shape(1) != 48
                || !encoded_->readonly() || !index_->readonly()) {
                value_error("Encoded shell source requires a read-only payload and CVOL index");
            }
            const int packed = PyObject_IsTrue(PyTuple_GET_ITEM(source, 3));
            if (packed < 0) {
                throw PythonError{};
            }
            packed_ = packed == 1;
        }
        if (shape_[0] <= 0 || shape_[1] <= 0 || shape_[2] <= 0) {
            value_error("Shell source must be nonempty");
        }
    }

    const Buffer* dense() const noexcept { return dense_.get(); }
    Py_ssize_t shape(int axis) const noexcept { return shape_[axis]; }

    template<class T>
    T value(Py_ssize_t shell, Py_ssize_t row, Py_ssize_t column) const {
        return dense_ ? dense_->load<T>(shell, row, column)
                      : static_cast<T>(encoded_bit(shell, row, column));
    }

    std::uint8_t uint8(Py_ssize_t shell, Py_ssize_t row, Py_ssize_t column) const {
        return dense_bool_
            ? static_cast<std::uint8_t>(dense_->load<bool>(shell, row, column))
            : value<std::uint8_t>(shell, row, column);
    }

    bool mask(Py_ssize_t shell, Py_ssize_t row, Py_ssize_t column) const {
        return uint8(shell, row, column) != 0;
    }

private:
    bool encoded_bit(Py_ssize_t shell, Py_ssize_t row, Py_ssize_t column) const {
        const auto* record = reinterpret_cast<const unsigned char*>(index_->ptr(shell, 0));
        if (record[0] == 0) {
            return false;
        }
        const std::uint64_t top = load_little_endian<std::uint32_t>(record + 24);
        const std::uint64_t left = load_little_endian<std::uint32_t>(record + 28);
        const std::uint64_t bottom = load_little_endian<std::uint32_t>(record + 32);
        const std::uint64_t right = load_little_endian<std::uint32_t>(record + 36);
        if (record[0] != 1 || top >= bottom || left >= right || bottom > std::uint64_t(shape_[1])
            || right > std::uint64_t(shape_[2])) {
            throw std::invalid_argument("Invalid encoded shell bounds");
        }
        if (row < 0 || column < 0 || std::uint64_t(row) < top || std::uint64_t(row) >= bottom
            || std::uint64_t(column) < left || std::uint64_t(column) >= right) {
            return false;
        }
        const auto local_column = std::uint64_t(column) - left;
        const auto width = right - left;
        const auto stride = packed_ ? (width + 7) / 8 : width;
        const auto relative = (std::uint64_t(row) - top) * stride
            + (packed_ ? local_column / 8 : local_column);
        const auto offset = load_little_endian<std::uint64_t>(record + 8);
        const auto payload_size = load_little_endian<std::uint64_t>(record + 16);
        const auto length = std::uint64_t(encoded_->shape(0));
        if (offset >= length || payload_size > length - offset || relative >= payload_size) {
            throw std::invalid_argument("Invalid encoded shell payload extent");
        }
        const auto value = encoded_->load<std::uint8_t>(static_cast<Py_ssize_t>(offset + relative));
        return packed_ ? ((value >> (local_column & 7)) & 1) != 0 : value != 0;
    }

    std::array<Py_ssize_t, 3> shape_{};
    std::unique_ptr<Buffer> dense_;
    std::unique_ptr<Buffer> encoded_;
    std::unique_ptr<Buffer> index_;
    bool packed_ = false;
    bool dense_bool_ = false;
};

} // namespace xta::cpu

#endif
