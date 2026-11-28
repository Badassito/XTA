#ifndef XTA_CPU_CVOL_SOURCE_HPP
#define XTA_CPU_CVOL_SOURCE_HPP

#include "cpu_coverage_pull.hpp"

namespace xta::cpu::coverage {
class CvolSource {
  public:
    const Buffer &bytes;
    const Buffer &records;
    bool packed;
    CvolSource(const Buffer &data, const Buffer &index, bool is_packed)
        : bytes(data), records(index), packed(is_packed) {
        bytes.require<std::uint8_t>();
        records.require<std::uint8_t>();
        if (bytes.ndim() != 1 || records.ndim() != 2 || records.shape(1) != 48 || records.stride(1) != 1)
            value_error("Invalid encoded CVOL projection source");
    }

    // CVOL is a persisted little-endian format. The rows can be unaligned and
    // strided (including backwards); they are 48 bytes by the format contract.
    template <class T> T field(I frame, int offset) const {
        return load_little_endian<T>(records.ptr(frame, offset));
    }

    std::uint8_t read(I frame, I row, I column, Stats &stats) const {
        if (frame < 0)
            frame += records.shape(0);
        if (frame < 0 || frame >= records.shape(0))
            throw std::invalid_argument("CVOL frame exceeds its index");
        if (field<std::uint8_t>(frame, 0) == 0)
            return 0;
        const U y0 = field<std::uint32_t>(frame, 24), x0 = field<std::uint32_t>(frame, 28);
        const U y1 = field<std::uint32_t>(frame, 32), x1 = field<std::uint32_t>(frame, 36);
        if (row < 0 || column < 0 || U(row) < y0 || U(row) >= y1 || U(column) < x0 || U(column) >= x1)
            return 0;
        const U width = x1 - x0, local = U(column) - x0;
        const U stride = packed ? (width + 7) / 8 : width;
        const U offset = field<U>(frame, 8);
        const U address = (U(row) - y0) * stride + (packed ? local / 8 : local);
        if (offset >= U(bytes.shape(0)) || address >= U(bytes.shape(0)) - offset ||
            address >= field<U>(frame, 16))
            throw std::invalid_argument("CVOL crop exceeds its encoded payload");
        auto pixel = bytes.load<std::uint8_t>(I(offset + address));
        if (packed)
            pixel = (pixel >> (local % 8)) & 1;
        ++stats.reads;
        stats.positive += (pixel != 0);
        return pixel;
    }
};

} // namespace xta::cpu::coverage
#endif
