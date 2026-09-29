"""Independent NumPy oracle for packed big-bit-order coordinate accumulation."""
import numpy as np


def scatter(destination, ti, yi, xi, *, out_h, packed_w):
    stride = np.int64(int(out_h) * int(packed_w))
    indices = ti.astype(np.int64, copy=False) * stride
    indices += yi.astype(np.int64, copy=False) * np.int64(packed_w)
    indices += xi.astype(np.int64, copy=False) >> np.int64(3)
    bits = np.left_shift(np.uint8(1),
                        (np.int32(7) - (xi.astype(np.int32, copy=False) & 7)).astype(np.uint8))
    np.bitwise_or.at(destination, indices, bits)
