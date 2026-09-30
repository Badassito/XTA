# Native TensorRT ring selection

`YOLO_TTA_NATIVE_TRT_RING` controls the optional two-slot TensorRT source path for native Radial and Spherical TTA views. Its default is `off`.

| Value | Native views admitted to the ring |
| --- | --- |
| `off` (default) | Neither family |
| `all` | Radial and Spherical |
| `radial` | Radial only |
| `spherical` | Spherical only |

The former boolean values remain accepted: `0`, `false`, and `no` mean `off`; `1`, `true`, `yes`, and `on` mean `all`. Values are case insensitive, surrounding spaces are ignored, and an unrecognized value raises an error. For a Radial-only run, set `YOLO_TTA_NATIVE_TRT_RING=radial` in the job environment.

This switch selects source delivery. A task still needs a resident GPU source, a nonempty batch-1 lease, and a compatible fixed-shape TensorRT engine; rejected tasks retain the generic path. The execution provenance records both the historical `native_trt_ring_requested` boolean and the new `native_trt_ring_mode` value so a Radial-only run can be distinguished from `all`.
