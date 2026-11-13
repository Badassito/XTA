"""Stream completed cutoff-native planes from the producer's active detector batch."""
from __future__ import annotations

import os
import threading

import numpy as np

# Input planes warped together by one sibling grid sample (float32 bytes).
_SIBLING_WARP_GROUP_BYTES = 64 * 1024 * 1024


def native_integrity_failure(error):
    from .sweep_views import SweepViewIntegrityError
    return isinstance(error, SweepViewIntegrityError)


def _disabled(task, reason):
    task["_sweep_native_decline_reason"] = str(reason)
    return None


def unsupported_worker_fanout(task, shape):
    """A producer receipt also explains why a window had to use ordinary scalar prediction."""
    return {"status":"unsupported", "supported":False,
            "reason":str(task.get("_sweep_native_decline_reason", "unsupported detector backend")),
            "model_name":str(task.get("model_name","")), "view_name":str(task["view"].name),
            "native_shape":[int(task["view"].num_slices), int(shape[-2]), int(shape[-1])],
            "slice_count":int(shape[0])}


def open_worker_fanout(task, cfg, *, backend, shape, has_scores, padding_count=0):
    """Open the parent's native window only for a qualified producer task."""
    if os.environ.get("XTA_YOLO_SWEEP_NATIVE_ROLE") != "produce":
        return None
    if task.get("sweep_fanout_eligible") is not True:
        return _disabled(task, "scalar configuration outside native fanout contract")
    if (task.get("kind") != "fullframe" or task.get("augmentation_pass_tasks")
            or padding_count or int(cfg.batch) != 1 or task.get("device_hole_fill")
            or task.get("sweep_fanout_hole_fill") or task.get("enable_yolo_hole_fill")
            or float(task.get("streaming_cleanup_min_conf", 0)) != 0
            or float(task.get("streaming_cleanup_min_radius", 0)) != 0):
        return _disabled(task, "unqualified tile, augmentation, padding or cleanup")
    if backend == "cuda" and cfg.task != "segment":
        return _disabled(task, "native GPU fanout requires direct instance segmentation")
    from .sweep_views import SweepViewCapacityError, begin_task_fanout
    frames, h, w = map(int, shape)
    full_shape = (int(task["view"].num_slices), h, w)
    try:
        floor_cent = int(os.environ["XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS"])
        writer = begin_task_fanout(task, cfg, backend=backend, expected_frames=frames,
                                   padding_count=0, native_shape=full_shape, scalar_owned_cutoff=floor_cent)
    except (OSError, MemoryError, SweepViewCapacityError) as error:
        return _disabled(task, f"native journal admission: {error}")
    if writer is None:
        return _disabled(task, "native journal not admitted")
    if os.environ.get("XTA_YOLO_SWEEP_NATIVE_ASYNC_WRITE", "1") != "0":
        # Packing, compression and appends leave the inference thread; one
        # FIFO encoder keeps payload order and index rows byte-identical.
        start = getattr(writer, "start_background_encoding", None)
        if callable(start):
            start()
    sink = WorkerCutoffPlanes(writer, floor_cent=floor_cent,
                              offset=int(task.get("slice_start", 0)), shape=(h,w),
                              has_scores=bool(has_scores), expected_frames=frames)
    sink.receipt_context = {"model_name":str(task.get("model_name","")), "view_name":str(task["view"].name),
                            "native_shape":list(full_shape)}
    return sink


class WorkerCutoffPlanes:
    """Serialize bounded writes; one family failure invalidates the entire task window."""
    def __init__(self, writer, *, floor_cent, offset, shape, has_scores, expected_frames):
        self.writer = writer
        self.floor_cent = floor_cent
        self.offset = offset
        self.shape = shape
        self.has_scores = has_scores
        self.expected_frames = expected_frames
        self._siblings = tuple(v for v in writer.cutoffs if v != floor_cent)
        self._lock = threading.RLock()
        self.reason = None
        self.frames_written = 0
        self.copied_bytes = 0
        self._completed = False
        self._warp = None
        self.device_counts = True
        self._axes = {}
        self._last_crops = {}
        self.scalar_owned_floor = getattr(writer, "scalar_owned_cutoff", None) == floor_cent
        self._floor_seen = np.zeros(expected_frames, dtype=bool) if self.scalar_owned_floor else None
        self._floor_counts = np.zeros(expected_frames, dtype=np.int64) if self.scalar_owned_floor else None
        self._floor_hits = np.full(expected_frames, -1, dtype=np.int8) if self.scalar_owned_floor else None
        self._floor_streams = {}
        self._empty_runs = {}
        self.receipt_context = {}
        self._abort_receipt = {}
        self.counters = dict(rendered_frames=0, render_batches=0, forward_calls=0, mask_decodes=0)

    @property
    def cutoffs(self):
        return () if self.reason is not None else self._siblings

    def configure_warp(self, out_size, matrix, shape):
        if tuple(shape) != self.shape:
            self.disable("fanout native warp grid mismatch")
            return
        self._warp = (int(out_size), np.asarray(matrix, dtype=np.float32))

    def disable(self, reason):
        with self._lock:
            if self.reason is None:
                self.reason = str(reason)
                self._last_crops.clear()
                self._floor_streams.clear()
                self._empty_runs.clear()
                try:
                    self._abort_receipt = self.writer.abort(self.reason) or {}
                except OSError:
                    # No complete acknowledgement is published. Parent-owned
                    # staging is reclaimed with the family work directory.
                    self._abort_receipt = {}

    def abort(self, reason="producer task failed"):
        self.disable(reason)

    def handle_error(self, error):
        if native_integrity_failure(error):
            raise error
        self.disable(error)

    def record_detector_work(self, *, rendered_frames=0, render_batches=0, forward_calls=0, mask_decodes=0):
        with self._lock:
            self.counters["rendered_frames"] += int(rendered_frames)
            self.counters["render_batches"] += int(render_batches)
            self.counters["forward_calls"] += int(forward_calls)
            self.counters["mask_decodes"] += int(mask_decodes)

    def record_proto_work(self, anchors_once, scalar_anchor_entries, active_bins):
        with self._lock:
            for key, amount in (("anchors_once",anchors_once),("scalar_anchor_entries",scalar_anchor_entries),
                                ("active_bins",active_bins)):
                self.counters[key] = self.counters.get(key,0) + int(amount)

    def _record_floor(self, index, count, frames):
        """The scalar owner publishes this mask; retain its real counts without copying pixels."""
        with self._lock:
            if self.reason is not None:
                return
            if not 0 <= index < self.expected_frames or self._floor_seen[index]:
                from .sweep_views import SweepViewIntegrityError
                raise SweepViewIntegrityError("Scalar-owned native floor frame duplicated or outside its task")
            if getattr(count, "is_cuda", False):
                import torch
                stream = torch.cuda.current_stream(count.device)
                key = (count.device, stream.cuda_stream)
                if key not in self._floor_streams:
                    buf = torch.zeros(self.expected_frames, dtype=torch.int64, device=count.device)
                    self._floor_streams[key] = (stream, buf, [])
                _, buf, indices = self._floor_streams[key]
                # Capture the original count before a static engine slot can be reused.
                buf[index].copy_(count.reshape(()), non_blocking=True)
                indices.append(index)
            else:
                self._floor_counts[index] = int(count)
            if frames is not None:
                self._floor_hits[index] = int(frames)
            self._floor_seen[index] = True
            self.frames_written += 1

    def _finish_floor(self):
        if not self.scalar_owned_floor:
            return
        if not self._floor_seen.all():
            from .sweep_views import SweepViewIntegrityError
            raise SweepViewIntegrityError("Scalar-owned floor has incomplete coverage")
        for stream, buf, indices in self._floor_streams.values():
            import torch
            with torch.cuda.stream(stream):
                counts = buf.cpu().numpy()
            self._floor_counts[indices] = counts[indices]
            self.counters["floor_count_d2h_bytes"] = self.counters.get("floor_count_d2h_bytes",0)+counts.nbytes
        self._floor_streams.clear()
        present = np.where(self._floor_hits < 0, self._floor_counts > 0, self._floor_hits)
        self.writer.put_scalar_owned_range(self.floor_cent, self.offset, self.offset+self.expected_frames,
            prediction_counts=self._floor_counts, frames_with_predictions=present)
        self.counters["scalar_owned_floor_frames"] = self.expected_frames

    def _flush_empty(self, cutoff):
        run = self._empty_runs.pop(cutoff, None)
        if run is not None:
            start, predictions, present = run
            self.writer.put_empty_range(cutoff, self.offset + start, self.offset + start + len(present),
                score_lane=self.has_scores, prediction_counts=predictions, frames_with_predictions=present)
            self.counters["empty_range_writes"] = self.counters.get("empty_range_writes",0)+1

    def _record_empty(self, cutoff, index, count, frames):
        run = self._empty_runs.get(cutoff)
        if run is None or run[0]+len(run[1]) != index or len(run[1]) >= 1024:
            self._flush_empty(cutoff)
            run = (index, [], [])
            self._empty_runs[cutoff] = run
        run[1].append(int(count))
        run[2].append(int(frames))
        self.frames_written += 1
        self.counters["bulk_empty_frames"] = self.counters.get("bulk_empty_frames",0)+1

    def _write(self, cutoff, index, mask, scores, bbox, count, frames, *, allow_alias=False, reused=False):
        if self.reason is not None:
            return
        if self.has_scores and scores is None:
            self.disable("qualified branch lacks requested observed scores")
            return
        with self._lock:
            if self.reason is not None:
                return
            if not mask.size and hasattr(self.writer, "put_empty_range"):
                self._record_empty(cutoff,index,count,frames)
            else:
                self._flush_empty(cutoff)
                self.writer.put(int(cutoff), self.offset + int(index), mask,
                                scores if self.has_scores else None, bbox=bbox,
                                prediction_count=int(count), frames_with_predictions=int(frames))
                self.frames_written += 1
            if not reused:
                self.copied_bytes += int(mask.nbytes) + (int(scores.nbytes) if self.has_scores else 0)
            if allow_alias:
                if hasattr(self.writer, "put_alias"):
                    # The durable writer already owns the crop. Only its record coordinates matter.
                    mask = scores = None
                self._last_crops[cutoff == self.floor_cent] = (index, cutoff, mask, scores, bbox, count, frames)

    def consume_same_result(self, index, cutoff, source_cutoff):
        """Reuse the preceding owned CPU crop only with exact same-frame detector identity."""
        with self._lock:
            if self.reason is not None:
                return False
            if self.scalar_owned_floor and (source_cutoff == self.floor_cent or cutoff == self.floor_cent):
                return False
            previous = self._last_crops.get(source_cutoff == self.floor_cent)
            if previous is None or previous[:2] != (index, source_cutoff):
                return False
            try:
                if hasattr(self.writer, "put_alias"):
                    bbox, count, frames = previous[4:]
                    if bbox == (0,0,0,0) and hasattr(self.writer, "put_empty_range"):
                        self._record_empty(cutoff,index,count,frames)
                    else:
                        self._flush_empty(cutoff)
                        self.writer.put_alias(cutoff,self.offset+index,source_cutoff=source_cutoff,
                                              prediction_count=count,frames_with_predictions=frames)
                        self.frames_written += 1
                        self.counters["native_packet_aliases"] = self.counters.get("native_packet_aliases",0)+1
                    self._last_crops[cutoff == self.floor_cent] = (index, cutoff, *previous[2:])
                else:
                    self._write(cutoff, index, *previous[2:], allow_alias=True, reused=True)
                self.counters["identical_native_crops_reused"] = self.counters.get("identical_native_crops_reused",0)+1
                return True
            except (OSError, MemoryError, ValueError, RuntimeError) as error:
                self.handle_error(error)
                return False

    def put(self, cutoff, index, mask, score, *, prediction_count, frames_with_predictions,
            proto_treated=False):
        """Emit one already-restored CPU plane, cropping endpoint-empty rows and columns."""
        if self.reason is not None:
            return
        try:
            if proto_treated:
                self.disable("unqualified proto treatment")
                return
            if self.scalar_owned_floor and cutoff == self.floor_cent:
                self._record_floor(index, prediction_count, frames_with_predictions)
                return
            mask = np.asarray(mask)
            if mask.shape != self.shape or (score is not None and np.shape(score) != self.shape):
                raise ValueError("native mask or confidence grid mismatch")
            occupied_y = np.flatnonzero(mask.any(axis=1))
            if occupied_y.size:
                occupied_x = np.flatnonzero(mask.any(axis=0))
                y0, y1 = int(occupied_y[0]), int(occupied_y[-1]) + 1
                x0, x1 = int(occupied_x[0]), int(occupied_x[-1]) + 1
            else:
                y0 = y1 = x0 = x1 = 0
            cropped = np.ascontiguousarray(mask[y0:y1,x0:x1], dtype=np.uint8)
            values = None if score is None else np.ascontiguousarray(score[y0:y1,x0:x1], dtype=np.uint8)
            if getattr(self.writer, "background_encoding", False):
                # The FIFO encoder reads crops later; never retain a caller buffer.
                if np.may_share_memory(cropped, mask):
                    cropped = cropped.copy()
                if values is not None and np.may_share_memory(values, score):
                    values = values.copy()
            self._write(cutoff, index, cropped, values, (y0,y1,x0,x1), prediction_count, frames_with_predictions)
        except (OSError, MemoryError, ValueError, RuntimeError) as error:
            self.handle_error(error)

    def consume_empty(self, index, cutoff, count=0, *, frames=None):
        if self.reason is not None:
            return
        try:
            if self.scalar_owned_floor and cutoff == self.floor_cent:
                self._record_floor(index,count,frames)
                return
            empty = np.empty((0,0), np.uint8)
            self._write(cutoff,index,empty,empty if self.has_scores else None,(0,0,0,0),count,
                        int(count > 0) if frames is None else frames, allow_alias=self._warp is not None)
        except (OSError, MemoryError, ValueError, RuntimeError) as error:
            self.handle_error(error)

    def consume_native(self, index, cutoff, native_mask, native_score, count, *, proto_treated=False,
                       frames=None, count_foreground=False):
        """Copy only the bounded native crop; count and bbox cross the device together."""
        if self.reason is not None:
            return
        try:
            if proto_treated:
                self.disable("unqualified proto treatment")
                return
            if self.scalar_owned_floor and cutoff == self.floor_cent:
                self._record_floor(index,count,frames)
                return
            import torch
            self._check_native_planes(native_mask, native_score)
            count_t = torch.as_tensor(count, device=native_mask.device, dtype=torch.int64).reshape(())
            height,width = self.shape
            y,x = self._native_axes(native_mask)
            ay,ax = native_mask.any(dim=1).bool(), native_mask.any(dim=0).bool()
            meta = torch.stack((torch.where(ay,y,height).amin(),torch.where(ay,y,-1).amax()+1,
                                torch.where(ax,x,width).amin(),torch.where(ax,x,-1).amax()+1,count_t))
            y0,y1,x0,x1,n = meta.cpu().tolist()
            if count_foreground:
                frames = int(y1 > y0)
            if y1 <= y0:
                self.consume_empty(index,cutoff,n,frames=frames)
                return
            if self.has_scores and native_score is None:
                raise ValueError("missing observed scores on nonempty native pixels")
            # One private buffer holds both crops: a single device-to-host copy,
            # and CPU tensors never alias static ring bindings.
            (mask, score), = self._crops_to_host(
                ((native_mask, native_score if self.has_scores else None, (y0,y1,x0,x1)),))
            self._write(cutoff,index,mask,score,(y0,y1,x0,x1),n, int(n > 0) if frames is None else frames,
                        allow_alias=self._warp is not None)
        except (OSError, MemoryError, ValueError, RuntimeError) as error:
            self.handle_error(error)

    def _check_native_planes(self, native_mask, native_score):
        import torch
        if tuple(native_mask.shape) != self.shape:
            raise ValueError("native device mask grid mismatch")
        if native_score is not None and (tuple(native_score.shape) != self.shape or native_score.dtype != torch.uint8):
            raise ValueError("native device confidence grid or precision mismatch")

    def _native_axes(self, native_mask):
        import torch
        height,width = self.shape
        # The scalar floor can arrive on a postprocess side stream while
        # sibling decodes retire on the inference or TensorRT post stream.
        stream = torch.cuda.current_stream(native_mask.device).cuda_stream if native_mask.is_cuda else 0
        axes_key = (native_mask.device, stream)
        if axes_key not in self._axes:
            self._axes[axes_key] = (torch.arange(height, device=native_mask.device),
                                    torch.arange(width, device=native_mask.device))
        return self._axes[axes_key]

    @staticmethod
    def _crops_to_host(crops):
        """Copy (mask, score|None, bbox) crops through one private uint8 buffer.

        Masks convert exactly as ``.to(torch.uint8)``. On CUDA this is one
        synchronizing transfer for every crop; the returned arrays are views of
        fresh host storage that no device or caller buffer aliases.
        """
        import torch
        layout, total = [], 0
        for _mask, score, (y0,y1,x0,x1) in crops:
            area = (y1-y0)*(x1-x0)
            layout.append((total, None if score is None else total+area, y1-y0, x1-x0))
            total += area*(1 if score is None else 2)
        flat = torch.empty(total, dtype=torch.uint8, device=crops[0][0].device)
        for (mask, score, (y0,y1,x0,x1)), (m, s, h, w) in zip(crops, layout):
            flat[m:m+h*w].view(h,w).copy_(mask[y0:y1,x0:x1])
            if s is not None:
                flat[s:s+h*w].view(h,w).copy_(score[y0:y1,x0:x1])
        host = flat.cpu().numpy()
        return [(host[m:m+h*w].reshape(h,w), None if s is None else host[s:s+h*w].reshape(h,w))
                for m, s, h, w in layout]

    def consume_result(self, index, cutoff, payload):
        """Apply the same scalar GPU native warp and score quantization to one sibling."""
        if self.reason is not None:
            return
        try:
            if self._warp is None:
                raise ValueError("native cutoff warp was not configured")
            if payload.union_gpu is None:
                self.consume_empty(index,cutoff,int(payload.instance_count),frames=0)
                return
            from .inference import _warp_flattened_native_planes
            if payload.ready_event is not None and payload.union_gpu.is_cuda:
                import torch
                torch.cuda.current_stream(payload.union_gpu.device).wait_event(payload.ready_event)
            size,matrix = self._warp
            mask,score = _warp_flattened_native_planes(payload,matrix,size,*self.shape,self.has_scores)
            count = payload.instance_count_device if payload.instance_count_device is not None else payload.instance_count
            self.consume_native(index,cutoff,mask,score,count,count_foreground=not self.device_counts)
        except (OSError, MemoryError, ValueError, RuntimeError) as error:
            self.handle_error(error)

    def consume_results(self, index, results):
        """Consume one frame's sibling ``(cutoff, payload)`` pairs with two host reads in all.

        Each payload is warped as it arrives, so a lazy iterable frees its decode
        workspace before the next decode. Bounds and device counts of every
        sibling then cross in one copy and all nonempty crops in a second. The
        writes are issued in payload order with exactly the arguments that
        successive consume_result calls would pass.
        """
        if self.reason is not None:
            return
        staged, pending, pending_bytes = [], [], 0
        try:
            if self._warp is None:
                raise ValueError("native cutoff warp was not configured")
            for cutoff, payload in results:
                if self.reason is not None:
                    return
                if self.scalar_owned_floor and cutoff == self.floor_cent:
                    self._warp_sibling_group(staged, pending)
                    pending, pending_bytes = [], 0
                    self._flush_sibling_results(index, staged)
                    staged = []
                    self.consume_result(index, cutoff, payload)
                    continue
                if payload.union_gpu is None:
                    staged.append([int(cutoff), None, None, int(payload.instance_count)])
                    continue
                if payload.ready_event is not None and payload.union_gpu.is_cuda:
                    import torch
                    torch.cuda.current_stream(payload.union_gpu.device).wait_event(payload.ready_event)
                count = payload.instance_count_device if payload.instance_count_device is not None else payload.instance_count
                staged.append([int(cutoff), None, None, count])
                pending.append((len(staged) - 1, payload))
                planes = 2 if self.has_scores and payload.conf_gpu is not None else 1
                pending_bytes += planes * 4 * int(payload.union_gpu.numel())
                if pending_bytes >= _SIBLING_WARP_GROUP_BYTES:
                    self._warp_sibling_group(staged, pending)
                    pending, pending_bytes = [], 0
            self._warp_sibling_group(staged, pending)
            self._flush_sibling_results(index, staged)
        except (OSError, MemoryError, ValueError, RuntimeError) as error:
            self.handle_error(error)

    def _warp_sibling_group(self, staged, pending):
        """Warp several payloads with one grid sample and one score quantization.

        Nearest sampling and the quantization are per-plane elementwise, so each
        sibling's planes equal ``_warp_flattened_native_planes`` exactly; a
        one-member group uses that helper itself.
        """
        if not pending:
            return
        from . import inference
        size,matrix = self._warp
        height,width = self.shape
        if len(pending) == 1:
            position, payload = pending[0]
            mask,score = inference._warp_flattened_native_planes(payload,matrix,size,height,width,self.has_scores)
        else:
            import torch
            import torch.nn.functional as F

            def prepared(plane):
                value = plane if plane.dtype == torch.float32 else plane.to(torch.float32)
                if int(value.shape[0]) != int(size) or int(value.shape[1]) != int(size):
                    value = F.interpolate(value.reshape(1, 1, int(value.shape[0]), int(value.shape[1])),
                                          size=(int(size), int(size)), mode='nearest').reshape(int(size), int(size))
                return value
            unions = [prepared(payload.union_gpu) for _, payload in pending]
            scored = [k for k, (_, payload) in enumerate(pending) if self.has_scores and payload.conf_gpu is not None]
            planes = unions + [prepared(pending[k][1].conf_gpu) for k in scored]
            if int(height) == int(size) and int(width) == int(size) and inference._warp_matrix_is_identity(matrix):
                warped = torch.stack(planes, dim=0)
            else:
                theta = inference._affine_theta_for_grid_sample(matrix, int(size), int(height), int(width))
                grid = inference._get_cached_affine_grid(theta, int(height), int(width), unions[0].device)
                warped = F.grid_sample(torch.stack(planes, dim=0).unsqueeze(0), grid, mode='nearest',
                                       padding_mode='zeros', align_corners=False)[0]
            masks = warped[:len(unions)] > 0.5
            scores = None
            if scored:
                value = warped[len(unions):]
                support = masks if len(scored) == len(unions) else masks[scored]
                scores = torch.where(support, (value.clamp(0.0, 1.0) * float(inference.CONF_U8_MAX)).round(),
                                     torch.zeros((), dtype=value.dtype, device=value.device)
                                     ).clamp(0.0, 255.0).to(torch.uint8)
            rows = {k: j for j, k in enumerate(scored)}
            for k, (position, _payload) in enumerate(pending):
                score = scores[rows[k]] if k in rows else None
                self._check_native_planes(masks[k], score)
                staged[position][1:3] = masks[k], score
            return
        self._check_native_planes(mask, score)
        staged[position][1:3] = mask, score

    def _flush_sibling_results(self, index, staged):
        if not staged or self.reason is not None:
            return
        import torch
        live = [i for i, item in enumerate(staged) if item[1] is not None]
        bounds, counts, crops = {}, {}, []
        if live:
            first = staged[live[0]][1]
            height,width = self.shape
            y,x = self._native_axes(first)
            planes = torch.stack([staged[i][1] if staged[i][1].dtype == torch.bool else staged[i][1].ne(0)
                                  for i in live])
            ay,ax = planes.any(dim=2), planes.any(dim=1)
            box = torch.stack((torch.where(ay,y,height).amin(dim=1), torch.where(ay,y,-1).amax(dim=1)+1,
                               torch.where(ax,x,width).amin(dim=1), torch.where(ax,x,-1).amax(dim=1)+1), dim=1)
            device_counts = [i for i in live if torch.is_tensor(staged[i][3])]
            parts = [box.reshape(-1)]
            if device_counts:
                parts.append(torch.stack([torch.as_tensor(staged[i][3], device=first.device, dtype=torch.int64).reshape(())
                                          for i in device_counts]))
            values = torch.cat(parts).cpu().tolist()  # the frame's one metadata synchronization
            for k, i in enumerate(live):
                bounds[i] = tuple(int(v) for v in values[4*k:4*k+4])
            counts = dict(zip(device_counts, values[4*len(live):]))
            for i in live:
                y0,y1,x0,x1 = bounds[i]
                if y1 > y0:
                    if self.has_scores and staged[i][2] is None:
                        raise ValueError("missing observed scores on nonempty native pixels")
                    crops.append((staged[i][1], staged[i][2] if self.has_scores else None, bounds[i]))
        host = iter(self._crops_to_host(crops) if crops else ())  # the frame's one crop transfer
        for i, (cutoff, _mask, _score, count) in enumerate(staged):
            if self.reason is not None:
                return
            if i not in bounds:
                self.consume_empty(index,cutoff,count,frames=0)
                continue
            y0,y1,x0,x1 = bounds[i]
            n = counts[i] if i in counts else int(count)
            frames = None if self.device_counts else int(y1 > y0)
            if y1 <= y0:
                self.consume_empty(index,cutoff,n,frames=frames)
                continue
            mask,score = next(host)
            self._write(cutoff,index,mask,score,(y0,y1,x0,x1),n, int(n > 0) if frames is None else frames,
                        allow_alias=self._warp is not None)

    def finish(self):
        with self._lock:
            expected = self.expected_frames * (len(self._siblings) + 1)
            if self.reason is None and self.frames_written != expected:
                self.disable(f"incomplete native detector products: {self.frames_written}/{expected}")
            detail = {}
            if self.reason is None and not self._completed:
                try:
                    self._finish_floor()
                    for cent in tuple(self._empty_runs):
                        self._flush_empty(cent)
                    detail = self.writer.finish() or {}
                    self._completed = True
                except (OSError, MemoryError, ValueError, RuntimeError) as error:
                    self.handle_error(error)
            receipt = dict(self.receipt_context)
            if self.reason is not None and isinstance(self._abort_receipt,dict):
                receipt.update(self._abort_receipt)
            elif isinstance(detail,dict):
                receipt.update(detail)
            receipt.setdefault("status", "complete" if self.reason is None else "unsupported")
            receipt.update(supported=self.reason is None, slice_count=self.expected_frames,
                           produced_cutoff_frames=self.frames_written, native_crop_bytes=self.copied_bytes)
            receipt["counters"] = dict(self.counters, native_crop_bytes=self.copied_bytes)
            self._last_crops.clear()
            if self.reason is not None:
                receipt["reason"] = self.reason
            return receipt
