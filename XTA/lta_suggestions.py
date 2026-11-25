"""Rank transverse annotation suggestions using settled directional masks."""

from collections.abc import Sequence


class SliceSuggestions:
    """Keep only the best eligible frame and exact disagreement counts."""

    def __init__(
        self,
        encoded_indices: Sequence[int | None],
        annotated_frames: Sequence[int],
        *,
        exemplar_index_origin: int,
    ):
        self._encoded_indices = encoded_indices
        self._exemplar_index_origin = exemplar_index_origin
        self._annotated_frames = frozenset(annotated_frames)
        self._best = None
        self._compared = 0
        self._disagreeing = 0

    def observe_counts(self, frame: int, xor_pixels: int, union_pixels: int) -> None:
        if not 0 <= frame < len(self._encoded_indices):
            raise ValueError("LTA suggestion frame lies outside the source volume")
        if not 0 <= xor_pixels <= union_pixels:
            raise ValueError("LTA suggestion disagreement exceeds directional mask union")
        if frame in self._annotated_frames or not union_pixels:
            return
        self._compared += 1
        if not xor_pixels:
            return
        self._disagreeing += 1
        if self._best is not None:
            best_frame, best_xor, best_union = self._best
            score = xor_pixels * best_union
            best_score = best_xor * union_pixels
            if score < best_score or (score == best_score and frame >= best_frame):
                return
        self._best = (frame, xor_pixels, union_pixels)

    def manifest_record(self) -> dict[str, object]:
        next_slice = None
        if self._best is not None:
            frame, xor_pixels, union_pixels = self._best
            encoded = self._encoded_indices[frame]
            next_slice = {
                "axis": "z",
                "slice_index": int(frame),
                "encoded_index": None if encoded is None else int(encoded),
                "exemplar_encoded_index": int(frame + self._exemplar_index_origin),
                "disagreement_pixels": int(xor_pixels),
                "union_pixels": int(union_pixels),
                "disagreement_fraction": xor_pixels / union_pixels,
            }
        return {
            "metric": "symmetric_difference_over_union",
            "comparison_scope": "jointly_observed_pixels",
            "slice_index_origin": 0,
            "compared_slice_count": self._compared,
            "disagreeing_slice_count": self._disagreeing,
            "next_slice": next_slice,
        }
