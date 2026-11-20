from __future__ import annotations

import unittest


from XTA.cuda_d1 import (  # noqa: E402
    D1PartialBitsetArtifact,
    _normalize_slice_ranges,
)


class D1GroupProtocolTests(unittest.TestCase):
    def test_partial_artifact_round_trip_preserves_opaque_ipc_handle(self) -> None:
        artifact = D1PartialBitsetArtifact(
            group_id="d1g-model-view",
            model_name="model",
            view_name="view",
            participant_rank=1,
            participant_worker_id=3,
            output_shape=(3, 5, 7),
            word_count=4,
            covered_ranges=((2, 4), (8, 10)),
            lease_token="lease-1",
            transport="cuda_ipc",
            ipc_handle=b"opaque-handle",
        )

        restored = D1PartialBitsetArtifact.from_payload(artifact.to_payload())

        self.assertEqual(restored, artifact)
        self.assertEqual(restored.byte_count, 16)

    def test_partial_artifact_rejects_invalid_shape_words_and_transport(self) -> None:
        base = dict(
            group_id="group",
            model_name="model",
            view_name="view",
            participant_rank=0,
            participant_worker_id=0,
            output_shape=(1, 1, 33),
            word_count=2,
            covered_ranges=((0, 1),),
            lease_token="lease",
            transport="cuda_ipc",
            ipc_handle=b"handle",
        )
        with self.assertRaisesRegex(ValueError, "word count"):
            D1PartialBitsetArtifact(**{**base, "word_count": 1})
        with self.assertRaisesRegex(ValueError, "missing its memory handle"):
            D1PartialBitsetArtifact(**{**base, "ipc_handle": b""})
        with self.assertRaisesRegex(ValueError, "unsupported D1 partial transport"):
            D1PartialBitsetArtifact(**{**base, "transport": "peer-magic"})

    def test_slice_range_normalization_coalesces_and_rejects_overlap(self) -> None:
        self.assertEqual(
            _normalize_slice_ranges(((4, 8), (0, 2), (2, 4)), total_slices=8),
            ((0, 8),),
        )
        with self.assertRaisesRegex(ValueError, "overlap"):
            _normalize_slice_ranges(((0, 4), (3, 8)), total_slices=8)
        with self.assertRaisesRegex(ValueError, "exceeds depth"):
            _normalize_slice_ranges(((0, 9),), total_slices=8)


if __name__ == "__main__":
    unittest.main()
