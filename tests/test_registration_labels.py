"""Regressions for sparse, oblique, multi-class registration label transfer."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from XTA.registration.labels import _polygon_rows, transfer_yolo_labels
from XTA.registration.packet import RegistrationPacket


def make_packet(shape, matrix, source_mask=None, target_mask=None, usable=True):
    scan = dict(
        sha256="0" * 64, shape_zyx=list(shape), spacing_xyz=[1.0, 1.0, 1.0], spacing_units="voxel"
    )
    metadata = dict(
        schema="xta.registration.packet",
        schema_version=1,
        pair="test",
        status="validated" if usable else "unavailable",
        fixed=dict(scan, filename="earlier.mkv"),
        moving=dict(scan, filename="later.mkv"),
        mapping=(
            dict(
                fixed_to_moving_xyz=matrix.tolist(),
                moving_to_fixed_xyz=np.linalg.inv(matrix).tolist(),
            )
            if usable
            else None
        ),
    )
    masks = {
        role: mask
        for role, mask in [("fixed", source_mask), ("moving", target_mask)]
        if mask is not None
    }
    return RegistrationPacket(metadata, masks)


def write_polygon(path, class_id, points, shape):
    coordinates = np.asarray(points, dtype=float) / np.asarray(shape[:0:-1])
    path.write_text(
        str(class_id) + " " + " ".join(str(value) for value in coordinates.ravel()) + "\n",
        encoding="utf-8",
    )


class RegistrationLabelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = self.root / "source"
        self.labels.mkdir()

    def test_oblique_sparse_multiclass_transfer_matches_dense_reference(self):
        shape = (9, 15, 17)
        first = np.array([[3, 3], [10, 3], [10, 9], [3, 9]], np.int32)
        second = np.array([[4, 4], [12, 4], [12, 10], [4, 10]], np.int32)
        write_polygon(self.labels / "volume_0003.txt", 0, first, shape)
        (self.labels / "volume_0005.txt").write_text("", encoding="utf-8")
        write_polygon(self.labels / "volume_0006.txt", 7, second, shape)
        angle = np.deg2rad(32)
        rotation = np.array(
            [[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]]
        )
        matrix = np.eye(4)
        matrix[:3, :3] = rotation * 1.03
        center = np.array([8, 7, 4])
        matrix[:3, 3] = center - matrix[:3, :3] @ center
        source_mask = np.ones(shape[1:], bool)
        source_mask[6, 6] = False
        target_mask = np.ones(shape[1:], bool)
        target_mask[:, :4] = False
        packet = make_packet(shape, matrix, source_mask, target_mask)
        summary = transfer_yolo_labels(packet, self.labels, self.root / "result", workers=2)

        source = np.zeros((2, *shape), np.uint8)
        cv2.fillPoly(source[0, 2], [first], 1)
        cv2.fillPoly(source[1, 5], [second], 1)
        known = np.zeros(shape[0], bool)
        known[[2, 4, 5]] = True
        xyz = np.indices(shape)[::-1].reshape(3, -1)
        mapped = np.linalg.inv(matrix)[:3, :3] @ xyz + np.linalg.inv(matrix)[:3, 3, None]
        indexes = np.floor(mapped + 0.5).astype(int)
        valid = ((indexes >= 0) & (indexes < np.asarray(shape)[::-1, None])).all(axis=0)
        safe = np.clip(indexes, 0, np.asarray(shape)[::-1, None] - 1)
        valid &= known[safe[2]] & source_mask[safe[1], safe[0]] & target_mask[xyz[1], xyz[0]]
        expected = source[:, safe[2], safe[1], safe[0]] * valid
        expected = expected.reshape(2, *shape)
        expected_coverage = valid.reshape(shape)
        checked_pixels = 0
        for row in summary["slices"]:
            with np.load(self.root / "result" / row["exact_mask"]) as actual:
                x0, y0 = actual["origin_xy"]
                height, width = actual["foreground"].shape[1:]
                z = row["native_target_z"]
                np.testing.assert_array_equal(actual["class_ids"], [0, 7])
                np.testing.assert_array_equal(
                    actual["foreground"], expected[:, z, y0 : y0 + height, x0 : x0 + width]
                )
                np.testing.assert_array_equal(
                    actual["known_source_coverage"],
                    expected_coverage[z, y0 : y0 + height, x0 : x0 + width],
                )
                checked_pixels += int(actual["foreground"].sum())
        self.assertEqual(checked_pixels, int(expected.sum()))
        self.assertEqual(summary["source_known_planes"], [2, 4, 5])
        self.assertEqual(summary["source_empty_planes"], [4])
        self.assertEqual(summary["source_unknown_plane_count"], 6)
        self.assertGreater(summary["positive_target_planes"], 2)
        with np.load(self.root / "result" / summary["annotation_domain"]) as domain:
            np.testing.assert_array_equal(domain["source_known_planes"], known)
            np.testing.assert_array_equal(domain["source_observed_yx"], source_mask)
            np.testing.assert_array_equal(domain["target_observed_yx"], target_mask)

    def test_reverse_direction_and_zero_based_filenames(self):
        shape = (5, 14, 16)
        points = np.array([[6, 3], [11, 3], [11, 8], [6, 8]], np.int32)
        write_polygon(self.labels / "later_0002.txt", 2, points, shape)
        matrix = np.eye(4)
        matrix[0, 3] = 2
        packet = make_packet(shape, matrix)
        result = transfer_yolo_labels(
            packet,
            self.labels,
            self.root / "result",
            direction="moving_to_fixed",
            filename_index_base=0,
        )
        self.assertEqual(result["direction"], "moving-to-fixed")
        self.assertEqual(result["source_known_planes"], [2])
        self.assertEqual(result["slices"][0]["encoded_index_one_based"], 3)
        row = result["slices"][0]
        self.assertTrue(row["yolo_file"].endswith("earlier_0003.txt"))
        with np.load(self.root / "result" / row["exact_mask"]) as actual:
            x, y = actual["origin_xy"]
            reconstructed = np.zeros(shape[1:], np.uint8)
            mask = actual["foreground"][0]
            reconstructed[y : y + mask.shape[0], x : x + mask.shape[1]] = mask
        expected = np.zeros(shape[1:], np.uint8)
        cv2.fillPoly(expected, [points - [2, 0]], 1)
        np.testing.assert_array_equal(reconstructed, expected)

    def test_empty_annotation_keeps_known_background_without_fabricated_positives(self):
        (self.labels / "source_0002.txt").write_text("", encoding="utf-8")
        packet = make_packet((4, 8, 9), np.eye(4))
        result = transfer_yolo_labels(packet, self.labels, self.root / "result")
        self.assertEqual(result["source_known_planes"], [1])
        self.assertEqual(result["source_empty_planes"], [1])
        self.assertEqual(result["positive_target_planes"], 0)
        self.assertEqual(list((self.root / "result" / "partial_yolo_labels").glob("*.txt")), [])
        with np.load(self.root / "result" / result["annotation_domain"]) as domain:
            np.testing.assert_array_equal(
                domain["source_known_planes"], [False, True, False, False]
            )

    def test_polygon_export_reports_holes_and_single_pixels(self):
        mask = np.zeros((12, 12), np.uint8)
        mask[2:8, 2:8] = 1
        mask[4:6, 4:6] = 0
        mask[10, 10] = 1
        rows, stats = _polygon_rows(mask, (0, 0), mask.shape, 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["holes_in_exact_mask"], 1)
        self.assertEqual(stats["omitted_degenerate_components"], 1)
        self.assertEqual(stats["false_positive_pixels"], 4)
        self.assertEqual(stats["false_negative_pixels"], 1)

    def test_bad_labels_and_unavailable_packet_leave_no_output(self):
        packet = make_packet((4, 8, 9), np.eye(4))
        output = self.root / "result"
        (self.labels / "source_0002.txt").write_text("0 nan .2 .4 .2 .4 .4\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Invalid normalized"):
            transfer_yolo_labels(packet, self.labels, output)
        self.assertFalse(output.exists())
        unavailable = make_packet((4, 8, 9), np.eye(4), usable=False)
        with self.assertRaisesRegex(ValueError, "no validated"):
            transfer_yolo_labels(unavailable, self.labels, output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
