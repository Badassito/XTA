"""Packet integrity and common validation for archive and direct-object callers."""

from copy import deepcopy
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

import numpy as np

from XTA.registration.packet import (
    RegistrationPacket,
    SCHEMA,
    SCHEMA_VERSION,
    load_packet,
    write_packet,
)


def packet_metadata():
    forward = np.eye(4)
    forward[:3, 3] = [0.25, -1.0, 2.0]
    scan = dict(shape_zyx=[3, 5, 7], spacing_xyz=[1.0, 2.0, 3.0], spacing_units="mm")
    return dict(
        schema=SCHEMA,
        schema_version=SCHEMA_VERSION,
        pair="synthetic",
        status="validated",
        fixed=dict(scan, filename="earlier.mkv", sha256="a" * 64),
        moving=dict(scan, filename="later.mkv", sha256="b" * 64),
        mapping=dict(
            fixed_to_moving_xyz=forward.tolist(),
            moving_to_fixed_xyz=np.linalg.inv(forward).tolist(),
        ),
    )


class RegistrationPacketTests(unittest.TestCase):
    def setUp(self):
        root = Path(
            os.environ.get(
                "XTA_REGISTRATION_TEST_TEMP",
                str(
                    Path(__file__).resolve().parents[2]
                    / "Scratch/Experiments/Registration/xta_tooling_validation"
                ),
            )
        )
        root.mkdir(parents=True, exist_ok=True)
        self.work = tempfile.TemporaryDirectory(prefix="packet_test_", dir=root)
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)

    def test_archive_checksums_and_bidirectional_points(self):
        metadata = packet_metadata()
        mask = np.ones((5, 7), bool)
        mask[1, 2] = False
        path = write_packet(self.root / "map.zip", metadata, {"fixed": mask})
        packet = load_packet(path)
        np.testing.assert_array_equal(packet.mask("fixed"), mask)
        points = np.array([[0.0, 0.0, 0.0], [2.5, 3.2, 1.7]])
        np.testing.assert_allclose(
            packet.transform_points(packet.transform_points(points), "moving-to-fixed"), points
        )
        tampered = self.root / "tampered.zip"
        with ZipFile(path) as original, ZipFile(tampered, "w") as changed:
            for name in original.namelist():
                content = original.read(name)
                if name == "masks/fixed.npy":
                    content = content[:-1] + bytes([content[-1] ^ 1])
                changed.writestr(name, content)
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_packet(tampered)

    def test_direct_objects_validate_schema_inverse_masks_and_video(self):
        metadata = packet_metadata()
        bad = deepcopy(metadata)
        del bad["schema"]
        with self.assertRaisesRegex(ValueError, "schema"):
            RegistrationPacket(bad, {})
        bad = deepcopy(metadata)
        bad["mapping"]["moving_to_fixed_xyz"][0][3] += 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            RegistrationPacket(bad, {})
        for mask in (np.ones((5, 7), np.uint8), np.ones((5, 6), bool)):
            with self.assertRaisesRegex(ValueError, "boolean YX"):
                RegistrationPacket(metadata, {"fixed": mask})
        bad = deepcopy(metadata)
        bad["fixed"].update(observed_mask="masks/fixed.npy", observed_mask_sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "lacks its fixed"):
            RegistrationPacket(bad, {})
        bad = deepcopy(metadata)
        bad["registered_video"] = dict(
            output_filename="registered.mkv",
            output_sha256="c" * 64,
            shape_zyx=[3, 5, 7],
            source_role="moving",
            destination_role="fixed",
            source_sha256="b" * 64,
            output_to_input_xyz=np.eye(4).tolist(),
        )
        with self.assertRaisesRegex(ValueError, "sampling matrix"):
            RegistrationPacket(bad, {})

    def test_constructor_isolates_inputs_and_accessors_recheck_mutation(self):
        metadata = packet_metadata()
        masks = {"fixed": np.ones((5, 7), bool)}
        packet = RegistrationPacket(metadata, masks)
        metadata["mapping"]["fixed_to_moving_xyz"][0][3] = 123
        masks["fixed"][:] = False
        self.assertEqual(packet.fixed_to_moving_xyz[0, 3], 0.25)
        self.assertTrue(packet.mask("fixed").all())
        packet.metadata["mapping"]["fixed_to_moving_xyz"][0][3] = 99
        for read in (lambda: packet.usable, lambda: packet.matrix(), lambda: packet.shape("fixed")):
            with self.assertRaisesRegex(ValueError, "disagree"):
                read()
        packet = RegistrationPacket(packet_metadata(), {})
        packet.metadata["moving"].update(
            observed_mask="masks/moving.npy", observed_mask_sha256="c" * 64
        )
        with self.assertRaisesRegex(ValueError, "lacks its moving"):
            packet.mask("moving")

    def test_unavailable_packet_has_no_accessible_transform(self):
        metadata = packet_metadata()
        metadata.update(status="unavailable", mapping=None)
        packet = RegistrationPacket(metadata, {})
        self.assertFalse(packet.usable)
        with self.assertRaisesRegex(ValueError, "no validated"):
            packet.matrix()
        loaded = load_packet(write_packet(self.root / "unavailable.zip", metadata))
        self.assertFalse(loaded.usable)

    def test_no_overwrite_publication_preserves_concurrent_destination(self):
        target = self.root / "race.zip"
        metadata = packet_metadata()
        original_publish = os.rename if os.name == "nt" else os.link
        method = "rename" if os.name == "nt" else "link"

        def create_destination_then_publish(source, destination):
            Path(destination).write_bytes(b"another writer")
            return original_publish(source, destination)

        with mock.patch(
            "XTA.registration.packet.os." + method, side_effect=create_destination_then_publish
        ):
            with self.assertRaises(FileExistsError):
                write_packet(target, metadata, overwrite=False)
        self.assertEqual(target.read_bytes(), b"another writer")
        self.assertFalse(list(self.root.glob("*.partial.zip")))
        write_packet(target, metadata)
        self.assertTrue(load_packet(target).usable)
        fresh = write_packet(self.root / "new.zip", metadata, overwrite=False)
        self.assertTrue(load_packet(fresh).usable)


if __name__ == "__main__":
    unittest.main()
