"""The cleanup writer must publish the exact native mask PTA will consume."""
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools import xta_native_filter as cleanup
from XTA import pta


class NativeCleanupPtaTests(unittest.TestCase):
    def test_native_label_discovery_and_voxels_survive_pta_import(self):
        mask = np.zeros((9, 13, 17), np.uint8)
        mask[2:7, 3:10, 4:13] = 1
        mask[4, 5:8, 6:10] = 0
        mask[6, 9, 13:16] = 1  # A one-voxel appendage must survive the handoff.
        with tempfile.TemporaryDirectory(prefix='native-cleanup-pta-') as folder:
            root = Path(folder)
            label = root / 'sample.nrrd'
            cleanup.write_nrrd(label, mask.shape, cleanup.array_slabs(mask, 2), 'Global')
            for index, plane in enumerate(mask, 1):
                self.assertTrue(cv2.imwrite(str(root / f'sample_{index:04d}.png'), plane * 127))
            warnings = pta.WarningLog()
            specs = pta.discover_volume_specs(root, force=False, warnings=warnings)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].segmentation_nrrd_path, label)
            self.assertEqual(specs[0].label_source, 'nrrd')
            actual = pta.load_nrrd_mask_for_volume(label, mask.shape, warnings=warnings)
            np.testing.assert_array_equal(actual, mask)


if __name__ == '__main__':
    unittest.main()
