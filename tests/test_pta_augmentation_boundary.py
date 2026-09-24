from __future__ import annotations

import inspect
import unittest


from XTA import pta
from XTA import pta_augmentation
from XTA import augmentation_policy


class PtaAugmentationBoundaryTests(unittest.TestCase):
    EXPORTED_NAMES = (
        "AugmentationDefinition",
        "LoadedAugmentation",
        "LoadedGpuAugmentation",
        "OfflineAugmentation",
        "_augmented_image_to_uint8",
        "_augmented_mask_to_binary",
        "_load_external_python_module",
        "apply_augmentation_pair",
        "assert_augmentation_definition_unchanged",
        "assert_augmentation_did_not_synthesize_mask",
        "inspect_augmentation_definition",
        "load_augmentation_definition",
        "load_gpu_augmentation_definition",
        "load_offline_augmentation_definition",
        "validate_seedable_augmentation_pipeline",
    )

    def test_pta_reexports_the_augmentation_owner_objects(self) -> None:
        self.assertEqual(pta_augmentation.__all__, self.EXPORTED_NAMES)
        shared_names = {
            "AugmentationDefinition", "assert_augmentation_definition_unchanged",
            "inspect_augmentation_definition",
        }
        for name in self.EXPORTED_NAMES:
            with self.subTest(name=name):
                owned = getattr(pta_augmentation, name)
                self.assertIs(getattr(pta, name), owned)
                if inspect.isfunction(owned) or inspect.isclass(owned):
                    owner = augmentation_policy if name in shared_names else pta_augmentation
                    self.assertIs(getattr(owner, name), owned)
                    self.assertEqual(owned.__module__, owner.__name__)


if __name__ == "__main__":
    unittest.main()
