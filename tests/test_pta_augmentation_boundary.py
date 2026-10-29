from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


from XTA import pta_augmentation
from XTA import augmentation_policy


class PtaAugmentationBoundaryTests(unittest.TestCase):
    def test_owner_import_does_not_load_orchestration_or_accelerators(self) -> None:
        program = (
            "from tools.smoke_import import install_stubs; install_stubs(); "
            "import sys; import XTA.pta_augmentation; "
            "forbidden={'XTA.pta','XTA.pta_workers','XTA.pta_dataset'}; "
            "assert not (forbidden & set(sys.modules)), forbidden & set(sys.modules); "
            "roots={name.split('.')[0] for name in sys.modules}; "
            "assert not ({'torch','cupy','nvidia','albumentations'} & roots), roots"
        )
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", program],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "YOLO_TTA_TELEMETRY": "0"},
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_inspection_uses_the_shared_policy_owner(self) -> None:
        for name in (
            "AugmentationDefinition", "assert_augmentation_definition_unchanged",
            "inspect_augmentation_definition",
        ):
            with self.subTest(name=name):
                self.assertIs(
                    getattr(pta_augmentation, name), getattr(augmentation_policy, name),
                )


if __name__ == "__main__":
    unittest.main()
