"""Exercise the registration command through XTA's installed command dispatcher."""

import contextlib
import io
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

from XTA.cli import run


class RegistrationCommandTests(unittest.TestCase):
    def test_help_does_not_require_registration_dependencies(self):
        code = (
            "import sys; from XTA.cli import run\n"
            "try: run(['register', '--help'])\n"
            "except SystemExit as exc: assert exc.code == 0\n"
            "assert not {'numpy','scipy','SimpleITK','cv2','nrrd','torch'} & set(sys.modules)"
        )
        process = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True
        )
        self.assertIn("pairs", process.stdout)

    def test_unsupported_fit_has_distinct_exit_and_machine_readable_result(self):
        def unavailable(**options):
            self.assertEqual(options["exclude_first"], [3, 5])
            self.assertEqual(options["contrast"], "both")
            self.assertTrue(options["affine"])
            print("progress")
            return {"packet": str(options["output"]), "status": "unavailable"}

        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("XTA.registration.workflow.fit_movies", side_effect=unavailable),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as caught,
        ):
            run(
                [
                    "register", "fit", "--fixed", "a.mkv", "--moving", "b.mkv", "--output", "pair.zip",
                    "--exclude-first", "3", "5", "--contrast", "both", "--affine",
                ]
            )
        self.assertEqual(caught.exception.code, 3)
        self.assertEqual(
            json.loads(stdout.getvalue()), {"packet": "pair.zip", "status": "unavailable"}
        )
        self.assertEqual(stderr.getvalue().strip(), "progress")


if __name__ == "__main__":
    unittest.main()
