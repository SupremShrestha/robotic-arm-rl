from pathlib import Path
import unittest
import yaml


class WorkflowTests(unittest.TestCase):
    def test_ci_configuration(self):
        data=yaml.safe_load(Path(".github/workflows/tests.yml").read_text())
        self.assertEqual(set(data["on"]),{"push","pull_request","workflow_dispatch"})
        self.assertEqual(data["permissions"],{"contents":"read"})
        job=data["jobs"]["tests"]
        self.assertEqual(set(job["strategy"]["matrix"]["os"]),{"ubuntu-latest","windows-latest"})
        commands=[step.get("run","") for step in job["steps"]]
        self.assertIn("python -m unittest discover -s tests -v",commands)
        self.assertIn("python scripts/check_training.py",commands)
