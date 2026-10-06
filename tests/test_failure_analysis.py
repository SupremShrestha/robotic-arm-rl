import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from robotic_arm.experiments.analyze_failures import analyze


class AnalysisTests(unittest.TestCase):
    def test_failure_report_and_replay_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{"seed":2000,"success":True,"ground_collision":False,"distance_m":0.04,"target_x":0.4,"target_y":0.1,"target_z":0.4},
                  {"seed":2001,"success":False,"ground_collision":True,"distance_m":0.4,"target_x":0.5,"target_y":0.2,"target_z":0.6}]
            with (root/"episodes.csv").open("w",newline="") as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                result=analyze(root/"episodes.csv",root/"analysis")
            self.assertEqual(result["failures"],1)
            self.assertEqual(result["failed_with_ground_contact"],1)
            self.assertEqual(result["worst_episode_seeds"],[2001])
            self.assertTrue((root/"analysis/targets.png").exists())
            self.assertIn("reaching_policy",(root/"analysis/report.md").read_text())
