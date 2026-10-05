import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pybullet as p
from environment.reach_env import ReachEnv
from evaluate import evaluate


class EvaluationTests(unittest.TestCase):
    def test_reproducible_ik_report(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            first, rows = evaluate(controller="ik", episodes=2, seed=0, output=directory)
            second, rows2 = evaluate(controller="ik", episodes=2, seed=0)
            self.assertEqual(rows, rows2)
            self.assertEqual(first["successes"], 2)
            self.assertEqual(json.loads((Path(directory)/"summary.json").read_text()), first)
            self.assertTrue((Path(directory)/"episodes.csv").is_file())
            self.assertTrue((Path(directory)/"distances.png").is_file())

    def test_bad_episode_count(self):
        with self.assertRaises(ValueError):
            evaluate(episodes=0)

    def test_missing_checkpoint(self):
        with self.assertRaises(FileNotFoundError):
            evaluate(model_path="models/does_not_exist.zip")

    def test_window_closed_before_step(self):
        env = ReachEnv()
        original_step = env.step

        def disconnect_then_step(action):
            p.disconnect(env.physics_client)
            return original_step(action)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as log:
            with patch("evaluate.ReachEnv", return_value=env), patch.object(env, "step", side_effect=disconnect_then_step):
                summary, rows = evaluate(controller="random", episodes=2, render=True, output=directory)
            self.assertIsNone(summary)
            self.assertEqual(rows, [])
            self.assertIn("0 completed episodes", log.getvalue())
            self.assertFalse((Path(directory)/"summary.json").exists())
            self.assertFalse(p.isConnected(env.physics_client))

    def test_window_closed_between_episodes(self):
        env = ReachEnv(max_episode_steps=1)

        def close_window(_seconds):
            if p.isConnected(env.physics_client):
                p.disconnect(env.physics_client)

        with contextlib.redirect_stdout(io.StringIO()), patch("evaluate.ReachEnv", return_value=env), patch("evaluate.time.sleep", side_effect=close_window):
            summary, rows = evaluate(controller="random", episodes=2, render=True)
        self.assertIsNone(summary)
        self.assertEqual(len(rows), 1)

    def test_unrelated_physics_error_not_hidden(self):
        env = ReachEnv()
        try:
            env.reset(seed=0)
            with patch("environment.reach_env.p.getJointState", side_effect=p.error("Unexpected physics failure")):
                with self.assertRaises(p.error):
                    env.step([0]*7)
        finally:
            env.close()

    def test_disconnect_during_physics_call(self):
        env = ReachEnv()
        env.reset(seed=0)

        def disconnect_during_call(*args, **kwargs):
            p.disconnect(env.physics_client)
            raise p.error("Not connected to physics server.")

        with contextlib.redirect_stdout(io.StringIO()), patch("evaluate.ReachEnv", return_value=env), patch("environment.reach_env.p.getJointState", side_effect=disconnect_during_call):
            summary, rows = evaluate(controller="random", episodes=1, render=True)
        self.assertIsNone(summary)
        self.assertEqual(rows, [])
