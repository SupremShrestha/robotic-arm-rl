import unittest
from unittest.mock import patch
import numpy as np
import pybullet as p
from stable_baselines3.common.env_checker import check_env
from environment.reach_env import ReachEnv


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = ReachEnv()

    def tearDown(self):
        self.env.close()

    def test_gym_contract(self):
        check_env(self.env, warn=True)

    def test_seed_and_client_isolation(self):
        obs1, _ = self.env.reset(seed=12)
        other = ReachEnv()
        try:
            obs2, _ = other.reset(seed=12)
            np.testing.assert_array_equal(obs1, obs2)
            for _ in range(3):
                self.env.step(np.ones(7, dtype=np.float32))
            np.testing.assert_array_equal(other._get_observation(), obs2)
        finally:
            other.close()

    def test_action_validation(self):
        self.env.reset(seed=1)
        for action in ([0]*6, [float("nan")]*7, [float("inf")]*7):
            with self.assertRaises(ValueError):
                self.env.step(action)

    def test_timeout_and_reset_guard(self):
        self.env.max_episode_steps = 1
        self.env.reset(seed=1)
        _, _, terminated, truncated, _ = self.env.step(np.zeros(7))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        with self.assertRaises(RuntimeError):
            self.env.step(np.zeros(7))

    def test_success_detection(self):
        obs, _ = self.env.reset(seed=1)
        self.env.target_position = obs[14:17].astype(float)
        _, reward, terminated, truncated, info = self.env.step(np.zeros(7))
        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertFalse(truncated)
        self.assertGreater(reward, 9)

    def test_motor_targets_within_limits(self):
        self.env.reset(seed=1)
        for j, upper in enumerate(self.env.joint_upper):
            p.resetJointState(self.env.arm_id, j, upper, physicsClientId=self.env.physics_client)
        with patch("environment.reach_env.p.setJointMotorControlArray", wraps=p.setJointMotorControlArray) as motor:
            self.env.step(np.ones(7))
            targets = np.array(motor.call_args.kwargs["targetPositions"])
        self.assertTrue(np.all(targets <= self.env.joint_upper))
        self.assertTrue(np.all(targets >= self.env.joint_lower))

    def test_render_and_close(self):
        env = ReachEnv(render_mode="rgb_array")
        try:
            env.reset(seed=1)
            frame = env.render()
            self.assertEqual(frame.shape, (480, 640, 3))
            self.assertEqual(frame.dtype, np.uint8)
        finally:
            env.close()
            env.close()


if __name__ == "__main__":
    unittest.main()
