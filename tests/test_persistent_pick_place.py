import tempfile
from pathlib import Path
import unittest
import numpy as np
import pybullet as p
from robotic_arm.environments.persistent_pick_place_env import PersistentPickPlaceEnv
from robotic_arm.utils.output_paths import prepare_output


class PersistentTests(unittest.TestCase):
    def test_task_switch_keeps_world_and_arm_positions(self):
        env=PersistentPickPlaceEnv()
        try:
            env.reset(seed=2000)
            ids=list(env.objects)
            positions=[p.getBasePositionAndOrientation(body,physicsClientId=env.physics_client) for body in ids]
            joints=[p.getJointState(env.arm,j,physicsClientId=env.physics_client)[0] for j in range(7)]
            with self.assertRaises(RuntimeError):env.next_task()
            env.stable_steps=10;env._done=True
            env.next_task()
            self.assertEqual(ids,env.objects)
            self.assertEqual(positions,[p.getBasePositionAndOrientation(body,physicsClientId=env.physics_client) for body in ids])
            np.testing.assert_allclose(joints,[p.getJointState(env.arm,j,physicsClientId=env.physics_client)[0] for j in range(7)])
            self.assertFalse(env.grasped)
            self.assertFalse(env.lifted)
        finally:env.close()

    def test_existing_reports_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            target=prepare_output(Path(folder)/"run")
            report=target/"summary.json";report.write_text("original")
            with self.assertRaises(ValueError):prepare_output(target)
            self.assertEqual(report.read_text(),"original")

    def test_camera_renders_rgb_without_changing_physics(self):
        env=PersistentPickPlaceEnv(render_mode="rgb_array")
        try:
            env.reset(seed=2000)
            before=env._observation().copy()
            frame=env.render()
            self.assertEqual(frame.shape,(240,320,3))
            self.assertEqual(frame.dtype,np.uint8)
            self.assertGreater(frame.std(),1)
            np.testing.assert_allclose(before,env._observation())
        finally:env.close()

    def test_shared_policy_completes_persistent_scene(self):
        from stable_baselines3 import PPO
        from robotic_arm.environments.scene_layouts import THREE_TASKS
        root=Path(__file__).resolve().parents[1]
        model=PPO.load(root/"models/panda_simulation_policy.zip",device="cpu")
        env=PersistentPickPlaceEnv()
        try:
            obs,_=env.reset(seed=56000)
            for task in range(3):
                for _ in range(env.max_episode_steps):
                    action,_=model.predict(obs,deterministic=True)
                    obs,_,t,tr,info=env.step(action)
                    if t or tr:break
                self.assertTrue(info["is_success"])
                self.assertTrue(info["grasped"])
                self.assertTrue(info["lifted"])
                self.assertTrue(info["released"])
                if task<2:obs,_=env.next_task()
            for body,(_,goal) in zip(env.objects,THREE_TASKS):
                position=p.getBasePositionAndOrientation(body,physicsClientId=env.physics_client)[0]
                self.assertLess(np.linalg.norm(np.array(position[:2])-goal),.05)
            self.assertEqual(p.getNumConstraints(physicsClientId=env.physics_client),0)
        finally:env.close()
