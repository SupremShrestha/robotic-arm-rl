import unittest
import numpy as np
from stable_baselines3.common.env_checker import check_env
from robotic_arm.environments.pick_place_env import PickPlaceEnv


class PickPlaceRLTests(unittest.TestCase):
    def test_contract_seed_and_physical_object(self):
        env=PickPlaceEnv(max_episode_steps=2)
        try:
            check_env(env,warn=True)
            first,_=env.reset(seed=9)
            second,_=env.reset(seed=9)
            np.testing.assert_allclose(first,second)
            import pybullet as p
            self.assertEqual(p.getNumConstraints(physicsClientId=env.physics_client),0)
            self.assertGreater(p.getDynamicsInfo(env.cube,-1,physicsClientId=env.physics_client)[0],0)
            _,_,terminated,truncated,info=env.step(np.zeros(4,dtype=np.float32))
            self.assertFalse(info["is_success"])
            _,_,terminated,truncated,_=env.step(np.zeros(4,dtype=np.float32))
            self.assertTrue(truncated)
            with self.assertRaises(RuntimeError): env.step(np.zeros(4))
        finally:
            env.close()

    def test_invalid_action_and_stage(self):
        with self.assertRaises(ValueError): PickPlaceEnv(stage="unknown")
        env=PickPlaceEnv()
        try:
            env.reset(seed=1)
            with self.assertRaises(ValueError): env.step([0,0,float("nan"),0])
        finally:
            env.close()

    def test_history_without_contact_does_not_pay_or_succeed(self):
        from unittest.mock import patch
        env=PickPlaceEnv(stage="lift",max_episode_steps=20)
        try:
            env.reset(seed=1)
            env.grasped=True
            env.best_reach_distance=0
            # Previously grasped, now airborne without either finger: not a valid held lift.
            state=(np.array([.45,-.15,.18]),np.array([.45,-.15,.18]),np.zeros(3),[.04,.04],[])
            with patch.object(env,"_state",return_value=state), patch.object(env,"_advance"):
                for _ in range(10):
                    _,reward,terminated,_,info=env.step(np.zeros(4))
                    self.assertLess(reward,0)
                    self.assertFalse(info["current_grasp"])
                    self.assertFalse(info["is_success"])
                    self.assertFalse(terminated)
        finally:
            env.close()

    def test_near_cube_reset_is_not_an_assisted_grasp(self):
        env=PickPlaceEnv(stage="lift",randomize=False,start_near_cube=True)
        try:
            _,info=env.reset(seed=1)
            self.assertFalse(info["grasped"])
            self.assertFalse(info["lifted"])
            self.assertLess(info["reach_distance_m"],.1)
            self.assertGreater(info["finger_opening_m"],.025)
        finally:
            env.close()

    def test_repeated_contact_does_not_earn_grasp_bonus(self):
        from unittest.mock import patch
        env=PickPlaceEnv(stage="lift")
        try:
            env.reset(seed=1)
            env.best_reach_distance=0
            env.grasp_bonus_paid=True
            contact=(0,0,0,9), (0,0,0,10)
            state=(np.array([.45,-.15,.035]),np.array([.45,-.15,.025]),np.zeros(3),[.025,.025],contact)
            with patch.object(env,"_state",return_value=state), patch.object(env,"_advance"):
                _,reward,_,_,_=env.step(np.zeros(4))
                self.assertLess(reward,0)
        finally:
            env.close()

    def test_ground_sliding_penalty_and_termination(self):
        from unittest.mock import patch
        env=PickPlaceEnv(stage="lift")
        try:
            env.reset(seed=1)
            env.best_reach_distance=0
            before=(np.array([.45,-.15,.035]),np.array([.45,-.15,.025]),np.zeros(3),[.04,.04],[])
            after=(np.array([.55,-.15,.035]),np.array([.55,-.15,.025]),np.zeros(3),[.04,.04],[])
            with patch.object(env,"_state",side_effect=[before,before,after,after,after]), patch.object(env,"_advance"):
                _,reward,terminated,_,info=env.step(np.zeros(4))
            self.assertTrue(terminated)
            self.assertTrue(info["sliding_failure"])
            self.assertFalse(info["is_success"])
            self.assertLess(reward,-5)
        finally:
            env.close()

    def test_feedback_teacher_lifts_without_attachments(self):
        from robotic_arm.training.learn_pick_place import collect
        import pybullet as p
        env=PickPlaceEnv(stage="lift",randomize=True,start_near_cube=True)
        try:
            observations,actions,successes=collect(env,[42,43])
            self.assertEqual(successes,2)
            self.assertEqual(observations.shape[1],24)
            self.assertEqual(actions.shape[1],4)
            self.assertEqual(p.getNumConstraints(physicsClientId=env.physics_client),0)
        finally:
            env.close()

    def test_feedback_teacher_completes_physical_placement(self):
        from robotic_arm.training.learn_pick_place import collect
        import pybullet as p
        env=PickPlaceEnv(stage="place",randomize=True)
        try:
            _,_,successes=collect(env,[42,45,46])
            self.assertEqual(successes,3)
            info=env._info()
            self.assertTrue(info["lifted"])
            self.assertTrue(info["released"])
            self.assertTrue(info["resting"])
            self.assertLess(info["placement_error_m"],.05)
            self.assertEqual(p.getNumConstraints(physicsClientId=env.physics_client),0)
        finally:
            env.close()

    def test_scene_configuration_and_seeded_randomization(self):
        env=PickPlaceEnv(domain_randomization=True)
        try:
            first,info=env.reset(seed=123)
            second,repeat=env.reset(seed=123)
            np.testing.assert_allclose(first,second)
            for key in ("size","mass","friction","object_shape"):
                self.assertEqual(info[key],repeat[key])
            _,info=env.reset(seed=3,options={"size":.06,"mass":.15,"friction":.5,"object_shape":"cylinder","obstacle_height":.24})
            self.assertEqual(info["size"],.06)
            self.assertIsNotNone(env.obstacle)
            self.assertAlmostEqual(env.goal[2],.03)
            with self.assertRaises(ValueError): env.reset(options={"mass":-1})
            with self.assertRaises(ValueError): env.reset(options={"unknown":1})
        finally:
            env.close()

    def test_normalized_policy_transform_survives_checkpoint(self):
        import tempfile
        from pathlib import Path
        from stable_baselines3 import PPO
        from robotic_arm.training.learn_pick_place import StandardizedObservation
        env=PickPlaceEnv()
        try:
            obs,_=env.reset(seed=1)
            kwargs={"features_extractor_class":StandardizedObservation,
                    "features_extractor_kwargs":{"mean":[.1]*24,"scale":[.05]*24},"net_arch":[16]}
            model=PPO("MlpPolicy",env,n_steps=32,batch_size=16,policy_kwargs=kwargs,device="cpu")
            first,_=model.predict(obs,deterministic=True)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"policy.zip"
                model.save(path)
                loaded=PPO.load(path,device="cpu")
                second,_=loaded.predict(obs,deterministic=True)
            np.testing.assert_allclose(first,second)
        finally:
            env.close()

    def test_obstacle_schema_and_collision_failure(self):
        from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
        env=ObstaclePickPlaceEnv(stage="place")
        try:
            obs,_=env.reset(seed=42,options={"size":.06,"obstacle_height":.24})
            self.assertEqual(obs.shape,(32,))
            self.assertAlmostEqual(float(obs[30]),.06,places=5)
            self.assertAlmostEqual(float(obs[29]),.12,places=5)
            env.obstacle_contact=True
            _,reward,terminated,truncated,info=env.step(np.zeros(4))
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["is_success"])
            self.assertTrue(info["collision_failure"])
            self.assertLess(reward,-19)
        finally:
            env.close()

    def test_obstacle_teacher_places_large_object_without_contact(self):
        from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
        from robotic_arm.training.learn_pick_place import expert_action
        env=ObstaclePickPlaceEnv(stage="place")
        try:
            obs,_=env.reset(seed=42,options={"size":.06,"obstacle_height":.24})
            for _ in range(env.max_episode_steps):
                obs,_,t,tr,info=env.step(expert_action(obs,"place",.34))
                if t or tr: break
            self.assertTrue(info["is_success"])
            self.assertFalse(info["obstacle_contact"])
            self.assertTrue(info["released"])
        finally:
            env.close()

    def test_extended_normalized_policy_checkpoint_and_gym_contract(self):
        import tempfile
        from pathlib import Path
        from stable_baselines3 import PPO
        from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
        from robotic_arm.training.learn_pick_place import StandardizedObservation
        env=ObstaclePickPlaceEnv(max_episode_steps=2)
        try:
            check_env(env,warn=True)
            obs,_=env.reset(seed=1)
            model=PPO("MlpPolicy",env,n_steps=32,batch_size=16,device="cpu",
                policy_kwargs={"features_extractor_class":StandardizedObservation,
                "features_extractor_kwargs":{"mean":[.1]*32,"scale":[.05]*32},"net_arch":[16]})
            first,_=model.predict(obs,deterministic=True)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/"policy.zip"
                model.save(path)
                loaded=PPO.load(path,device="cpu")
                self.assertEqual(loaded.observation_space.shape,(32,))
                second,_=loaded.predict(obs,deterministic=True)
            np.testing.assert_allclose(first,second)
        finally:
            env.close()

    def test_large_object_release_opens_before_withdrawing(self):
        from robotic_arm.training.learn_pick_place import expert_action
        obs=np.zeros(32,dtype=np.float32)
        obs[:3]=[.45,.2,.04]
        obs[3:6]=[.45,.2,.03]
        obs[6:9]=[.45,.2,.03]
        obs[18:20]=.03
        obs[20:24]=1
        obs[30]=.06
        action=expert_action(obs,"place",.34)
        np.testing.assert_allclose(action[:3],0)
        self.assertEqual(action[3],1)
        obs[18:20]=.04
        withdrawn=expert_action(obs,"place",.34)
        self.assertGreater(withdrawn[2],0)
        self.assertEqual(withdrawn[3],1)

    def test_quarter_turn_teacher_regresses_six_failed_scenes(self):
        import pybullet as p
        from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
        from robotic_arm.training.learn_pick_place import expert_action
        env=OrientedPickPlaceEnv(stage="place")
        try:
            for seed in (2000,2002,2005,2006,2010,2017):
                with self.subTest(seed=seed):
                    obs,_=env.reset(seed=seed,options={"size":.06,"obstacle_height":.24})
                    for _ in range(env.max_episode_steps):
                        obs,_,t,tr,info=env.step(expert_action(obs,"place",.34))
                        if t or tr: break
                    self.assertTrue(info["is_success"])
                    self.assertFalse(info["obstacle_contact"])
                    self.assertTrue(info["released"])
                    self.assertTrue(info["resting"])
                    self.assertEqual(p.getNumConstraints(physicsClientId=env.physics_client),0)
        finally:
            env.close()

    def test_release_features_preserve_existing_predictions(self):
        from stable_baselines3 import PPO
        from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
        from robotic_arm.training.learn_pick_place import StandardizedObservation
        from robotic_arm.training.refine_obstacle_policy import add_release_features
        env=ObstaclePickPlaceEnv()
        try:
            obs,_=env.reset(seed=1)
            old=PPO("MlpPolicy",env,n_steps=32,batch_size=16,device="cpu",
                policy_kwargs={"net_arch":[256,256],"features_extractor_class":StandardizedObservation,
                "features_extractor_kwargs":{"mean":[0.]*32,"scale":[1.]*32}})
            new=add_release_features(old,env)
            np.testing.assert_allclose(old.predict(obs,deterministic=True)[0],new.predict(obs,deterministic=True)[0],atol=1e-7)
        finally:
            env.close()

    def test_varied_scene_geometry_and_seed_reproducibility(self):
        import pybullet as p
        from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
        env=OrientedPickPlaceEnv(scene_randomization=True)
        try:
            first,info=env.reset(seed=123)
            second,repeat=env.reset(seed=123)
            np.testing.assert_allclose(first,second)
            self.assertEqual(info["object_yaw"],repeat["object_yaw"])
            obs,info=env.reset(seed=1,options={"source_xy":[.36,-.22],"destination_xy":[.52,.24],
                "object_yaw":.7,"obstacle_height":.16,"obstacle_offset_xy":[.02,-.01],"obstacle_half_xy":[.03,.045]})
            np.testing.assert_allclose(env.goal[:2],[.52,.24])
            np.testing.assert_allclose(obs[27:30],[.03,.045,.08])
            np.testing.assert_allclose(obs[24:26],np.array([.44,.01])+[.02,-.01],atol=1e-6)
            yaw=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(env.cube,physicsClientId=env.physics_client)[1])[2]
            self.assertAlmostEqual(yaw,.7,places=3)
            for bad in ({"object_yaw":float("nan")},{"source_xy":[2,0]},{"obstacle_half_xy":[0,.04]}):
                with self.assertRaises(ValueError): env.reset(options=bad)
        finally:
            env.close()

    def test_yaw_alignment_preserves_default_and_wrist_limits(self):
        from robotic_arm.environments.yaw_aligned_pick_place_env import YawAlignedPickPlaceEnv
        from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
        aligned=YawAlignedPickPlaceEnv()
        original=OrientedPickPlaceEnv()
        try:
            first,_=aligned.reset(seed=42)
            second,_=original.reset(seed=42)
            np.testing.assert_allclose(first,second,atol=1e-7)
            np.testing.assert_allclose(aligned._orientation(),original._orientation(),atol=1e-8)
            for yaw in np.linspace(-np.pi,np.pi,17):
                aligned.scene_config["object_yaw"]=yaw
                angle=aligned._gripper_yaw()
                self.assertLessEqual(abs(.8+angle),2.8973)
                self.assertAlmostEqual(np.sin(2*(angle-np.pi/2-yaw)),0,places=6)
        finally:
            aligned.close()
            original.close()

    def test_cross_workspace_positions_are_separated_and_seeded(self):
        from robotic_arm.environments.scene_layouts import sample_cross_workspace
        for seed in range(100):
            a=sample_cross_workspace(np.random.default_rng(seed))
            b=sample_cross_workspace(np.random.default_rng(seed))
            self.assertEqual(a,b)
            source=np.array(a["source_xy"]);goal=np.array(a["destination_xy"])
            self.assertGreaterEqual(abs(source[0]-goal[0]),.10-1e-8)
            self.assertGreaterEqual(np.linalg.norm(source-goal),.31)
            self.assertLess(source[1],0)
            self.assertGreater(goal[1],0)
