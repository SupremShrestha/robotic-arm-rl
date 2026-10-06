"""Seeded, joint-limited KUKA reaching environment (distances in metres)."""
from functools import wraps
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class SimulationDisconnected(RuntimeError):
    """The physics connection was closed, including by closing the GUI."""


def requires_connection(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if not p.isConnected(self.physics_client):
            raise SimulationDisconnected("Simulation window or physics connection is closed")
        try:
            return method(self, *args, **kwargs)
        except p.error as error:
            # The GUI can close between the initial check and a physics call.
            if not p.isConnected(self.physics_client):
                raise SimulationDisconnected("Simulation window or physics connection is closed") from error
            raise
    return guarded


class ReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 48}

    def __init__(self, render_mode=None, reward_mode="legacy", max_episode_steps=200):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode")
        if reward_mode not in ("legacy", "shaped"):
            raise ValueError("reward_mode must be legacy or shaped")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.render_mode, self.reward_mode = render_mode, reward_mode
        self.max_episode_steps = int(max_episode_steps)
        self.controllable_joints = list(range(7))
        self.end_effector_link_index = 6
        self.num_joints = 7
        self.max_action_delta, self.action_repeat = 0.05, 5
        self.success_threshold, self.success_bonus = 0.05, 10.0
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.physics_client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)
        self.arm_id = self.target_id = self.plane_id = None
        self.target_position = np.array([0.4, 0.3, 0.6])
        self.current_step = 0
        self._done = True

    @requires_connection
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        c = self.physics_client
        p.resetSimulation(physicsClientId=c)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        p.setGravity(0, 0, -9.8, physicsClientId=c)
        p.setTimeStep(1 / 240, physicsClientId=c)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=c)
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True, physicsClientId=c)
        joint_info = [p.getJointInfo(self.arm_id, j, physicsClientId=c) for j in self.controllable_joints]
        self.joint_lower = np.array([j[8] for j in joint_info])
        self.joint_upper = np.array([j[9] for j in joint_info])
        self.target_position = self._sample_random_target()
        if options and "target" in options:
            target = np.asarray(options["target"], dtype=float)
            if target.shape != (3,) or not np.all(np.isfinite(target)):
                raise ValueError("target must contain three finite coordinates")
            self.target_position = target.copy()
        shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0.15, 0.1, 1], physicsClientId=c)
        self.target_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=shape,
                                         basePosition=self.target_position.tolist(), physicsClientId=c)
        self.current_step, self._done = 0, False
        self.previous_distance = self._get_distance_to_target()
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(1.9, 45, -25, [0.2, 0, 0.65], physicsClientId=c)
        return self._get_observation(), self._info()

    @requires_connection
    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset before stepping a new episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (7,) or not np.all(np.isfinite(action)):
            raise ValueError("action must contain seven finite values")
        action = np.clip(action, -1, 1)
        positions = np.array([p.getJointState(self.arm_id, j, physicsClientId=self.physics_client)[0]
                              for j in self.controllable_joints])
        targets = np.clip(positions + action * self.max_action_delta, self.joint_lower, self.joint_upper)
        p.setJointMotorControlArray(self.arm_id, self.controllable_joints, p.POSITION_CONTROL,
                                   targetPositions=targets.tolist(), forces=[300] * 7,
                                   physicsClientId=self.physics_client)
        collision = False
        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)
            collision = collision or self._ground_collision()
        self.current_step += 1
        info = self._info()
        info["ground_collision"] = bool(collision)
        distance = info["distance"]
        reward = -distance
        if self.reward_mode == "shaped":
            reward += 10 * (self.previous_distance - distance) - 0.001 * float(np.square(action).sum())
            reward -= float(collision)
        if info["is_success"]:
            reward += self.success_bonus
        self.previous_distance = distance
        terminated = info["is_success"]
        truncated = bool(self.current_step >= self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        return self._get_observation(), float(reward), terminated, truncated, info

    def _ground_collision(self):
        return any(contact[3] > 0 for contact in p.getContactPoints(
            self.arm_id, self.plane_id, physicsClientId=self.physics_client))

    def _info(self):
        distance = float(self._get_distance_to_target())
        return {"distance": distance, "is_success": bool(distance < self.success_threshold),
                "ground_collision": bool(self._ground_collision()), "steps": self.current_step,
                "target": self.target_position.tolist()}

    def _get_observation(self):
        states = p.getJointStates(self.arm_id, self.controllable_joints, physicsClientId=self.physics_client)
        end = p.getLinkState(self.arm_id, 6, physicsClientId=self.physics_client)[0]
        return np.array([s[0] for s in states] + [s[1] for s in states] + list(end)
                        + self.target_position.tolist(), dtype=np.float32)

    def _get_distance_to_target(self):
        end = p.getLinkState(self.arm_id, 6, physicsClientId=self.physics_client)[0]
        return float(np.linalg.norm(np.asarray(end) - self.target_position))

    def _sample_random_target(self):
        # Preserve the original distribution for honest checkpoint comparisons.
        return self.np_random.uniform([0.3, -0.4, 0.3], [0.6, 0.4, 0.7])

    @requires_connection
    def ik_action(self):
        """Classical baseline: IK plus bounded incremental motor commands, not RL."""
        angles = p.calculateInverseKinematics(self.arm_id, 6, self.target_position.tolist(),
                    lowerLimits=self.joint_lower.tolist(), upperLimits=self.joint_upper.tolist(),
                    jointRanges=(self.joint_upper - self.joint_lower).tolist(),
                    restPoses=[0] * 7, maxNumIterations=100, residualThreshold=0.001,
                    physicsClientId=self.physics_client)
        current = self._get_observation()[:7]
        return np.clip((np.asarray(angles[:7]) - current) / self.max_action_delta, -1, 1).astype(np.float32)

    @requires_connection
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        view = p.computeViewMatrixFromYawPitchRoll([0.2, 0, 0.65], 1.9, 45, -25, 0, 2)
        projection = p.computeProjectionMatrixFOV(60, 640 / 480, 0.1, 10)
        pixels = p.getCameraImage(640, 480, viewMatrix=view, projectionMatrix=projection,
                                 renderer=p.ER_TINY_RENDERER, physicsClientId=self.physics_client)[2]
        return np.asarray(pixels, dtype=np.uint8).reshape(480, 640, 4)[:, :, :3]

    def close(self):
        if p.isConnected(self.physics_client):
            try:
                p.disconnect(self.physics_client)
            except p.error:
                if p.isConnected(self.physics_client):
                    raise
        self._done = True
