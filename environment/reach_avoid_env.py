import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class ReachAvoidEnv(gym.Env):
    """
    Like ReachEnv, but with a spherical obstacle placed randomly between
    the arm and the target each episode. The agent must reach the target
    while staying clear of the obstacle.
    """

    def __init__(self, render_mode=None, obs_mode="relvec", success_threshold=0.05):
        super().__init__()

        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.controllable_joints = [0, 1, 2, 3, 4, 5, 6]
        self.end_effector_link_index = 6
        self.num_joints = len(self.controllable_joints)

        self.max_action_delta = 0.05
        self.action_repeat = 5
        self.success_threshold = success_threshold
        self.success_bonus = 10.0
        self.obstacle_radius = 0.08
        self.collision_penalty = 5.0
        self.min_target_obstacle_sep = 0.15
        self.max_episode_steps = 200
        self.current_step = 0

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )

        obs_dim = self.num_joints * 2 + 3 + 3 + 3
        if self.obs_mode == "relvec":
            obs_dim += 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        self.arm_id = None
        self.target_id = None
        self.obstacle_id = None
        self.plane_id = None
        self.target_position = np.array([0.4, 0.3, 0.6])
        self.obstacle_position = np.array([0.3, 0.0, 0.5])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.arm_id = p.loadURDF(
            "kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True,
            physicsClientId=self.physics_client,
        )

        self.target_position = self._sample_random_target()
        self.obstacle_position = self._sample_obstacle_position(self.target_position)

        target_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=(1, 0, 0, 1),
            physicsClientId=self.physics_client,
        )
        self.target_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=target_visual,
            basePosition=self.target_position.tolist(),
            physicsClientId=self.physics_client,
        )

        obstacle_visual = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=self.obstacle_radius, rgbaColor=(0.6, 0.6, 0.6, 0.9),
            physicsClientId=self.physics_client,
        )
        self.obstacle_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=obstacle_visual,
            basePosition=self.obstacle_position.tolist(),
            physicsClientId=self.physics_client,
        )

        self.current_step = 0
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_deltas = action * self.max_action_delta

        for i, joint_index in enumerate(self.controllable_joints):
            current_angle = p.getJointState(
                self.arm_id, joint_index, physicsClientId=self.physics_client
            )[0]
            target_angle = current_angle + joint_deltas[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id, jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL, targetPosition=target_angle,
                force=300, physicsClientId=self.physics_client,
            )

        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)

        observation = self._get_observation()
        distance = self._get_distance_to_target()
        obstacle_distance = self._get_distance_to_obstacle()

        reward = -distance
        success = distance < self.success_threshold
        collision = obstacle_distance < self.obstacle_radius

        if success:
            reward += self.success_bonus
        if collision:
            reward -= self.collision_penalty

        terminated = bool(success or collision)
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps
        info = {
            "is_success": success,
            "is_collision": collision,
            "distance": distance,
            "obstacle_distance": obstacle_distance,
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        joint_positions = []
        joint_velocities = []
        for joint_index in self.controllable_joints:
            joint_state = p.getJointState(
                self.arm_id, joint_index, physicsClientId=self.physics_client
            )
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])

        link_state = p.getLinkState(
            self.arm_id, self.end_effector_link_index,
            physicsClientId=self.physics_client,
        )
        end_effector_position = link_state[0]

        obs = (
            joint_positions
            + joint_velocities
            + list(end_effector_position)
            + list(self.target_position)
            + list(self.obstacle_position)
        )
        if self.obs_mode == "relvec":
            relative_vector = np.array(self.target_position) - np.array(end_effector_position)
            obs += list(relative_vector)
        return np.array(obs, dtype=np.float32)

    def _get_distance_to_target(self):
        link_state = p.getLinkState(
            self.arm_id, self.end_effector_link_index,
            physicsClientId=self.physics_client,
        )
        end_effector_position = np.array(link_state[0])
        return np.linalg.norm(end_effector_position - self.target_position)

    def _get_distance_to_obstacle(self):
        link_state = p.getLinkState(
            self.arm_id, self.end_effector_link_index,
            physicsClientId=self.physics_client,
        )
        end_effector_position = np.array(link_state[0])
        return np.linalg.norm(end_effector_position - self.obstacle_position)

    def _sample_random_target(self):
        x = self.np_random.uniform(0.3, 0.6)
        y = self.np_random.uniform(-0.4, 0.4)
        z = self.np_random.uniform(0.3, 0.7)
        return np.array([x, y, z])

    def _sample_obstacle_position(self, target_position):
        for _ in range(50):
            x = self.np_random.uniform(0.25, 0.5)
            y = self.np_random.uniform(-0.3, 0.3)
            z = self.np_random.uniform(0.3, 0.6)
            candidate = np.array([x, y, z])
            if np.linalg.norm(candidate - target_position) > self.min_target_obstacle_sep:
                return candidate
        return candidate

    def close(self):
        if p.isConnected(physicsClientId=self.physics_client):
            p.disconnect(physicsClientId=self.physics_client)
