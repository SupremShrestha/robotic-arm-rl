import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class PickPlaceEnv(gym.Env):
    """
    Franka Panda picks up a cube from a random position on a table and
    places it at a random goal position. Dense, phase-based reward:
    reach -> grasp -> carry -> place.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.arm_joints = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joints = [9, 10]
        self.grasp_frame_link = 11  # panda_grasptarget_hand
        self.num_arm_joints = len(self.arm_joints)

        self.max_action_delta = 0.05
        self.max_finger_delta = 0.01
        self.action_repeat = 5
        self.max_episode_steps = 300
        self.current_step = 0

        self.grasp_contact_threshold = 0.035
        self.lift_height = 0.05
        self.place_threshold = 0.04
        self.grasp_bonus = 5.0
        self.place_bonus = 20.0
        self.drop_penalty = 15.0
        self.holding_bonus = 0.05
        self.prev_dist_cube_goal = None

        self.table_height = 0.62
        self.cube_half_extent = 0.02

        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )

        obs_dim = 7 + 7 + 2 + 3 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        self.panda_id = None
        self.table_id = None
        self.cube_id = None
        self.goal_marker_id = None
        self.cube_position = np.array([0.5, 0.0, 0.68])
        self.goal_position = np.array([0.5, 0.2, 0.68])
        self.has_been_grasped = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation(physicsClientId=self.physics_client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.physics_client)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.physics_client
        )

        p.loadURDF("plane.urdf", physicsClientId=self.physics_client)
        self.table_id = p.loadURDF(
            "table/table.urdf", [0.5, 0, 0], useFixedBase=True,
            physicsClientId=self.physics_client,
        )
        self.panda_id = p.loadURDF(
            "franka_panda/panda.urdf", [0, 0, self.table_height], useFixedBase=True,
            physicsClientId=self.physics_client,
        )

        for j in self.finger_joints:
            p.resetJointState(self.panda_id, j, 0.04, physicsClientId=self.physics_client)

        self.cube_position = self._sample_cube_position()
        self.cube_id = p.loadURDF(
            "cube_small.urdf",
            [self.cube_position[0], self.cube_position[1], self.table_height + 0.06],
            physicsClientId=self.physics_client,
        )

        self.goal_position = self._sample_goal_position(self.cube_position)
        goal_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.03, length=0.002,
            rgbaColor=(0, 1, 0, 0.6), physicsClientId=self.physics_client,
        )
        self.goal_marker_id = p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=goal_visual,
            basePosition=[self.goal_position[0], self.goal_position[1], self.table_height + 0.001],
            physicsClientId=self.physics_client,
        )

        for _ in range(60):
            p.stepSimulation(physicsClientId=self.physics_client)

        self.current_step = 0
        self.has_been_grasped = False
        self.prev_dist_cube_goal = None
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        arm_deltas = action[:7] * self.max_action_delta
        gripper_command = action[7]

        for i, joint_index in enumerate(self.arm_joints):
            current_angle = p.getJointState(
                self.panda_id, joint_index, physicsClientId=self.physics_client
            )[0]
            target_angle = current_angle + arm_deltas[i]
            p.setJointMotorControl2(
                bodyUniqueId=self.panda_id, jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL, targetPosition=target_angle,
                force=300, physicsClientId=self.physics_client,
            )

        finger_target = 0.02 + gripper_command * 0.02
        finger_target = float(np.clip(finger_target, 0.0, 0.04))
        for j in self.finger_joints:
            p.setJointMotorControl2(
                bodyUniqueId=self.panda_id, jointIndex=j,
                controlMode=p.POSITION_CONTROL, targetPosition=finger_target,
                force=40, physicsClientId=self.physics_client,
            )

        for _ in range(self.action_repeat):
            p.stepSimulation(physicsClientId=self.physics_client)

        observation = self._get_observation()
        gripper_pos = self._get_gripper_position()
        cube_pos = self._get_cube_position()

        dist_gripper_cube = np.linalg.norm(gripper_pos - cube_pos)
        dist_cube_goal = np.linalg.norm(cube_pos - self.goal_position)
        is_grasping = self._check_grasping()
        cube_lifted = (cube_pos[2] - self.table_height) > self.lift_height

        reward = 0.0
        if not self.has_been_grasped:
            reward = -dist_gripper_cube
            if is_grasping and cube_lifted:
                reward += self.grasp_bonus
                self.has_been_grasped = True
                self.prev_dist_cube_goal = dist_cube_goal
        else:
            if self.prev_dist_cube_goal is not None:
                reward = (self.prev_dist_cube_goal - dist_cube_goal) * 10.0
            self.prev_dist_cube_goal = dist_cube_goal
            if is_grasping:
                reward += self.holding_bonus
            elif not self._is_placed(cube_pos):
                reward -= self.drop_penalty

        placed = self.has_been_grasped and self._is_placed(cube_pos) and not is_grasping
        if placed:
            reward += self.place_bonus

        dropped_early = self.has_been_grasped and not is_grasping and not placed

        terminated = bool(placed or dropped_early)
        self.current_step += 1
        truncated = self.current_step >= self.max_episode_steps

        info = {
            "is_success": placed,
            "is_grasping": is_grasping,
            "has_been_grasped": self.has_been_grasped,
            "dropped": dropped_early,
            "dist_gripper_cube": dist_gripper_cube,
            "dist_cube_goal": dist_cube_goal,
        }
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        arm_pos = []
        arm_vel = []
        for j in self.arm_joints:
            state = p.getJointState(self.panda_id, j, physicsClientId=self.physics_client)
            arm_pos.append(state[0])
            arm_vel.append(state[1])

        finger_pos = [
            p.getJointState(self.panda_id, j, physicsClientId=self.physics_client)[0]
            for j in self.finger_joints
        ]

        gripper_pos = self._get_gripper_position()
        cube_pos = self._get_cube_position()
        relative_vec = cube_pos - gripper_pos
        is_grasping = float(self._check_grasping())

        obs = (
            arm_pos
            + arm_vel
            + finger_pos
            + list(gripper_pos)
            + list(cube_pos)
            + list(relative_vec)
            + list(self.goal_position)
            + [is_grasping]
        )
        return np.array(obs, dtype=np.float32)

    def _get_gripper_position(self):
        link_state = p.getLinkState(
            self.panda_id, self.grasp_frame_link, physicsClientId=self.physics_client
        )
        return np.array(link_state[0])

    def _get_cube_position(self):
        pos, _ = p.getBasePositionAndOrientation(
            self.cube_id, physicsClientId=self.physics_client
        )
        return np.array(pos)

    def _check_grasping(self):
        contacts_1 = p.getContactPoints(
            bodyA=self.panda_id, bodyB=self.cube_id, linkIndexA=9,
            physicsClientId=self.physics_client,
        )
        contacts_2 = p.getContactPoints(
            bodyA=self.panda_id, bodyB=self.cube_id, linkIndexA=10,
            physicsClientId=self.physics_client,
        )
        return len(contacts_1) > 0 and len(contacts_2) > 0

    def _is_placed(self, cube_pos):
        return np.linalg.norm(cube_pos[:2] - self.goal_position[:2]) < self.place_threshold and \
               abs(cube_pos[2] - self.table_height) < 0.03

    def _sample_cube_position(self):
        x = self.np_random.uniform(0.4, 0.6)
        y = self.np_random.uniform(-0.2, 0.2)
        return np.array([x, y, self.table_height])

    def _sample_goal_position(self, cube_position):
        for _ in range(50):
            x = self.np_random.uniform(0.4, 0.6)
            y = self.np_random.uniform(-0.2, 0.2)
            candidate = np.array([x, y, self.table_height])
            if np.linalg.norm(candidate[:2] - cube_position[:2]) > 0.1:
                return candidate
        return candidate

    def close(self):
        if p.isConnected(physicsClientId=self.physics_client):
            p.disconnect(physicsClientId=self.physics_client)
