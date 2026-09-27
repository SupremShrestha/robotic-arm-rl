import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data


class ReachEnv(gym.Env):
    """
    A Gymnasium environment where a KUKA iiwa robotic arm learns to move
    its end-effector to a target position in 3D space.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.controllable_joints = [0, 1, 2, 3, 4, 5, 6]
        self.end_effector_link_index = 6
        self.max_action_delta = 0.05  # max radians a joint can move per step
        self.num_joints = len(self.controllable_joints)

        # --- Connect to PyBullet ---
        # GUI mode if we want to watch it, DIRECT (headless) for fast training
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # --- Define observation space ---
        # [7 joint positions, 7 joint velocities, 3 end-effector xyz, 3 target xyz] = 20 values
        # Using generous bounds; PPO doesn't need these to be perfectly tight.
        obs_dim = self.num_joints + self.num_joints + 3 + 3
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )

        # --- Define action space ---
        # One continuous delta per joint, bounded to [-1, 1].
        # We'll scale this to actual radians inside step() in Step 10.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )

        # Placeholders - populated in reset()
        self.arm_id = None
        self.target_id = None
        self.plane_id = None
        self.target_position = np.array([0.4, 0.3, 0.6])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.plane_id = p.loadURDF("plane.urdf")
        self.arm_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)

        # Fixed target for now - Step 17 will randomize this
        self.target_position = np.array([0.4, 0.3, 0.6])
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE, radius=0.03, rgbaColor=(1, 0, 0, 1)
        )
        self.target_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=self.target_position.tolist(),
        )

        # Placeholder observation - real implementation in Step 9
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        # Clip defensively in case the action came from outside action_space.sample()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Scale action from [-1, 1] to [-max_action_delta, +max_action_delta] radians
        joint_deltas = action * self.max_action_delta

        for i, joint_index in enumerate(self.controllable_joints):
            current_angle = p.getJointState(self.arm_id, joint_index)[0]
            target_angle = current_angle + joint_deltas[i]

            p.setJointMotorControl2(
                bodyUniqueId=self.arm_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_angle,
                force=300,
            )

        p.stepSimulation()

        observation = self._get_observation()

        # Placeholder reward and termination - real logic in Step 11/12
        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        joint_positions = []
        joint_velocities = []

        for joint_index in self.controllable_joints:
            joint_state = p.getJointState(self.arm_id, joint_index)
            joint_positions.append(joint_state[0])   # index 0 = position
            joint_velocities.append(joint_state[1])  # index 1 = velocity

        link_state = p.getLinkState(self.arm_id, self.end_effector_link_index)
        end_effector_position = link_state[0]  # (x, y, z)

        observation = np.array(
            joint_positions
            + joint_velocities
            + list(end_effector_position)
            + list(self.target_position),
            dtype=np.float32,
        )

        return observation

    def close(self):
        p.disconnect()