"""Alternative gripper orientation for grasping beside known obstacles.

This profile does not guarantee collision avoidance. Contacts still fail evaluation.
"""
import numpy as np
import pybullet as p
from robotic_arm.environments.obstacle_pick_place_env import ObstaclePickPlaceEnv
from robotic_arm.environments.reaching_env import requires_connection


class OrientedPickPlaceEnv(ObstaclePickPlaceEnv):
    """Quarter-turn grasp profile; every movement uses physical motor control."""
    def _orientation(self):
        return p.getQuaternionFromEuler([np.pi,0,np.pi/2])

    @requires_connection
    def reset(self,seed=None,options=None):
        super().reset(seed=seed,options=options)
        if not self.start_near_cube:
            p.resetJointState(self.arm,6,2.37,physicsClientId=self.physics_client)
            self.target=np.array(p.getLinkState(self.arm,11,computeForwardKinematics=True,physicsClientId=self.physics_client)[4])
            self._advance(.04)
            self.best_reach_distance=self._info()["reach_distance_m"]
        return self._observation(),self._info()

    def _info(self):
        return {**super()._info(),"environment_profile":"quarter_turn_v2"}
