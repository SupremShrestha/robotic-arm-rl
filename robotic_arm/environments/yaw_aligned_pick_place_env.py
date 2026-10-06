"""Top-down grasp alignment using known initial object yaw, not learned perception."""
import numpy as np
import pybullet as p
from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
from robotic_arm.environments.reaching_env import requires_connection


class YawAlignedPickPlaceEnv(OrientedPickPlaceEnv):
    def _gripper_yaw(self):
        angle=np.pi/2+self.scene_config.get("object_yaw",0.)
        angle=(angle+np.pi)%(2*np.pi)-np.pi
        # Parallel jaws are symmetric under 180 degrees; choose a wrist-compatible yaw.
        while .8+angle>2.8973: angle-=np.pi
        while .8+angle< -2.8973: angle+=np.pi
        return angle

    def _orientation(self):
        return p.getQuaternionFromEuler([np.pi,0,self._gripper_yaw()])

    @requires_connection
    def reset(self,seed=None,options=None):
        super().reset(seed=seed,options=options)
        if not self.start_near_cube and abs(self._gripper_yaw()-np.pi/2)>1e-10:
            p.resetJointState(self.arm,6,.8+self._gripper_yaw(),physicsClientId=self.physics_client)
            self.target=np.array(p.getLinkState(self.arm,11,computeForwardKinematics=True,physicsClientId=self.physics_client)[4])
            self._advance(.04)
            self.best_reach_distance=self._info()["reach_distance_m"]
        return self._observation(),self._info()

    def _info(self):
        return {**super()._info(),"environment_profile":"yaw_aligned_v3","gripper_yaw_rad":float(self._gripper_yaw())}
