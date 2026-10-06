"""Extended policy observations for object geometry and a known obstacle."""
import numpy as np
from gymnasium import spaces
from robotic_arm.environments.pick_place_env import PickPlaceEnv
from robotic_arm.environments.reaching_env import requires_connection


class ObstaclePickPlaceEnv(PickPlaceEnv):
    def __init__(self, **kwargs):
        kwargs.setdefault("max_episode_steps",500)
        super().__init__(**kwargs)
        self.observation_space=spaces.Box(-np.inf,np.inf,(32,),np.float32)

    @requires_connection
    def reset(self,seed=None,options=None):
        options=dict(options or {})
        if self.domain_randomization:
            # Balanced corner curriculum; explicit evaluation overrides are preserved.
            index=seed if seed is not None else int(self.np_random.integers(100000))
            options.setdefault("size",[.04,.05,.06][index%3])
            if not self.scene_randomization:
                options.setdefault("obstacle_height",[0.,.08,.24][(index//3)%3])
        super().reset(seed=seed,options=options)
        if not self.start_near_cube:
            import pybullet as p
            # One common home posture for every scene, including unobstructed scenes.
            for joint,angle in enumerate([-.5,-.4,1.5,-2.,0.,1.6,.8]):
                p.resetJointState(self.arm,joint,angle,physicsClientId=self.physics_client)
            self.target=np.array(p.getLinkState(self.arm,11,computeForwardKinematics=True,physicsClientId=self.physics_client)[4])
            self._advance(.04)
            self.best_reach_distance=self._info()["reach_distance_m"]
        return self._observation(),self._info()

    def _observation(self):
        basic=super()._observation()
        if self.obstacle is None:
            center=np.zeros(3); half=np.zeros(3)
        else:
            import pybullet as p
            center=np.array(p.getBasePositionAndOrientation(self.obstacle,physicsClientId=self.physics_client)[0])
            half=np.array([*self.scene_config["obstacle_half_xy"],self.scene_config["obstacle_height"]/2])
        return np.r_[basic,center,half,self.scene_config["size"],self.scene_config["object_shape"]=="cylinder"].astype(np.float32)

    def _info(self):
        return {**super()._info(),"environment_profile":"obstacle_aware_v1","episode_horizon":self.max_episode_steps}

    def step(self,action):
        obs,reward,t,tr,info=super().step(action)
        if self.obstacle_contact:
            info["is_success"]=False
            info["collision_failure"]=True
            self._done=True
            return obs,reward-20.,True,False,info
        info["collision_failure"]=False
        return obs,reward,t,tr,info
