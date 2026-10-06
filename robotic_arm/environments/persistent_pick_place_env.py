"""Three physical objects remain in one scene; no between-task world resets."""
import numpy as np
import pybullet as p
from robotic_arm.environments.oriented_pick_place_env import OrientedPickPlaceEnv
from robotic_arm.environments.scene_layouts import THREE_TASKS


class PersistentPickPlaceEnv(OrientedPickPlaceEnv):
    def reset(self,seed=None,options=None):
        source,goal=THREE_TASKS[0]
        super().reset(seed=seed,options={"source_xy":list(source),"destination_xy":list(goal)})
        self.task_index=0
        self.objects=[self.cube]
        c=self.physics_client
        for index,(source,goal) in enumerate(THREE_TASKS[1:],1):
            collision=p.createCollisionShape(p.GEOM_BOX,halfExtents=[self.half]*3,physicsClientId=c)
            color=([.2,.6,.9,1],[.8,.3,.8,1])[index-1]
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[self.half]*3,rgbaColor=color,physicsClientId=c)
            body=p.createMultiBody(baseMass=.08,baseCollisionShapeIndex=collision,baseVisualShapeIndex=visual,
                basePosition=[*source,self.half],physicsClientId=c)
            p.changeDynamics(body,-1,lateralFriction=2.,spinningFriction=.01,rollingFriction=.001,physicsClientId=c)
            self.objects.append(body)
            marker=p.createVisualShape(p.GEOM_CYLINDER,radius=.05,length=.002,rgbaColor=color,physicsClientId=c)
            p.createMultiBody(baseVisualShapeIndex=marker,basePosition=[*goal,.002],physicsClientId=c)
        return self._observation(),self._info()

    def next_task(self):
        if not self._done or not self._info()["is_success"]:
            raise RuntimeError("Complete the current task successfully before switching objects")
        if self.task_index>=2:
            raise RuntimeError("All three tasks are complete")
        self.task_index+=1
        self.cube=self.objects[self.task_index]
        source,goal=THREE_TASKS[self.task_index]
        self.scene_config.update(source_xy=list(source),destination_xy=list(goal))
        self.goal=np.array([*goal,self.half])
        self.steps=self.stable_steps=0
        self.grasped=self.lifted=self.ground_contact=self.obstacle_contact=False
        self._done=False
        self.grasp_bonus_paid=False
        self.max_held_height=self.half
        self.ground_slide_distance=0.
        self.source=self._state()[1][:2].copy()
        self.best_reach_distance=self._info()["reach_distance_m"]
        self.target=self._state()[0].copy()
        return self._observation(),self._info()
