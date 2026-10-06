"""Policy-controlled Panda manipulation with physical contacts, never attachments."""
import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from robotic_arm.environments.reaching_env import requires_connection


class PickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, render_mode=None, stage="place", randomize=True, max_episode_steps=300, start_near_cube=False, domain_randomization=False, scene_randomization=False, cross_workspace=False):
        super().__init__()
        if render_mode not in (None, "human", "rgb_array") or stage not in ("reach", "lift", "place"):
            raise ValueError("Unsupported render mode or training stage")
        if max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        self.render_mode, self.stage, self.randomize = render_mode, stage, randomize
        self.max_episode_steps = max_episode_steps
        self.start_near_cube = start_near_cube
        self.domain_randomization = domain_randomization
        self.scene_randomization = scene_randomization
        self.cross_workspace = cross_workspace
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        # EE xyz, cube xyz, goal xyz, relative cube/goal, cube velocity,
        # finger openings, contact/lift history and current contacts.
        self.observation_space = spaces.Box(-np.inf, np.inf, (24,), np.float32)
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        if self.physics_client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        self._done = True

    @requires_connection
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        config={"mass":.08,"friction":2.,"size":.05,"object_shape":"box","obstacle_height":0.,"object_yaw":0.,"source_xy":None,"destination_xy":None,
                "obstacle_offset_xy":[0.,0.],"obstacle_half_xy":[.04,.04]}
        if self.domain_randomization:
            config.update(mass=float(self.np_random.uniform(.04,.15)),
                friction=float(self.np_random.uniform(.5,2.)),size=float(self.np_random.uniform(.04,.06)),
                object_shape=str(self.np_random.choice(["box","cylinder"])))
        if self.scene_randomization:
            config.update(source_xy=self.np_random.uniform([.36,-.22],[.54,-.10]).tolist(),
                destination_xy=self.np_random.uniform([.36,.12],[.54,.26]).tolist(),
                object_yaw=float(self.np_random.uniform(-np.pi,np.pi)),
                obstacle_offset_xy=self.np_random.uniform([-.025,-.025],[.025,.025]).tolist(),
                obstacle_half_xy=self.np_random.uniform([.025,.025],[.045,.045]).tolist(),
                obstacle_height=float(self.np_random.choice([0.,.08,.16,.24])))
        if self.cross_workspace:
            from robotic_arm.environments.scene_layouts import sample_cross_workspace
            config.update(sample_cross_workspace(self.np_random))
        options=options or {}
        if set(options)-set(config):
            raise ValueError("Unknown scene options")
        config.update(options)
        if (config["object_shape"] not in ("box","cylinder") or
            not all(np.isfinite(config[k]) for k in ("mass","friction","size","obstacle_height")) or
            not .02<=config["size"]<=.075 or not 0<config["mass"]<=1 or config["friction"]<0 or not 0<=config["obstacle_height"]<=.4):
            raise ValueError("Invalid object physics, size or obstacle height")
        if not np.isfinite(config["object_yaw"]):
            raise ValueError("Object yaw must be finite")
        for key in ("source_xy","destination_xy","obstacle_offset_xy","obstacle_half_xy"):
            value=config[key]
            if value is None and key in ("source_xy","destination_xy"): continue
            arr=np.asarray(value,dtype=float)
            if arr.shape!=(2,) or not np.all(np.isfinite(arr)):
                raise ValueError(f"{key} must contain two finite numbers")
            if key in ("source_xy","destination_xy") and not (.25<=arr[0]<=.60 and -.30<=arr[1]<=.30):
                raise ValueError("Position outside supported workspace")
            if key=="obstacle_half_xy" and not np.all((arr>=.01)&(arr<=.08)):
                raise ValueError("Obstacle half widths must be between .01 and .08 m")
            if key=="obstacle_offset_xy" and np.any(np.abs(arr)>.10):
                raise ValueError("Obstacle offset exceeds .10 m")
        self.scene_config=config
        self.half=config["size"]/2
        c = self.physics_client
        p.resetSimulation(physicsClientId=c)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=c)
        p.setGravity(0, 0, -9.81, physicsClientId=c)
        p.setTimeStep(1/240, physicsClientId=c)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=c)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=c)
        self.arm = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=c)
        for j, angle in enumerate([0, -.4, 0, -2, 0, 1.6, .8]):
            p.resetJointState(self.arm, j, angle, physicsClientId=c)
        for j in (9, 10):
            p.resetJointState(self.arm, j, .04, physicsClientId=c)
            p.changeDynamics(self.arm, j, lateralFriction=config["friction"], physicsClientId=c)
        source = np.array([.45, -.15])
        destination = np.array([.45, .20])
        if self.randomize:
            source += self.np_random.uniform([-.04, -.03], [.04, .03])
            destination += self.np_random.uniform([-.04, -.04], [.04, .02])
        if config["source_xy"] is not None: source=np.array(config["source_xy"],dtype=float)
        if config["destination_xy"] is not None: destination=np.array(config["destination_xy"],dtype=float)
        self.goal = np.array([*destination,self.half])
        geometry=p.GEOM_BOX if config["object_shape"]=="box" else p.GEOM_CYLINDER
        dimensions={"halfExtents":[self.half]*3} if geometry==p.GEOM_BOX else {"radius":self.half,"height":config["size"]}
        shape = p.createCollisionShape(geometry,physicsClientId=c,**dimensions)
        visual_dimensions={"halfExtents":[self.half]*3} if geometry==p.GEOM_BOX else {"radius":self.half,"length":config["size"]}
        visual = p.createVisualShape(geometry,rgbaColor=[.1,.8,.3,1],physicsClientId=c,**visual_dimensions)
        self.cube = p.createMultiBody(baseMass=config["mass"],baseCollisionShapeIndex=shape,
            baseVisualShapeIndex=visual,basePosition=[*source,self.half],
            baseOrientation=p.getQuaternionFromEuler([0,0,config["object_yaw"]]),physicsClientId=c)
        p.changeDynamics(self.cube,-1,lateralFriction=config["friction"],spinningFriction=.01,rollingFriction=.001,physicsClientId=c)
        self.obstacle=None
        self.obstacle_contact=False
        if config["obstacle_height"]:
            height=config["obstacle_height"]
            shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[*config["obstacle_half_xy"],height/2],physicsClientId=c)
            visual=p.createVisualShape(p.GEOM_BOX,halfExtents=[*config["obstacle_half_xy"],height/2],rgbaColor=[.3,.3,.8,1],physicsClientId=c)
            middle=(source+destination)/2+np.asarray(config["obstacle_offset_xy"])
            self.obstacle=p.createMultiBody(baseMass=0,baseCollisionShapeIndex=shape,baseVisualShapeIndex=visual,
                basePosition=[*middle,height/2],physicsClientId=c)
        marker = p.createVisualShape(p.GEOM_CYLINDER, radius=.05, length=.002,
            rgbaColor=[.1,.8,.3,.7], physicsClientId=c)
        p.createMultiBody(baseVisualShapeIndex=marker, basePosition=[*destination,.002], physicsClientId=c)
        self.lower = [p.getJointInfo(self.arm,j,physicsClientId=c)[8] for j in range(7)]
        self.upper = [p.getJointInfo(self.arm,j,physicsClientId=c)[9] for j in range(7)]
        if self.start_near_cube:
            angles = p.calculateInverseKinematics(self.arm,11,[*source,.045],
                self._orientation(),maxNumIterations=200,physicsClientId=c)
            for j, angle in enumerate(angles[:7]):
                p.resetJointState(self.arm,j,float(np.clip(angle,self.lower[j],self.upper[j])),physicsClientId=c)
        self.target = np.array(p.getLinkState(self.arm,11,computeForwardKinematics=True,physicsClientId=c)[4])
        self.steps, self.stable_steps = 0, 0
        self.grasped, self.lifted, self.ground_contact = False, False, False
        self._done = False
        if self.render_mode:
            p.resetDebugVisualizerCamera(1.5,55,-35,[.35,0,.3],physicsClientId=c)
        self._advance(.04)
        self.source = self._state()[1][:2].copy()
        self.best_reach_distance = self._info()["reach_distance_m"]
        self.max_held_height = self.half
        self.grasp_bonus_paid = False
        self.ground_slide_distance = 0.0
        return self._observation(), self._info()

    def _orientation(self):
        return p.getQuaternionFromEuler([np.pi,0,0])

    def _joint_targets(self):
        return p.calculateInverseKinematics(self.arm,11,self.target.tolist(),
            self._orientation(),maxNumIterations=100,residualThreshold=1e-5,physicsClientId=self.physics_client)

    def _advance(self, opening):
        c = self.physics_client
        angles = self._joint_targets()
        p.setJointMotorControlArray(self.arm,list(range(7)),p.POSITION_CONTROL,
            targetPositions=np.clip(angles[:7],self.lower,self.upper).tolist(),
            forces=[87,87,87,87,12,12,12],physicsClientId=c)
        p.setJointMotorControlArray(self.arm,[9,10],p.POSITION_CONTROL,
            targetPositions=[opening]*2,forces=[10,10],physicsClientId=c)
        for _ in range(12):
            p.stepSimulation(physicsClientId=c)
            contacts = p.getContactPoints(self.arm,self.cube,physicsClientId=c)
            self.grasped |= {x[3] for x in contacts if x[3] in (9,10)} == {9,10}
            self.lifted |= {x[3] for x in contacts if x[3] in (9,10)} == {9,10} and p.getBasePositionAndOrientation(self.cube,physicsClientId=c)[0][2] > .15
            self.ground_contact |= any(x[3]>0 for x in p.getContactPoints(self.arm,self.plane,physicsClientId=c))
            if self.obstacle is not None:
                self.obstacle_contact |= bool(p.getContactPoints(self.arm,self.obstacle,physicsClientId=c) or
                    p.getContactPoints(self.cube,self.obstacle,physicsClientId=c))
            if self.render_mode=="human":
                time.sleep(1/240)

    def _state(self):
        c = self.physics_client
        ee = np.array(p.getLinkState(self.arm,11,computeForwardKinematics=True,physicsClientId=c)[4])
        cube = np.array(p.getBasePositionAndOrientation(self.cube,physicsClientId=c)[0])
        velocity = np.array(p.getBaseVelocity(self.cube,physicsClientId=c)[0])
        fingers = [p.getJointState(self.arm,j,physicsClientId=c)[0] for j in (9,10)]
        contacts = p.getContactPoints(self.arm,self.cube,physicsClientId=c)
        return ee,cube,velocity,fingers,contacts

    def _observation(self):
        ee,cube,velocity,fingers,contacts = self._state()
        return np.concatenate([ee,cube,self.goal,cube-ee,self.goal-cube,velocity,fingers,
            [self.grasped,self.lifted], [any(x[3]==j for x in contacts) for j in (9,10)]]).astype(np.float32)

    def _info(self):
        ee,cube,velocity,fingers,contacts = self._state()
        resting = abs(cube[2]-self.half)<.015 and np.linalg.norm(velocity)<.03
        released = not contacts and min(fingers)>.025
        placed = np.linalg.norm(cube[:2]-self.goal[:2])<.05 and resting and released and self.grasped and self.lifted
        return {"is_success": bool(self.stable_steps>=10), "grasped":bool(self.grasped),
            "lifted":bool(self.lifted), "released":bool(released), "resting":bool(resting),
            "placement_candidate":bool(placed), "placement_error_m":float(np.linalg.norm(cube[:2]-self.goal[:2])),
            "reach_distance_m":float(np.linalg.norm(ee-cube)), "ground_contact":bool(self.ground_contact),
            "stage":self.stage, "reward_version":3,
            "current_grasp":{x[3] for x in contacts if x[3] in (9,10)} == {9,10},
            "cube_height_m":float(cube[2]), "finger_opening_m":float(np.mean(fingers)), "obstacle_contact":bool(self.obstacle_contact), **self.scene_config}

    @requires_connection
    def step(self, action):
        if self._done:
            raise RuntimeError("Reset before stepping a finished episode")
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (4,) or not np.all(np.isfinite(action)):
            raise ValueError("Action must contain four finite values")
        action = np.clip(action,-1,1)
        before = self._info()
        ee_before,cube_before,_,_,_ = self._state()
        # Anchor increments to the measured pose: commands cannot accumulate behind a stalled arm.
        self.target = np.clip(ee_before + .01*action[:3], [.25,-.35,.035], [.65,.35,.5])
        self._advance(float((action[3]+1)*.02))
        self.steps += 1
        info = self._info()
        ee,cube,_,_,_ = self._state()
        reward = -.05 - .01*float(np.square(action).sum())
        # Pay new achievements only: contact cycling and height oscillations cannot farm progress.
        distance = info["reach_distance_m"]
        reward += 10*max(self.best_reach_distance-distance,0)
        self.best_reach_distance = min(self.best_reach_distance,distance)
        sliding = float(np.linalg.norm(cube[:2]-cube_before[:2])) if cube[2]<.06 else 0.0
        self.ground_slide_distance += sliding
        if self.stage != "reach":
            reward -= 40*sliding
            if info["current_grasp"] and not self.grasp_bonus_paid:
                reward += 1
                self.grasp_bonus_paid = True
            if before["current_grasp"] and not info["current_grasp"]:
                reward -= 1
            if info["current_grasp"]:
                reward += 100*max(min(cube[2],.18)-self.max_held_height,0)
                self.max_held_height = max(self.max_held_height,min(cube[2],.18))
                if cube[2]>.06:
                    reward += .2*min((cube[2]-.06)/.09,1)
            if info["lifted"] and not before["lifted"]:
                reward += 5
        if self.stage == "place" and info["lifted"]:
            reward += 15*(before["placement_error_m"]-info["placement_error_m"])
        candidate = info["reach_distance_m"]<.04 if self.stage=="reach" else (
            cube[2]>.15 and info["current_grasp"] if self.stage=="lift" else info["placement_candidate"])
        self.stable_steps = self.stable_steps+1 if candidate else 0
        success = self.stable_steps>=10
        slid_out = self.stage == "lift" and self.ground_slide_distance>.08
        terminated = bool(success or slid_out or cube[2]<-.05 or np.linalg.norm(cube[:2])>1)
        if slid_out:
            reward -= 5
        truncated = bool(self.steps>=self.max_episode_steps and not terminated)
        self._done = terminated or truncated
        info["is_success"] = bool(success)
        info["max_held_height_m"] = float(self.max_held_height)
        info["ground_slide_distance_m"] = float(self.ground_slide_distance)
        info["sliding_failure"] = bool(slid_out)
        if success:
            reward += 50
        if self.ground_contact:
            reward -= .5
        return self._observation(),float(reward),terminated,truncated,info

    @requires_connection
    def render(self):
        if self.render_mode!="rgb_array":
            return None
        view=p.computeViewMatrixFromYawPitchRoll([.4,0,.12],1.3,45,-45,0,2)
        projection=p.computeProjectionMatrixFOV(55,320/240,.05,3.)
        pixels=p.getCameraImage(320,240,viewMatrix=view,projectionMatrix=projection,renderer=p.ER_TINY_RENDERER,physicsClientId=self.physics_client)[2]
        return np.asarray(pixels,dtype=np.uint8).reshape(240,320,4)[:,:,:3].copy()

    def close(self):
        if p.isConnected(self.physics_client):
            p.disconnect(self.physics_client)
