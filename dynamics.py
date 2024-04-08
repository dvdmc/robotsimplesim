import numpy as np
from abc import ABC, abstractmethod
from basics import Pose, wrapAngle
from objects import Agent2D, BaseObject

class BaseDynamicModel(ABC):
    def __init__(self, agent: BaseObject):
        self.agent = agent

    @abstractmethod
    def update(self, pose: Pose):
        pass

class BasicVelocityDynamicModel2D(BaseDynamicModel):
    def __init__(self, agent: Agent2D, v: float, w: float, dt: float):
        super(BasicVelocityDynamicModel2D, self).__init__(agent)
        self.v = v
        self.w = w
        self.dt = dt

    def setVelocity(self, v: float, w: float):
        self.v = v
        self.w = w

    def update(self) -> Pose:
        new_pose = self.agent.pose.copy()
        delta_pose = np.array([self.v * np.cos(self.agent.pose.rotation), self.v * np.sin(self.agent.pose.rotation)]) * self.dt
        new_pose.translation = self.agent.pose.translation + delta_pose
        new_pose.rotation = self.agent.pose.rotation + self.w * self.dt

        # Normalize theta between -pi and pi
        new_pose.rotation = wrapAngle(new_pose.rotation)
        
        return new_pose
    
    def linearize(self, pose: Pose) -> np.ndarray:
        F = np.array([
            [1, 0, -self.v * np.sin(pose.rotation)],
            [0, 1, self.v * np.cos(pose.rotation)],
            [0, 0, 1]
        ])
        # F here is the Jacobian of the dynamic model with respect to the state
        # This is useful for EKF in the equation x = Fx + Gu + w where w is the noise term
        # and u is the control input (v, w)
        # The steps to apply EKF are:
        # 1. Predict the state using the dynamic model. eq: x = Fx + Gu + w
        # 2. Predict the covariance using the Jacobian of the dynamic model
        # 3. Update the state using the measurement model
        # 4. Update the covariance using the Jacobian of the measurement model


        G = np.array([
            [np.cos(pose.rotation) * self.dt, 0],
            [np.sin(pose.rotation) * self.dt, 0],
            [0, self.dt]
        ])
        # G here is the Jacobian of the dynamic model with respect to the control input
        # It is used in step 2 to predict the covariance, following the equation
        # P = FPF' + GQG'
        return F
    
class RoughnessTerrainDynamicModel2D(BasicVelocityDynamicModel2D):
    def __init__(self, agent: Agent2D, v: float, w: float, dt: float, roughtness_map: np.ndarray):
        super(RoughnessTerrainDynamicModel2D, self).__init__(agent, v, w, dt)
        # Environment terrain is a 2D array of floats
        self.roughtness_map = roughtness_map

    def update(self) -> Pose:
        new_pose = self.agent.pose.copy()
        delta_pose = np.array([self.v * np.cos(self.agent.pose.rotation), self.v * np.sin(self.agent.pose.rotation)]) * self.dt
        terrain_roughness = self.roughtness_map.getRoughtness(self.agent.pose.translation[0], self.agent.pose.translation[1])
        new_pose.translation = self.agent.pose.translation + delta_pose / terrain_roughness
        new_pose.rotation = self.agent.pose.rotation + self.w * self.dt
        
        # Normalize theta between -pi and pi
        new_pose.rotation = wrapAngle(new_pose.rotation)
        
        return self.agent.pose