from typing import List
import matplotlib.pyplot as plt
import numpy as np

from objects import *
from belief import *

# TODO: We should uniform the input to functions. There is a conflict between the use of Pose and np.ndarray
# that comes from the fact that, in the Belief modules we use np arrays and have to notion of Pose.

class BaseEnvironment(object):
    def __init__(self, shape: np.ndarray, dt: float) -> None:
        self.shape = shape
        self.dt = dt

        self.agents = []
        self.targets = []
        self.belief = []
        self.occluders = [] # Occluders are a special label for objects that can occlude the perception
        self.lights = [] # Lights are a special label for objects that can create backlighting
        self.uid_counter = 0

    def addOccupiedBorder(self, border):
        self.occupied_borders.append(border)
        
    def createUID(self):
        self.uid_counter += 1
        return self.uid_counter
    
    def addAgent(self, agent):
        self.agents.append(agent)

    def addTarget(self, target):
        self.targets.append(target)
    
    def addOccluder(self, occluder):
        self.occluders.append(occluder)

    def addLight(self, light):
        self.lights.append(light)

    def addBelief(self, belief):
        self.belief.append(belief)
        
    def update(self):
        for agent in self.agents:
            agent.update(self.dt)
        for target in self.targets:
            target.update(self.dt)


class Environment2D(BaseEnvironment):
    def __init__(self, height: float, width: float, dt: float):
        super(Environment2D, self).__init__(np.array([height, width]), dt)
        self.height: float = height
        self.width: float = width

        # Add typing for specific 2D claasses
        self.agents: list[Agent2D] = []
        self.targets: list[Target2D | Target2DSemantic] = []
        self.belief: list[Position2dEKF | Position2dEKFWithUtility | SemanticBayes | SemanticBayesWithUtility] = []
        self.occluders: list[Occluder2D] = [] # Occluders are a special label for objects that can occlude the perception
        self.lights: list[PlaneLight2D] = [] # Lights are a special label for objects that can create backlighting
        self.occupied_borders = [] # Helper to assign a border to the lights

    def addOccupiedBorder(self, border: np.ndarray):
        self.occupied_borders.append(border)

    def isPositionValid(self, position: Pose, margin: float = 0.1, avoid_ids=[]):
        # Check if the position is withing the limits
        # and if it is does not collide with any element
        if position.translation[0] < -self.height/2 or position.translation[0] > self.height/2 or position.translation[1] < -self.width/2 or position.translation[1] > self.width/2:
            return False
        for target in self.targets:
            # Check distance
            if np.sqrt((position.translation[0] - target.pose.x)**2 + (position.translation[1] - target.pose.y)**2) < target.size + margin:
                return False
        # Checking the agent has the problem of the agent colliding with itself
        #for agent in self.agents:
            # Check distance
        #    if np.sqrt((position.translation[0] - agent.pose.x)**2 + (position.translation[1] - agent.pose.y)**2) < margin:
        #        return False
        for occluder in self.occluders:
            # Check distance
            if np.sqrt((position.translation[0] - occluder.pose.x)**2 + (position.translation[1] - occluder.pose.y)**2) < occluder.size + margin:
                return False
        return True


class Environment2DRoughness(Environment2D):
    def __init__(self, height: float, width: float, dt: float, resolution: float = 0.1):
        super(Environment2DRoughness, self).__init__(height, width, dt)
        # Create the roughness map with 1 for all positions at resolution and 0.3 for a square of 15% of the size in the middle
        self.roughness_map = np.ones(
            (int(self.height / resolution), int(self.width / resolution)))
        
        # TODO: load it or generate a random one
        self.roughness_map[int(self.height / resolution / 2 - self.height / resolution / 2 * 0.15):int(self.height / resolution / 2 + self.height / resolution / 2 * 0.15),
                           int(self.width / resolution / 2 - self.width / resolution / 2 * 0.15):int(self.width / resolution / 2 + self.width / resolution / 2 * 0.15)] = 1.5
        
        self.resolution = resolution

    def getRoughtness(self, x: float, y: float) -> float:
        return self.roughness_map[int(x / self.resolution), int(y / self.resolution)]

    def plotRoughnessMap(self) -> None:
        self.ax.imshow(self.roughness_map, extent=[
                       0, self.height, 0, self.width], origin='lower')
