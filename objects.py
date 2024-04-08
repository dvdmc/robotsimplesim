import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from basics import wrapAngle, Pose
from dynamics import BasicVelocityDynamicModel2D
from abc import ABC, abstractmethod

from typing import Any

from perception import BaseObservationModel

class BaseObject(ABC):
    def __init__(self, id: int, pose: Pose) -> None:
        self.id = id
        self.__pose = pose

    @abstractmethod
    def update(self, dt: float) -> None:
        ...

    @abstractmethod
    def updateVisualization(self, dt: float) -> None:
        ...

    @abstractmethod
    def setPose(self, pose: Pose) -> None:
        ...
    
    @abstractmethod
    def getPose(self) -> Pose:
        ...
    
from typing import Optional

class Agent2D(BaseObject):
    def __init__(self, id: int, pose: Pose, size: float = 0.1) -> None:
        super().__init__(id, pose)
        self.dynamic = None
        self.perception = None
        self.size = size

        # Visualization
        self.o_vertices = np.array([
            [-self.size * np.sin(np.radians(60)), self.size * np.cos(np.radians(60))],
            [self.size, 0],
            [-self.size * np.sin(np.radians(60)),
             -self.size * np.cos(np.radians(60))]
        ])
        self.vertices = self.o_vertices.copy()
        self.visualization = plt.Polygon(self.vertices, color='r')
        self.updateVisualization()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose.copyPose()

        # Update the triangle vertices
        self.updateVisualization()

    def getPose(self) -> Pose:
        return self.pose.copyPose()

    def setDynamic(self, dynamic: BasicVelocityDynamicModel2D) -> None:
        self.dynamic = dynamic

    def setPerception(self, perception: BaseObservationModel) -> None:
        self.perception = perception
        self.updateVisualization()

    def setVelocity(self, v: float, w: float) -> None:
        if self.dynamic is not None:
            self.dynamic.setVelocity(v, w)

    def update(self, dt: float) -> None:
        if self.dynamic is not None:
            # Update the pose
            self.pose = self.dynamic.update(dt)

        # Update the triangle vertices
        self.updateVisualization()
        
    def updateVisualization(self) -> None:
        # Update the triangle vertices with X front (0 degrees)
        # Theta is counter-clockwise
        rotation_matrix = np.array([
            [np.cos(-self.pose.rotation), -np.sin(-self.pose.rotation)],
            [np.sin(-self.pose.rotation), np.cos(-self.pose.rotation)]
        ])

        self.vertices = self.o_vertices.dot(rotation_matrix)

        # Update triangle position
        self.visualization.set_xy(self.vertices + self.pose.translation)

        if self.perception is not None:
            # Update the perception
            self.perception.updateVisualization()


class Target2D(BaseObject):
    def __init__(self, id, pose, size=0.1):
        super(Target2D, self).__init__(id, pose)
        self.size = size

        # Visualization
        self.visualization = plt.Circle((self.pose.x, self.pose.y), self.size, color='b')
        self.updateVisualization()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose.copyPose()

        # Update the circle
        self.updateVisualization()

    def getPose(self) -> Pose:
        return self.getPose()
    
    def update(self, dt: float) -> None:
        # Don't do anything yet
        pass

    def updateVisualization(self) -> None:
        self.visualization.set_xy([self.pose.translation[0], self.pose.translation[1]])

class Target2DSemantic(BaseObject):
    def __init__(self, id: int, pose: Pose, size: float = 0.1, class_id: int = 0, color: str = 'b'):
        super(Target2DSemantic, self).__init__(id, pose)
        self.size = size
        self.class_id = class_id

        # Color depends on the class 
        # Visualization
        self.o_vertices = np.array([
            [-self.size * np.sin(np.radians(60)), self.size * np.cos(np.radians(60))],
            [self.size, 0],
            [-self.size * np.sin(np.radians(60)),
             -self.size * np.cos(np.radians(60))]
        ])
        self.vertices = self.o_vertices.copy()
        self.visualization = plt.Polygon(self.vertices, color=color)
        self.updateVisualization()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose.copyPose()

        # Update the circle
        self.updateVisualization()

    def getPose(self) -> Pose:
        return self.getPose()
    
    def update(self, dt) -> None:
        # Don't do anything yet
        pass

    def updateVisualization(self) -> None:
        # Update the triangle vertices with X front (0 degrees)
        # Theta is counter-clockwise
        rotation_matrix = np.array([
            [np.cos(-self.pose.theta), -np.sin(-self.pose.theta)],
            [np.sin(-self.pose.theta), np.cos(-self.pose.theta)]
        ])

        self.vertices = self.o_vertices.dot(rotation_matrix)

        # Update triangle position
        self.visualization.set_xy(self.vertices + self.pose.translation)

class Occluder2D(BaseObject):
    def __init__(self, id: int, pose: Pose, size: float = 0.1):
        super(Occluder2D, self).__init__(id, pose)
        self.size = size

        # Visualization
        self.visualization = plt.Circle((self.pose.x, self.pose.y), self.size, color='k')
        self.updateVisualization()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose.copyPose()

        # Update the circle
        self.updateVisualization()

    def getPose(self) -> Pose:
        return self.getPose()
    
    def update(self, dt) -> None:
        # Don't do anything yet
        pass

    def updateVisualization(self) -> None:
        self.visualization.set_center((self.pose.translation[0], self.pose.translation[1]))

class PlaneLight2D(BaseObject):
    """
        A plane of light (line in 2D) that generates backlight on the targets if they are observed directly
        The plane is defined by a center point, it expands size/2 in both directions
        The angle defines how the light is emitted from the plane
    """
    def __init__(self, id: int, pose: Pose, size: float = 0.1, angle: float = 0.0, extent: float = 10):
        super(PlaneLight2D, self).__init__(id, pose, size)
        self.angle = angle
        self.extent = extent

        # The body line extends perpendicular to the position.theta
        # The side lines extend from the extremes of the body line in addition to pose.theta by the angle
        # B    / extent
        #     /
        # A  / angle
        #   | size/2
        #   |
        #   pose xy
        #   |
        #   | -size/2
        # C  \ -angle
        #     \
        # D    \ -extent
        # The extent refers to length of the forward direction. So, it should be increased by:
        self.extent = self.extent / np.cos(self.angle)
        self.v_A = np.array([self.pose.translation[0] - self.size/2 * np.sin(self.pose.rotation), self.pose.translation[1] + self.size/2 * np.cos(self.pose.rotation)])
        self.v_B = np.array([self.v_A[0] + self.extent * np.cos(wrapAngle(self.pose.rotation + self.angle)), self.v_A[1] + self.extent * np.sin(wrapAngle(self.pose.rotation + self.angle))])
        self.v_C = np.array([self.pose.translation[0] + self.size/2 * np.sin(self.pose.rotation), self.pose.translation[1] - self.size/2 * np.cos(self.pose.rotation)])
        self.v_D = np.array([self.v_C[0] + self.extent * np.cos(wrapAngle(self.pose.rotation - self.angle)), self.v_C[1] + self.extent * np.sin(wrapAngle(self.pose.rotation - self.angle))])

        self.body_line = np.array([self.v_A, self.v_C])
        self.body_vector = np.array([self.v_C[0] - self.v_A[0], self.v_C[1] - self.v_A[1]])
        # Emission direction is perpendicular to the body line
        self.emission_direction = np.array([-self.body_vector[1], self.body_vector[0]])
        self.side_line_1 = np.array([self.v_A, self.v_B])
        self.side_line_2 = np.array([self.v_C, self.v_D])

        self.visualization = plt.Polygon([self.v_A, self.v_B, self.v_D, self.v_C], color='y', alpha=0.5)
        self.updateVisualization()

    def setPose(self, pose: Pose) -> None:
        self.pose = pose.copyPose()

        # Update the light polygon
        self.updateVisualization()

    def contains(self, pose: Pose, margin: float = 0.0) -> bool:
        """
            Check if the pose is inside the light polygon
            margin is the margin of error
        """
        point = Point(pose.translation[0], pose.translation[1])
        polygon = Polygon([self.v_A, self.v_B, self.v_D, self.v_C])
        # Check if the point is inside the polygon
        # considering the margin
        margins = [Point(pose.translation[0] + margin, pose.translation[1] + margin), Point(pose.translation[0] + margin, pose.translation[1] - margin), Point(pose.translation[0] - margin, pose.translation[1] + margin), Point(pose.translation[0] - margin, pose.translation[1] - margin)]
        for margin in margins:
            if polygon.contains(margin):
                return True
        return polygon.contains(point)

    def update(self, dt: float) -> None:
        # Don't do anything yet
        pass

    def updateVisualization(self) -> None:
        # Update the polygon
        self.visualization.set_xy([self.v_A, self.v_B, self.v_D, self.v_C])