import numpy as np

def wrapAngle(angle: np.ndarray) -> np.ndarray:
    """
        Wrap angle between -pi and pi
        Assume that angle can be larger than 2pi or smaller than -2pi
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def wrapRollPitchYaw(roll: np.ndarray, pitch: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    """
        Wrap roll, and yaw between -pi and pi and pitch between -pi/2 and pi/2
        Assume that angle can be larger than 2pi or smaller than -2pi
    """
    while roll > np.pi:
        roll -= 2 * np.pi
    while roll < -np.pi:
        roll += 2 * np.pi
    while yaw > np.pi:
        yaw -= 2 * np.pi
    while yaw < -np.pi:
        yaw += 2 * np.pi
    while pitch > np.pi / 2:
        pitch -= np.pi
    while pitch < -np.pi / 2:
        pitch += np.pi
    return roll, pitch, yaw

import numpy as np

class Pose(object):
    def __init__(self, translation: np.ndarray, rotation: np.ndarray):
        """
        Initializes a Pose object with the given translation and rotation arrays.

        Args:
        - translation (np.ndarray): A numpy array representing the translation values.
        - rotation (np.ndarray): A numpy array representing the rotation values.

        Convention:
        - The translation array is of size 2 or 3 and represents the x, y, and z coordinates of the pose.
        - The rotation array is of size 1, 2, or 3 and represents the yaw, pitch, and roll of the pose in that specific order 
          as it is simple to interpret and allows to easily transfer from 2D to 3D in the case of yaw.
        - This is a general structure that does not differentiate. Each class that uses this structure should specify the convention.
        """
        self.translation = translation
        self.rotation = rotation

    def copyPose(self) -> 'Pose':
        """
        Returns a copy of the pose avoiding reference problems.

        Returns:
        - A new Pose object with the same translation and rotation values as the original.
        """
        return Pose(self.translation.copy(), self.rotation.copy())
    
    def __str__(self) -> str:
        """
        Returns a string representation of the Pose object.

        Returns:
        - A string containing the translation and rotation values of the Pose object.
        """
        return "Pose: translation: " + str(self.translation) + " rotation: " + str(self.rotation)
    
    def __eq__(self, __value: object) -> bool:
        """
        Compares two Pose objects for equality.

        Args:
        - __value (object): The object to compare to the current Pose object.

        Returns:
        - True if the two Pose objects have the same translation and rotation values, False otherwise.
        """
        return np.allclose(self.translation, __value.translation) and np.allclose(self.rotation, __value.rotation)