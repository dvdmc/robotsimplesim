import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.lines as lines
from objects import BaseObject, Occluder2D
from perception import isOccluded2DCircles, BaseObservationModel
from basics import Pose
from priors import BasePrior

class Position2dEKF(object):
    def __init__(self, initial_position: np.ndarray, P: np.ndarray, id: int, perception: BaseObservationModel, Q: np.ndarray, R: np.ndarray, dt: float):
        """
            This is an implementation of the Extended Kalman Filter for
            estimating the state of a static target object.
            - The state is currently defined as [x, y].
            - The measurement is currently defined as [distance, bearing].
        """
        self.position = initial_position
        self.P = P
        self.id = id
        self.perception = perception
        self.Q = Q
        self.R = R
        self.dt = dt

        # A control variable that flags when the belief stops being consistent
        # due to being used for planning
        self.is_real_belief = True # TODO: decide name for this

        # Visualization is a ellipse with 2 standard deviations
        self.visualization = matplotlib.patches.Ellipse((0, 0), 0, 0, color='r')
        self.visualization.set_alpha(0.5)
        self.updateVisualization()

    def copyBelief(self) -> 'Position2dEKF':
        """
            Returns a copy of the belief object maintining the id, perception, Q, R and dt.
            But deep copying the state and covariance to allow for independent updates.
        """
        return Position2dEKF(self.position.copy(), self.P.copy(), self.id, self.perception, self.Q, self.R, self.dt)
    
    def predict(self):
        # This does nothing as the target is stationary
        pass
    
    def update(self, z: np.ndarray, measurement_pose: Pose) -> None:
        # Estimate the measurement
        target_pose = Pose(self.position[0], self.position[1], 0)
        observation = self.perception.estimateObservation(measurement_pose, target_pose)

        z_hat = np.array([observation.distance, observation.bearing])

        # Compute the Kalman gain
        H = self.perception.linearize(measurement_pose, target_pose)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state
        self.position = self.position + K @ (z - z_hat)
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P + 1e-6 * np.eye(self.P.shape[0]) # Add a small value to avoid singularities
        self.updateVisualization()

    def simulateUpdate(self, measurement_pose: np.ndarray) -> bool:
        """
            Simulate the update from a measurement.
            This makes the belief object inconsistent as the
            state is not updated but the covariance is
        """
        self.is_real_belief = False
        # Estimate the measurement
        target_pose = Pose(self.position[0], self.position[1], 0)
        
        # We check if there could be a measurement. More restrictive than the update
        # In the update, there was a measurement and we work around the non-linearity 
        # of the predicted measurement by forgetting about the limits. Here we want
        # to know if there would be a measurement
        observation = self.perception.estimateObservation(measurement_pose, target_pose, check_limits=True)
        if observation is None:
            return False

        # Compute the covariance update of the simulated update
        # Compute the Kalman gain
        H = self.perception.linearize(measurement_pose, target_pose)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P + 1e-6 * np.eye(self.P.shape[0]) # Add a small value to avoid singularities

        return True
    
    def getEntropy(self):
        """
            Returns the entropy of the belief.
        """
        eig_vals, eig_vecs = np.linalg.eig(self.P)
        return np.sum(np.log(eig_vals + 1e-6))
    
    def getError(self, target):
        """
            Returns the errors of the belief.
        """
        return np.sqrt((self.position[0] - target.pose.x)**2 + (self.position[1] - target.pose.y)**2)
    
    def getErrorXY(self, target):
        """
            Returns the errors of the belief.
        """
        return np.array([self.position[0] - target.pose.x, self.position[1] - target.pose.y])
    
    def getNEES(self, target):
        """
            Returns the NEES of the belief.
        """
        error = self.getErrorXY(target)
        return error.T @ np.linalg.inv(self.P) @ error
    
    def updateVisualization(self):
        # Update the ellipse position
        self.visualization.center = (self.position[0], self.position[1])
        # Update the ellipse from the covariance matrix eigenvalues
        eig_vals, eig_vecs = np.linalg.eig(self.P)
        self.visualization.width = 2 * np.sqrt(max(eig_vals[0],0)) * 2
        self.visualization.height = 2 * np.sqrt(max(eig_vals[1],0)) * 2
        self.visualization.angle = np.rad2deg(np.arctan2(eig_vecs[1, 0], eig_vecs[0, 0]))

class OccluderFactor(object):
    """
        The occluder factor is a parabole defined by its center, orientation and width.
    """
    def __init__(self, center, orientation, width):
        self.center = center
        self.orientation = orientation
        self.width = width

class LightFactor(object):
    """
        The light factor is a parabole defined by its center, orientation and width.
    """
    def __init__(self, center, orientation, width):
        self.center = center
        self.orientation = orientation
        self.width = width


class UtilityEstimationGrid2D(object):
    """
        This is an implementation of the utility estimation for a static target.
        The utility stores the reward of an observation viewpoint.
        It is a R^2 -> R function that varies depending on the previously acquired measurements (p_{0...n}).
        This is evaluated for the whole map.

        The utility is defined as:
            U(x) = prior / (prod_{i = p_{0...n}} 1 + exp(-norm2(x,p_i)/(2 * sigma^2)))
    """
    def __init__(self, height: float, width: float, prior: BasePrior, sigma: float = 0.1, use_numeric_factor: bool = True):
        self.height = height
        self.width = width
        self.prior = prior
        self.sigma = sigma
        self.poses = []
        self.occluders = []
        self.occluders_factors = []
        self.lights = []
        self.light_factors = []
        self.target = None
        self.use_numeric_factor = use_numeric_factor
        self.x_axis = np.arange(-self.width / 2, self.width / 2, 0.3)
        self.y_axis = np.arange(-self.height / 2, self.height / 2, 0.3)
        
        self.fig = None
        self.ax = None
        self.configured = False
        self.is_visualize = False

        if self.is_visualize:
            self.visualize()

    def copyUtility(self) -> 'UtilityEstimationGrid2D':
        """
            Returns a copy of the utility object maintaining the height, width, prior, sigma, current poses and occluders.
        """
        utility = UtilityEstimationGrid2D(self.height, self.width, self.prior, self.sigma)
        utility.poses = self.poses.copy()
        utility.occluders = self.occluders.copy()
        utility.occluders_factors = self.occluders_factors.copy()
        utility.lights = self.lights.copy()
        utility.light_factors = self.light_factors.copy()
        utility.target = self.target
        return utility
    
    def addMeasurementPose(self, pose: Pose) -> None:
        self.poses.append(pose)
        if self.is_visualize:
            self.visualize()

    def addTarget(self, target: np.ndarray) -> None:
        self.target = target
        if self.prior.__class__.__name__ == "DistanceBasedPrior":
            self.prior.origin = np.array([target[0], target[1]])

    def addOccluder(self, occluder: Occluder2D) -> None:
        self.occluders.append(occluder)
        if self.target is None:
            raise ValueError("The target must be set before adding occluders")
        if(self.target is not None) and self.is_visualize:
            self.visualize()
    
    def addLight(self, light: np.ndarray) -> None:
        self.lights.append(light)
        if self.target is None:
            raise ValueError("The target must be set before adding occluders")
        if(self.target is not None) and self.is_visualize:
            self.visualize()

    def evaluate(self, x1: float, x2: float, withTheta=False, theta=0.0, verbose=False) -> float:
        """
            This function evaluates the utility function at the given pose.
        """
        product = 1
        # Evaluate pose similarity
        if withTheta:
            theta_scale = np.array([1,1,10])
            x = np.array([x1, x2, theta])
            for pose in self.poses:
                p = np.array([pose.x, pose.y, pose.theta])
                product *= (1 + 10 * np.exp(-np.linalg.norm((x - p) * theta_scale)**2 / (2 * self.sigma)))
        else:
            x = np.array([x1, x2])
            for pose in self.poses:
                p = np.array([pose.x, pose.y])
                product *= 1 + 10 * np.exp(-np.linalg.norm(x - p)**2 / (2 * self.sigma))

        # Evaluate occlusion
        if self.target is not None and (len(self.occluders) > 0 or len(self.lights) > 0):
            agent_pose = Pose(np.array([x1, x2]), np.array([theta]))

            if self.use_numeric_factor:
                for occluder in self.occluders:
                    # Compute the factor parameters
                    # Compute the utility with the parabole approximation
                    o = occluder.pose.translation
                    t = np.array([self.target[0], self.target[1]])
                    # The center of the parabole is the occluder displaced in the opposite direction of the target
                    # by its size
                    x_0 = o[0]
                    y_0 = o[1]
                    v_t_o = o - t
                    distance = np.linalg.norm(v_t_o)
                    direction = v_t_o / distance
                    x_0 -= direction[0] * occluder.size 
                    y_0 -= direction[1] * occluder.size
                    # The orientation is given by the direction of the target
                    # The angle of the parabole is the angle between the occluder and the target
                    orientation = np.arctan2(-v_t_o[1], v_t_o[0]) + np.pi / 2
                    # Compute the rotation angle depending on the orientation
                    occluder_factor = OccluderFactor(np.array([x_0, y_0]), orientation, occluder.size * 4)

                    # Displace the parabole
                    x_d = agent_pose.translation[0] - occluder_factor.center[0]
                    y_d = agent_pose.translation[1] - occluder_factor.center[1]
                    # Rotate the parabole
                    x_r = x_d * np.cos(occluder_factor.orientation) - y_d * np.sin(occluder_factor.orientation)
                    y_r = x_d * np.sin(occluder_factor.orientation) + y_d * np.cos(occluder_factor.orientation)
                    # Adjust the width
                    width = occluder_factor.width
                    # Evaluate the parabole to check if x-y is inside
                    distance_to_parabole = x_r**2 / width**2 - y_r
                    if distance_to_parabole < 0: # The point is inside the parabole
                        # When the distance to the parable is high, the product increases
                        # The value is between 1 for 0 and 100 for max distance (width/2)
                        # With quadratic increase
                        if verbose:
                            print("(1,2) Occlusion considered for occluder and target {}, {}, {}, {}!".format(o[0], o[1], t[0], t[1]))
                        product *= 1 + 3 * np.exp(-(x_r)**2/(y_r))

                for light in self.lights:
                        # Precompute the factor parameters
                        # Compute the utility with the parabole approximation
                        l = np.array([light.pose.translation[0], light.pose.translation[1]])
                        t = np.array([self.target[0], self.target[1]])
                        # The center of the parabole is the target 
                        x_0 = t[0]
                        y_0 = t[1]
                        v_l_o = light.emission_direction
                        direction = v_l_o / np.linalg.norm(v_l_o)
                        # The orientation is given by the direction of the target
                        # The angle of the parabole is the angle between the occluder and the target
                        orientation = np.arctan2(-v_l_o[1], v_l_o[0]) + np.pi / 2
                        # Compute the rotation angle depending on the orientation
                        light_factor = LightFactor(np.array([x_0, y_0]), orientation, 3.0)
                        # Displace the parabole
                        x_d = agent_pose.translation[0] - light_factor.center[0]
                        y_d = agent_pose.translation[1] - light_factor.center[1]
                        # Rotate the parabole
                        x_r = x_d * np.cos(light_factor.orientation) - y_d * np.sin(light_factor.orientation)
                        y_r = x_d * np.sin(light_factor.orientation) + y_d * np.cos(light_factor.orientation)
                        # Adjust the width
                        width = light_factor.width
                        # Evaluate the parabole to check if x-y is inside
                        distance_to_parabole = x_r**2 / width**2 - y_r
                        if distance_to_parabole < 0: # The point is inside the parabole
                            # When the distance to the parable is high, the product increases
                            # The value is between 1 for 0 and 5 for max distance (width/2)
                            # With quadratic increase
                            if verbose:
                                print("(3) Light considered for light target {}, {}, {}, {}!".format(l[0], l[1], t[0], t[1]))
                            product *= 1 + 2 * np.exp(-(x_r)**2/(y_r))

            else:
                # If the target is behind the occluder, utility is 0
                is_occluded = isOccluded2DCircles(self.occluders, Pose(np.array([self.target[0], self.target[1]]),np.array([0])), agent_pose, 0.3)
                # If the target is affected by the light, utility is 0
                for light in self.lights:
                    # Precompute the factor parameters
                        # Compute the utility with the parabole approximation
                        l = np.array([light.pose.translation[0], light.pose.translation[1]])
                        t = np.array([self.target[0], self.target[1]])
                        # The center of the parabole is the target 
                        x_0 = t[0]
                        y_0 = t[1]
                        v_l_o = light.emission_direction
                        direction = v_l_o / np.linalg.norm(v_l_o)
                        # The orientation is given by the direction of the target
                        # The angle of the parabole is the angle between the occluder and the target
                        orientation = np.arctan2(-v_l_o[1], v_l_o[0]) + np.pi / 2
                        # Compute the rotation angle depending on the orientation
                        light_factor = LightFactor(np.array([x_0, y_0]), orientation, 3.0)
                        # Displace the parabole
                        x_d = agent_pose.translation[0] - light_factor.center[0]
                        y_d = agent_pose.translation[1] - light_factor.center[1]
                        # Rotate the parabole
                        x_r = x_d * np.cos(light_factor.orientation) - y_d * np.sin(light_factor.orientation)
                        y_r = x_d * np.sin(light_factor.orientation) + y_d * np.cos(light_factor.orientation)
                        # Adjust the width
                        width = light_factor.width
                        # Evaluate the parabole to check if x-y is inside
                        distance_to_parabole = x_r**2 / width**2 - y_r
                        if distance_to_parabole < 0: # The point is inside the parabole
                            product *= 100
                if is_occluded:
                    product *= 100

        return self.prior(np.array([x1,x2])) * min(product,100)
    
    def evaluateWithTheta(self, x1, x2, theta, verbose=False) -> float:
        return self.evaluate(x1, x2, withTheta=True, theta=theta, verbose=verbose)
    
    def visualize(self) -> None:
        """
            This function updates the utility visualization.
        """
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
        
        # Log for visualization
        # y = [[np.log(self.evaluate(x1, x2)) for x1 in self.x_axis] for x2 in self.y_axis]
        y = [[self.evaluate(x1, x2) for x1 in self.x_axis] for x2 in self.y_axis]
        # Flip the y axis
        y = np.flip(y, 0)
        self.ax.clear()
        img = self.ax.imshow(y, extent=[-self.width / 2, self.width / 2, -self.height / 2, self.height / 2])
        # Plot the occluders
        occluders_positions = []
        for occluder in self.occluders:
            occluders_positions.append([occluder.pose.translation[0], occluder.pose.translation[1]])
        if len(occluders_positions) > 0:
            occluders_positions = np.array(occluders_positions)
            self.ax.scatter(occluders_positions[:, 0], occluders_positions[:, 1], c='black', marker='o', s=50)
        # Plot the target
        if self.target is not None:
            self.ax.scatter(self.target[0], self.target[1], c='red', marker='x', s=50)
        self.ax.set_title("Perceptual map estimation")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        # Add colorbar
        self.fig.colorbar(img, ax=self.ax)
        self.fig.show()
        pass

class Position2dEKFWithUtility(Position2dEKF):
    def __init__(self, x: np.ndarray, P: np.ndarray, id: int, perception: BaseObservationModel, Q: np.ndarray, R: np.ndarray, dt: float, utility: UtilityEstimationGrid2D):
        super().__init__(x, P, id, perception, Q, R, dt)
        self.utility = utility

    def copyBelief(self):
        """
            Returns a copy of the belief object maintining the id, perception, Q, R, dt and copying the utility.
            But deep copying the state and covariance to allow for independent updates.
        """
        belief = Position2dEKFWithUtility(self.position.copy(), self.P.copy(), self.id, self.perception, self.Q, self.R, self.dt, self.utility.copyUtility())
        return belief

    def update(self, z, measurement_pose):
        # Estimate the measurement
        target_pose = Pose(np.array([self.position[0], self.position[1]]), np.array([0]))
        observation = self.perception.estimateObservation(measurement_pose, target_pose)
        z_hat = np.array([observation.distance, observation.bearing])

        # Compute the Kalman gain
        H = self.perception.linearize(measurement_pose, target_pose)
        u = self.utility.evaluateWithTheta(measurement_pose.translation[0], measurement_pose.translation[1], measurement_pose.rotation[0], verbose=True)
        S = H @ self.P @ H.T + self.R * u 
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update the state
        self.position = self.position + K @ (z - z_hat)
        self.utility.addTarget(self.position)
        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P + 1e-6 * np.eye(self.P.shape[0]) # Add a small value to avoid singularities
        self.updateVisualization()

    def simulateUpdate(self, measurement_pose: np.ndarray) -> bool:
        """
            Simulate the update from a measurement.
            This makes the belief object inconsistent as the
            state is not updated but the covariance is
        """
        self.is_real_belief = False
        # Estimate the measurement
        target_pose = Pose(np.array([self.position[0], self.position[1]]), np.array([0]))
        
        # We check if there could be a measurement. More restrictive than the update
        # In the update, there was a measurement and we work around the non-linearity 
        # of the predicted measurement by forgetting about the limits. Here we want
        # to know if there would be a measurement
        observation = self.perception.estimateObservation(measurement_pose, target_pose, check_limits=True)
        if observation is None:
            return False

        # Compute the covariance update of the simulated update
        # Compute the Kalman gain
        H = self.perception.linearize(measurement_pose, target_pose)
        u = self.utility.evaluateWithTheta(measurement_pose.translation[0], measurement_pose.translation[1], measurement_pose.rotation[0])
        S = H @ self.P @ H.T + self.R * u
        K = self.P @ H.T @ np.linalg.inv(S)

        self.P = (np.eye(self.P.shape[0]) - K @ H) @ self.P + 1e-6 * np.eye(self.P.shape[0]) # Add a small value to avoid singularities

        # Make the state inconsistent to raise an error if it is used
        self.position = np.ones(self.position.shape) * np.inf

        return True

class SemanticBayes(object):
    def __init__(self, num_classes: int, prior: BasePrior, id: int, target_pose: np.ndarray, perception: BaseObservationModel, colors: list):

        self.class_p = prior
        self.id = id
        self.position = target_pose
        self.perception = perception
        self.visualization_spacing = 0.2
        self.visualization_offset = 0.2
        self.num_classes = num_classes
        # Visualization is a ellipse with 2 standard deviations
        self.colors = colors
        self.visualization = [matplotlib.patches.Rectangle((0,0), 0.1, 0, color=colors[i]) for i in range(num_classes)]
        self.visualization.append(lines.Line2D([], [], color='black'))
        self.updateVisualization()

    def copyBelief(self) -> 'SemanticBayes':
        """
            Returns a copy of the belief object maintining the id, position, perception, properties, etc.
            But deep copying the state and covariance to allow for independent updates.
        """
        return SemanticBayes(self.num_classes, self.class_p.copy(), self.id, self.position, self.perception, self.colors)
    
    def predict(self) -> None:
        # This does nothing as the target is stationary
        pass

    def update(self, z: np.ndarray, measurement_pose: Pose) -> None:
        # Compute the semantic update
        self.class_p = self.class_p * z
        self.class_p = self.class_p / np.sum(self.class_p)
        self.updateVisualization()

    def updateVisualization(self) -> None:
        # Update the rectangles position
        for idx, rectangle in enumerate(self.visualization):
            if isinstance(rectangle, matplotlib.patches.Rectangle):
                rectangle.set_xy((self.position[0]+self.visualization_offset+self.visualization_spacing/2+self.visualization_spacing*idx, self.position[1]+self.visualization_offset))
                rectangle.set_width(0.1)
                rectangle.set_height(self.class_p[idx]*0.5)
            else:
                rectangle.set_xdata([self.visualization_offset+self.position[0], self.visualization_offset+self.position[0], self.visualization_offset+self.position[0] + self.visualization_spacing * self.num_classes + self.visualization_spacing])
                rectangle.set_ydata([self.visualization_offset+self.position[1] + 0.5, self.visualization_offset+self.position[1], self.visualization_offset+self.position[1]])

    def simulateUpdate(self, measurement_pose: np.ndarray) -> bool:
        """
            Simulate the update from a measurement.
            This makes the belief object inconsistent as the
            state is not updated but the covariance is
        """
        self.is_real_belief = False
        # Estimate the measurement
        target_pose = Pose(np.array([self.position[0], self.position[1]]), np.array([0]))
        
        # We check if there could be a measurement. More restrictive than the update
        # In the update, there was a measurement and we work around the non-linearity 
        # of the predicted measurement by forgetting about the limits. Here we want
        # to know if there would be a measurement
        observation = self.perception.estimateObservation(measurement_pose, target_pose, self.class_p, 1, check_limits=True)
        if observation is None:
            return False

        # Compute the semantic update
        self.class_p = self.class_p * observation.probabilities
        self.class_p = self.class_p / np.sum(self.class_p)

        return True
    
    def getEntropy(self) -> float:
        """
            Returns the entropy of the belief.
            Entropy for a categorical of three classes will have its maximum value at 0.33... with 
            and a minimum of 0 at 1.0 or 0.0 if there is another class with 0.0.
            We can say under 0.001, the true class is known.
            https://www.wolframalpha.com/input?i=-+%28+x+*+log%28x%29+%2B+2+*+%28%281-x%29%2F2%29+*+log%28%281-x%29%2F2%29+%29+between+0+and+1
        """
        return -np.sum(self.class_p * np.log(self.class_p))
    
    def getError(self, target: BaseObject) -> float:
        """
            Returns the errors of the belief.
            The error is 0 if the target is classified correctly.
        """
        classified = np.argmax(self.class_p)
        error = target.class_id != classified
        return error
    
    def getErrorTH(self, target: BaseObject, th: float) -> float:
        """
            Returns the errors of the belief.
            The error is 0 if the target is classified correctly with confidence over threshold.
        """
        return self.getConfidence(target) < th
    
    def getConfidence(self, target: BaseObject) -> float:
        """
            Returns the confidence of the true class.
        """
        return self.class_p[target.class_id]
    
class SemanticBayesWithUtility(SemanticBayes):
    def __init__(self, num_classes: int, prior: BasePrior, id: int, target_pose: np.ndarray, perception: BaseObservationModel, colors: list, utility): # TODO: add an abstract utility for this interface
        super().__init__(num_classes, prior, id, target_pose, perception, colors)
        self.utility = utility

    def copyBelief(self):
        return SemanticBayesWithUtility(self.num_classes, self.class_p.copy(), self.id, self.position, self.perception, self.colors, self.utility.copyUtility())
    
    def update(self, z: np.ndarray, measurement_pose: Pose) -> None:
        u = self.utility.evaluateWithTheta(measurement_pose.translation[0], measurement_pose.translation[1], measurement_pose.rotation[0], verbose=True)
        z_weighted = np.array(z) ** (1/u) # This is the result of z^(w_j/w_max) where w_j = 1 / u and w_max = 1.
        # Compute the semantic update
        self.class_p = self.class_p * z_weighted
        self.class_p = self.class_p / np.sum(self.class_p)
        self.updateVisualization()

    def simulateUpdate(self, measurement_pose: np.ndarray) -> bool:
        """
            Simulate the update from a measurement.
            This makes the belief object inconsistent as the
            state is not updated but the covariance is
        """
        self.is_real_belief = False
        # Estimate the measurement
        target_pose = Pose(np.array([self.position[0], self.position[1]]), np.array([0]))
        
        # We check if there could be a measurement. More restrictive than the update
        # In the update, there was a measurement and we work around the non-linearity 
        # of the predicted measurement by forgetting about the limits. Here we want
        # to know if there would be a measurement
        observation = self.perception.estimateObservation(measurement_pose, target_pose, self.class_p, 1, check_limits=True)
        if observation is None:
            return False

        # Compute the semantic update
        u = self.utility.evaluateWithTheta(measurement_pose.translation[0], measurement_pose.translation[1], measurement_pose.rotation[0])
        z_weighted = np.array(observation.probabilities) ** (1/u)
        self.class_p = self.class_p * z_weighted
        self.class_p = self.class_p / np.sum(self.class_p)

        return True