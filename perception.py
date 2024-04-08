
import random
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from abc import ABC, abstractmethod
from environment import BaseEnvironment

from logger import Logger

from basics import wrapAngle, Pose
from objects import Agent2D, BaseObject

# TODO: We should uniform the input to functions. There is a conflict between the use of Pose and np.ndarray
# that comes from the fact that, in the Belief modules we use np arrays and have to notion of Pose.

# General visual functions
def isOccluded2DCircles(occluders: list[BaseObject], target_pose: Pose, agent_pose: Pose, margin=0.0):
    # Check if a target is occluded by an obstacle

    # Get vector from agent to target
    v_a_t = target_pose.translation - agent_pose.translation

    for occluder in occluders:
        v_a_o = occluder.pose.translation - agent_pose.translation

        # 1. Project v_a_o onto v_a_t
        proj_v_a_o = np.dot(v_a_o, v_a_t) / np.linalg.norm(v_a_t)

        # 2. Check if the projection is within the range of v_a_t
        if proj_v_a_o < 0 or proj_v_a_o > np.linalg.norm(v_a_t):
            continue

        # 3. Check if the projection is within the range of v_a_o
        if np.linalg.norm(v_a_o) < proj_v_a_o:
            continue

        # 4. Compute the distance between the projection and the target
        d_t_proj = np.linalg.norm(v_a_o - proj_v_a_o * v_a_t / np.linalg.norm(v_a_t))

        # 5. Check if the distance is smaller than the size of the occluder
        if d_t_proj < occluder.size + margin:
            return True

    return False

def checkOcclusions2DCircles(occluders: list[BaseObject], target: BaseObject, agent_pose: Pose, margin=0.0):
    # Check if a target is occluded by an obstacle

    # Get vector from agent to target
    v_a_t = target.pose.translation - agent_pose.translation
    occlusions = []
    occlusion_amounts = []
    for occluder in occluders:
        v_a_o = occluder.pose.translation - agent_pose.translation

        # 1. Project v_a_o onto v_a_t
        proj_v_a_o = np.dot(v_a_o, v_a_t) / np.linalg.norm(v_a_t)

        # 2. Check if the projection is within the range of v_a_t accounting for size
        if proj_v_a_o < 0 or proj_v_a_o > np.linalg.norm(v_a_t):
            continue

        # 3. Check if the projection is within the range of v_a_o
        # This is not necessary because the projection is already within the range of v_a_t 
        # if np.linalg.norm(v_a_o) < proj_v_a_o:
        #     continue

        # 4. Compute the distance between the projection and the target
        direction_v_a_t = v_a_t / np.linalg.norm(v_a_t)
        d_t_proj = np.linalg.norm(v_a_o - proj_v_a_o * direction_v_a_t)

        # 5. Check if the distance is smaller than the size of the occluder or the target
        # The second considers the case of an occluder smaller than a target
        if d_t_proj < occluder.size + margin:
            occlusions.append(occluder)
            occlusion_amounts.append(-1)
            return occlusions, occlusion_amounts

        # If not, check if there is a partial occlusion and save the distance
        elif d_t_proj < occluder.size + target.size:
            # For the partial occlusion, consider if the occluder is smaller than the target
            if d_t_proj + occluder.size < target.size:
                # We account for the whole occluder for the occlusion amount
                occlusion_amounts.append((2*occluder.size) / (2*target.size))
            else:
                # We account for the partial occlusion
                occlusion_amounts.append((target.size - d_t_proj + occluder.size)/(2*target.size))
            occlusions.append(occluder)

    return occlusions, occlusion_amounts

class RangeBearingObservation(object):
    def __init__(self, id: int, distance: float, bearing: float, noisy_distance: float, noisy_bearing: float):
        self.id = id
        self.distance = distance
        self.bearing = bearing
        self.noisy_distance = noisy_distance
        self.noisy_bearing = noisy_bearing

class SemanticObservation(object):
    def __init__(self, id: int, probabilities: np.ndarray):
        self.id = id
        self.probabilities = probabilities

class BaseObservationModel(ABC):
    def __init__(self, agent: BaseObject):
        self.agent = agent

    @abstractmethod
    def observe(self, environment: BaseEnvironment):
        ...

    @abstractmethod
    def estimateObservation(self, agent_pose: Pose, target_pose: Pose, check_limits:bool = True):
        ...
    
    @abstractmethod
    def linearize(self, agent_pose: Pose, target_pose: Pose) -> np.ndarray:
        ...

    @abstractmethod
    def updateVisualization(self):
        ...

class RangeBearingFoVObservationModel2D(BaseObservationModel):
    # This is characterized by its covariance matrix and fov in radians and distance
    # TODO: This actually transfers to Agents3D using 2D lidar. Right now we will limit it to 2D for isolation but most can be copied to 3D
    def __init__(self, agent: Agent2D, fov: float, max_range: float, min_range: float, sigma_d: float, sigma_w: float):
        super(RangeBearingFoVObservationModel2D, self).__init__(agent)

        self.fov = fov
        self.max_range = max_range
        self.min_range = min_range
        self.sigma_d = sigma_d
        self.sigma_w = sigma_w

        self.total_occlusions_hit = 0
        self.partial_occlusions_hit = 0
        self.lights_hit = 0

        # Visualization
        # The visualization is a circle with a radius equal to the distance
        # and an angle equal to the fov
        self.visualization = Wedge((0, 0), self.max_range, 0, 0, color='g', fill=True)
        self.visualization.set_theta1(np.rad2deg(self.agent.pose.rotation[0] - self.fov / 2))
        self.visualization.set_theta2(np.rad2deg(self.agent.pose.rotation[0] + self.fov / 2))
        self.visualization.set_alpha(0.5)

    def observe(self, environment: BaseEnvironment):
        self.observations = []

        for target in environment.targets:
            # Calculate the distance from the sensor to the target
            v_a_t = target.pose.translation - self.agent.pose.translation
            distance = np.linalg.norm(v_a_t)

            # If the target is within the max_range, add it to the observations
            if distance <= self.max_range and distance >= self.min_range:
                # Calculate the angle from the sensor to the target
                angle = np.arctan2(target.pose.translation[1] - self.agent.pose.translation[1], target.pose.translation[0] - self.agent.pose.translation[0])

                # Calculate the bearing from the sensor to the target
                bearing = wrapAngle(angle - self.agent.pose.rotation[0])

                # If the bearing is within the fov, add the observation
                if abs(bearing) <= self.fov / 2:
                    
                    # Remove the target from the occluders to avoid self occlusion
                    occluders_no_target = [occluder for occluder in environment.occluders if occluder.id != target.id]

                    # Check if the target is occluded by an obstacle
                    occluding_occluders, occ_amounts = checkOcclusions2DCircles(occluders_no_target, target, self.agent.pose)
                    noise_increment = 1

                    # If the target is occluded, there is no observation
                    if -1 in occ_amounts:
                        print("(1) Observation occluded of target {}".format(target.translation))
                        self.total_occlusions_hit += 1
                        continue
                    # If the target is partially occluded, add noise to the observation
                    else:
                        # Check if the target is partially occluded by an obstacle. Each partial occlusion adds noise
                        for occluder, occ_amount in zip(occluding_occluders, occ_amounts):
                            print("(2) Noise increased due to partial occlusion of target {}".format(target.translation))
                            self.partial_occlusions_hit += 1
                            noise_increment += occ_amount


                    noise_increment = min(2.0, noise_increment)

                    # Check for the possibility of skipping measurements
                    if np.random.uniform(1, 2) < noise_increment:
                        print("Skipping measurement of target {}".format(target.translation))
                        continue

                    # Check the backlighting of the target
                    for light in environment.lights:
                        # Check if the target is inside the light area
                        if light.contains(target.pose):
                            # Check if the target is in the back of the light
                            # This means that the andgle between line of sight and light.body_vector
                            # is greater than 45 degrees
                            cos_angle = np.dot(v_a_t, light.body_vector) / (np.linalg.norm(v_a_t) * np.linalg.norm(light.body_vector))
                            dist_a_l = np.linalg.norm(light.pose.translation - self.agent.pose.translation)
                            dist_t_l = np.linalg.norm(light.pose.translation - target.pose.translation)
                            if abs(cos_angle) < 0.7 and dist_t_l < dist_a_l:
                                noise_increment += 2.0
                                self.lights_hit += 1
                                print("(3) Noise increased due to backlight of target {}".format(target.pose.translation))
                            
                            # Check for the possibility of skipping measurements
                            if np.random.uniform(2.0, 6.0) < noise_increment:
                                print("Skipping measurement of target {}".format(target.pose.translation))
                                continue

                    # Add noise to the distance and bearing
                    noisy_distance = np.random.normal(distance, self.sigma_d * noise_increment)
                    noisy_bearing = np.random.normal(bearing, self.sigma_w * noise_increment)
                    self.observations.append(RangeBearingObservation(target.id, distance, bearing,
                                                                     noisy_distance, noisy_bearing))

        return self.observations

    def estimateObservation(self, agent_pose: Pose, target_pose: Pose, check_limits:bool = True):
        # It also does not add noise to the observation

        # Calculate the distance from the sensor to the target
        distance = np.linalg.norm(target_pose.translation - agent_pose.translation)

        # If there is no observation, return None
        if check_limits and (distance > self.max_range or distance < self.min_range):  
            # This happens if target is out of distance or FoV
            return None
    
        # Calculate the angle from the sensor to the target
        angle = np.arctan2(target_pose.translation[1] - agent_pose.translation[1], target_pose.translation[0] - agent_pose.translation[0])
        
        # Calculate the bearing from the sensor to the target accouning for the rotation limits
        bearing = wrapAngle(angle - agent_pose.rotation[0])
        
        # If the bearing is within the fov, return the observation
        # Otherwise None. Add a small margin of 0.01 radians to make
        # sure that the bearing is not exactly on the edge of the FoV
        if check_limits and abs(bearing) > self.fov / 2 - 0.1:
            # This happens if target is out of distance or FoV
            return None

        # Add the observation
        return RangeBearingObservation(0, distance, bearing,
                                    distance, bearing)

    def linearize(self, agent_pose: Pose, target_pose: Pose):
        # Linearized observation for a single target
        # The pose of the target is most probably a ESTIMATE
        # The linearization is done around the targets pose
        # with given agent pose

        # Calculate the distance from the sensor to the target. Add delta for numerical stability
        squares = (target_pose.translation[0] - agent_pose.translation[0])**2 + (target_pose.translation[1] - agent_pose.translation[1])**2 + 1e-9
        distance = np.sqrt(squares) + 1e-9

        if distance < 1.0:
            print("Distance is too small: {}".format(distance))

        # Calculate the angle from the sensor to the target
        angle = np.arctan2(target_pose.translation[1] - agent_pose.translation[1], target_pose.translation[0] - agent_pose.translation[0])

        # Calculate the bearing from the sensor to the target
        bearing = wrapAngle(angle - agent_pose.rotation[0])

        # Calculate the Jacobian x is target_x, target_y
        # The Jacobian is a 2x2 matrix: [dr/dx, dr/dy; db/dx, db/dy]
        H = np.array([
            [(target_pose.translation[0] - agent_pose.translation[0]) / distance, (target_pose.translation[1] - agent_pose.translation[1]) / distance],
            [-(target_pose.translation[1] - agent_pose.translation[1]) / squares, (target_pose.translation[0] - agent_pose.translation[0]) / squares]
        ])

        return H

    def inverseObservation(self, observation: RangeBearingObservation, noise: bool = False):
        # Calculate the position of the target
        if noise:
            x = self.agent.pose.translation[0] + observation.noisy_distance * np.cos(self.agent.pose.rotation[0] + observation.noisy_bearing)
            y = self.agent.pose.translation[1] + observation.noisy_distance * np.sin(self.agent.pose.rotation[0] + observation.noisy_bearing)
        else:
            x = self.agent.pose.translation[0] + observation.distance * np.cos(self.agent.pose.rotation[0] + observation.bearing)
            y = self.agent.pose.translation[1] + observation.distance * np.sin(self.agent.pose.rotation[0] + observation.bearing)

        return x, y

    def updateVisualization(self):
        self.visualization.set_center((self.agent.pose.translation[0], self.agent.pose.translation[1]))
        self.visualization.set_theta1(np.rad2deg(self.agent.pose.rotation[0] - self.fov / 2))
        self.visualization.set_theta2(np.rad2deg(self.agent.pose.rotation[0] + self.fov / 2))

class ProbabilisticClassification(object):
    """
        This class defines the model of the NN that classifies the target in 3 classes.
        This class works for 2D and 3D. In 3D the angle for the prediction can be understood
        as the angle between the agent's line of sight and the target's orientation in 3D.
        It is developed in the "semantic_perception_test.py" script and can be
        visualized executing that script.
        The parameters of the model are:
    """
    def __init__(self, decay_rate: float = 3, decay_distance: float = 3, optimal_distance: float = 1, back_angle: float = np.pi/4, confusing_class: int = 0, noise: float = 0.1, noise_factor: float = 5):
        self.decay_rate = decay_rate
        self.decay_distance = decay_distance
        self.back_angle = back_angle
        self.confusing_class = confusing_class
        self.noise = noise # Base noise
        self.noise_factor = noise_factor # This is how much it decays with distance
        self.fig = None
        self.ax = None
        self.configured = False
    def predict(self, target: BaseObject, distance: float, angle: float, noise_increment: float = 1.0):
        # The angle difference between the agent's line of sight direction and the target's orientation to check if it's in its back
        # Wrap the angle between -pi and pi
        orientation_diff = wrapAngle(angle - target.pose.rotation[0])

        # Determine if the target is in front or back based on the angle difference (the back of the target is from -pi/2+pi/12 to pi/2-pi/12)
        is_target_behind = abs(orientation_diff) <= self.back_angle / 2

        # Heuristics for classification probability
        # Assuming the agent is more accurate when closer to the target and facing the target from the front
        class_probs = [0.0, 0.0, 0.0]
        for class_idx in range(3):
            if class_idx == target.class_id:
                # Higher probability for the true class defined by sigmoid function
                class_probs[class_idx] = self.probability_model_correct_class(distance)
                if is_target_behind:
                    # If the target is behind, the probability is lower
                    class_probs[class_idx] /= 1000
            elif class_idx == self.confusing_class:
                # Lower probability for incorrect classes, with overconfident wrong predictions
                class_probs[class_idx] = self.probability_model_confusion_class(distance)
            else:
                # The last class has lower probability in general
                class_probs[class_idx] = self.probability_model_rest_class(distance)

        # Normalize the probabilities to sum up to 1
        prob_sum = sum(class_probs)
        prob_vector = [prob / prob_sum for prob in class_probs]

        # Add some random noise to simulate uncertainty (increasing with the distance up to decay_distance, then constant)
        if not is_target_behind:
            noise = [random.uniform(0.0, (self.noise * noise_increment) * (1 + 5*np.exp(-(distance-self.decay_distance)**2/(2*(self.decay_distance/10))))) for class_id in range(3)]
        else:
            noise = [random.uniform(0.0, self.noise) for _ in range(3)] # If we are observing from a wrong point of view, the noise doesn't really matter
        
        # With some probability, the measurement is detected as an outlier
        if np.random.uniform(0, 6) < noise_increment:
            prob_vector = [0.0, 0.0, 0.0]
            prob_vector[self.confusing_class] = 1.0 - np.random.uniform(0.0, 0.1)
            return prob_vector
        
        # Add the noise to the probabilities
        prob_vector = prob_vector + np.array(noise)
        prob_sum = sum(prob_vector)
        prob_vector = [prob / prob_sum for prob in prob_vector]

        return prob_vector
    
    """
        The models below return the probability of the target being in the correct class
        given the distance to the target. The probability is calculated using a sigmoid.
        You can visualize the result with the plot function.
    """
    
    def probability_model_correct_class(self, distance: float):
        prob = 1 / (1 + np.exp(distance*self.decay_rate - self.decay_distance*self.decay_rate)) + 0.1
        return prob

    def probability_model_confusion_class(self, distance: float):
        prob = 1 / (1 + np.exp(-(distance*self.decay_rate - self.decay_distance*self.decay_rate))) + 0.1
        return prob

    def probability_model_rest_class(self, distance: float):
        prob = 1 / (5*(1 + np.exp(distance*self.decay_rate/2 - self.decay_distance*self.decay_rate/2))) + 0.1
        return prob
    
    def plot(self, target):
        # Plot the probability model of the correct class for a given target 
        # position and asuming agent is always looking towards the target
        # Represent a grid of 8x8 meters around the target
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
        x = np.linspace(target.pose.x - 4, target.pose.x + 4, 100)
        y = np.linspace(target.pose.y - 4, target.pose.y + 4, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                distance = np.sqrt((target.pose.x - X[i, j]) ** 2 + (target.pose.y - Y[i, j]) ** 2)
                angle = np.arctan2(target.pose.y - Y[i, j], target.pose.x - X[i, j])
                Z[i, j] = self.predict(target, distance, angle)[target.class_id]
        sem_map = self.ax.imshow(Z, cmap='viridis', origin='lower', extent=(-4, 4, -4, 4))
        if not self.configured:
            self.fig.colorbar(label='Probability', mappable=sem_map)
            self.configured = True
        self.ax.set_xlabel('Agent X Position')
        self.ax.set_ylabel('Agent Y Position')
        self.ax.set_title('Classification Probabilities Heatmap')
        self.fig.show()

class SemanticFoVObservationModel2D(BaseObservationModel):
    # This is characterized by its covariance matrix and fov in radians and distance
    def __init__(self, agent: Agent2D, fov: float, max_range: float, min_range: float, num_classes: int):
        super(SemanticFoVObservationModel2D, self).__init__(agent)

        self.fov = fov
        self.max_range = max_range
        self.min_range = min_range
        self.num_classes = num_classes
        self.classifier = ProbabilisticClassification()

        self.total_occlusions_hit = 0
        self.partial_occlusions_hit = 0
        self.lights_hit = 0

        # Visualization
        # The visualization is a circle with a radius equal to the distance
        # and an angle equal to the fov
        self.visualization = Wedge((0, 0), self.max_range, 0, 0, color='g', fill=True)
        self.visualization.set_theta1(np.rad2deg(self.agent.pose.rotation[0] - self.fov / 2))
        self.visualization.set_theta2(np.rad2deg(self.agent.pose.rotation[0] + self.fov / 2))
        self.visualization.set_alpha(0.5)
        
    def observe(self, environment: BaseEnvironment):
        self.observations = []

        for target in environment.targets:
            # Calculate the distance from the sensor to the target for noise
            v_a_t = target.pose.translation - self.agent.pose.translation
            distance = np.linalg.norm(v_a_t)

            # If the target is within the max_range, add it to the observations
            if distance <= self.max_range and distance >= self.min_range:
                # Calculate the angle of Line-of-sight from the sensor to the target
                # (in the agent's frame of reference)
                angle = np.arctan2(target.pose.translation[1] - self.agent.pose.translation[1], target.pose.translation[0] - self.agent.pose.translation[0])

                # Calculate the bearing from the sensor to the target
                bearing = wrapAngle(angle - self.agent.pose.rotation[0])
                
                # If the bearing is within the fov, add the observation
                if abs(bearing) <= self.fov / 2:

                    # Remove the target from the occluders to avoid self occlusion
                    occluders_no_target = [occluder for occluder in environment.occluders if occluder.id != target.id]

                    # Check if the target is occluded by an obstacle
                    occluding_occluders, occ_amounts = checkOcclusions2DCircles(occluders_no_target, target, self.agent.pose)
                    noise_increment = 1

                    # If the target is occluded, there is no observation
                    if -1 in occ_amounts:
                        print("(1) Observation occluded of target {}".format(target.translation))
                        self.total_occlusions_hit += 1
                        continue
                    # If the target is partially occluded, add noise to the observation
                    else:
                        # Check if the target is partially occluded by an obstacle. Each partial occlusion adds noise
                        for occluder, occ_amount in zip(occluding_occluders, occ_amounts):
                            print("(2) Noise increased due to partial occlusion of target {}".format(target.translation))
                            self.partial_occlusions_hit += 1
                            noise_increment += occ_amount


                    noise_increment = min(4.0, noise_increment)

                    # Check for the possibility of skipping measurements
                    if np.random.uniform(1, 3) < noise_increment:
                        print("Skipping measurement of target {}".format(target.translation))
                        continue

                    # Check the backlighting of the target
                    for light in environment.lights:
                        # Check if the target is inside the light area
                        if light.contains(target.pose):
                            # Check if the target is in the back of the light
                            # This means that the andgle between line of sight and light.body_vector
                            # is greater than 45 degrees
                            cos_angle = np.dot(v_a_t, light.body_vector) / (np.linalg.norm(v_a_t) * np.linalg.norm(light.body_vector))
                            dist_a_l = np.linalg.norm(light.pose.translation - self.agent.pose.translation)
                            dist_t_l = np.linalg.norm(light.pose.translation - target.pose.translation)
                            if abs(cos_angle) < 0.7 and dist_t_l < dist_a_l:
                                noise_increment += 2.0
                                self.lights_hit += 1
                                print("(3) Noise increased due to backlight of target {}".format(target.pose.translation))
                            
                            # Check for the possibility of skipping measurements
                            if np.random.uniform(2.0, 6.0) < noise_increment:
                                print("Skipping measurement of target {}".format(target.pose.translation))
                                continue

                    # Now we simulate a NN that classifies the target
                    prob_vector = self.classifier.predict(target, distance, angle, noise_increment)
                    # self.classifier.plot(target)
                    # Add the observation
                    self.observations.append(SemanticObservation(target.id, prob_vector))

        return self.observations


    def estimateObservation(self, agent_pose, target_pose, target_probabilities, optimal_distance, check_limits=False):
        # It also does not add noise to the observation

        # Calculate the distance from the sensor to the target
        distance = np.linalg.norm(target_pose.translation - agent_pose.translation)

        # If there is no observation, return None
        if check_limits and (distance > self.max_range or distance < self.min_range):  
            # This happens if target is out of distance or FoV
            return None
    
        # Calculate the angle from the sensor to the target
        angle = np.arctan2(target_pose.translation[1] - agent_pose.translation[1], target_pose.translation[0] - agent_pose.translation[0])
        
        # Calculate the bearing from the sensor to the target
        bearing = wrapAngle(angle - agent_pose.rotation[0])
        
        # If the bearing is within the fov, return the observation
        # Otherwise None
        if check_limits and abs(bearing) > self.fov / 2:
            # This happens if target is out of distance or FoV
            return None

        # Assume that the probability is the current one with some regularization
        num_classes = len(target_probabilities)
        current_target_class = np.random.choice(np.where(target_probabilities == target_probabilities.max())[0])

        # Calculate the probability of the current class
        if(distance < optimal_distance):
            prob_current_class = -((distance - optimal_distance)/ optimal_distance)**2 + 1
        else:
            prob_current_class = max(1.0/num_classes+0.1, -((distance - optimal_distance)/ (optimal_distance/4))**2 + 1)
        
        # prob_current_class = 0.7 # This is the alternative used in the paper

        # Calculate the probability of the other classes
        prob_other_classes = (1 - prob_current_class) / (num_classes - 1)

        # Create the probability vector
        prob_vector = np.zeros(num_classes)
        for i in range(num_classes):
            if i == current_target_class:
                prob_vector[i] = prob_current_class
            else:
                prob_vector[i] = prob_other_classes

        # Add the observation
        return SemanticObservation(0, prob_vector)

    def linearize(self, agent_pose, target_pose):
        # Linearized observation for a single target
        # The lienariazation is done around the targets pose
        # As it depends on the distance, it is not linear
        # TODO: Compute the jacobian for the probability above
        pass

    def inverseObservation(self, observation, noise=False):
        # Calculate the class of the target from the observation
        # NOT implemented
        pass
    
    def updateVisualization(self):
        self.visualization.set_center((self.agent.pose.translation[0], self.agent.pose.translation[1]))
        self.visualization.set_theta1(np.rad2deg(self.agent.pose.rotation[0] - self.fov / 2))
        self.visualization.set_theta2(np.rad2deg(self.agent.pose.rotation[0] + self.fov / 2))