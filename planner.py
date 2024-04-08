import math
import random
import copy
from tqdm import tqdm
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines

from abc import ABC, abstractmethod
from basics import Pose2D, wrapAngle
from belief import ExtendedKalmanFilter
from objects import Agent2D
from perception import RangeBearingFoVObservationModel

class BasePlanner(ABC):
    def __init__(self, agent):
        self.agent = agent
        pass

    @abstractmethod
    def plan(self, dt):
        pass

    @abstractmethod
    def update(self, dt):
        pass
    
class RandomPlanner(BasePlanner):
    """
        Random planner plans 50 steps with constant speed
        which can change with increasing probability
    """

    def __init__(self, agent):
        super(RandomPlanner, self).__init__(agent)

        self.steps = 50
        self.step = 0

    def plan(self):
        self.step = 0

    def update(self):
        if self.step < self.steps:
            self.step += 1
            # Change the velocity with probability 0.1
            if np.random.rand() < 0.3:
                v = self.agent.dynamic.v + np.random.rand() * 0.1 - 0.05
                w = self.agent.dynamic.w + np.random.rand() * 0.1 - 0.05
                self.agent.dynamic.setVelocity(v, w)
        else:
            # Replan
            self.plan()


class RandomPlannerTrajectory(BasePlanner):
    """
        Random planner generates 10 waypoints in the map and
        checks if the robot has reached one of them.
    """

    def __init__(self, agent, low_level_planner):
        super(RandomPlannerTrajectory, self).__init__(agent)

        self.waypoints = []
        self.waypoint = 0
        self.num_waypoints = 10
        self.low_level_planner = low_level_planner
        self.visualization = lines.Line2D([], [], color='black')

    def plan(self):
        self.waypoint = 0

        # Generate 10 waypoints
        self.waypoints = []
        for i in range(self.num_waypoints):
            x = np.random.rand() * 10
            y = np.random.rand() * 10
            self.waypoints.append((x, y))

        self.updateVisualization()
        # Send the first waypoint to the low level planner
        x, y = self.waypoints[self.waypoint]
        self.low_level_planner.setGoal(Pose2D(x, y, 0))
        self.low_level_planner.plan()

    def update(self):
        # Check if the robot has reached the waypoint
        x, y = self.waypoints[self.waypoint]
        if np.linalg.norm(np.array([x, y]) - np.array([self.agent.pose.x, self.agent.pose.y])) < 0.1:
            self.waypoint += 1
            if self.waypoint < len(self.waypoints):
                # Set the goal for the low level planner
                x, y = self.waypoints[self.waypoint]
                self.low_level_planner.setGoal(Pose2D(x, y, 0))
                self.low_level_planner.plan()
            else:
                # Replan
                self.plan()
        else:
            self.low_level_planner.update()

    def updateVisualization(self):
        # Add points and splines to the visualization
        self.visualization.set_xdata([x for x, y in self.waypoints])
        self.visualization.set_ydata([y for x, y in self.waypoints])

class IPPState(object):
    def __init__(self, remaining_budget, viewpoint, beliefs):
        # Make copies of references to keep them isolated:

        self.viewpoint = viewpoint.copyPose()

        self.remaining_budget = remaining_budget
        self.beliefs = []
        for belief in beliefs:
            self.beliefs.append(belief.copyBelief())
        # print("State created with viewpoint {}".format(self.viewpoint))

class MCTSNode(object):
    def __init__(self, node_id, depth, parent, state: IPPState, actions, reward, gamma):
        
        self.node_id = node_id
        self.depth = depth
        self.parent = parent
        self.state = state
        self.actions = actions
        self.gamma = gamma
        self.reward = reward
        self.value = 0
        self.visits = 0
        self.children = []

    def isTerminal(self):
        """
            Returns if the node is a terminal state.
        """
        return self.state.remaining_budget <= 0

    def isLeaf(self):
        """
            Returns if the node is a leaf node.
        """
        # Careful because visited nodes in MCTS can still be leaf
        # by not having children
        return len(self.children) == 0

    def addChild(self, node):
        self.children.append(node)

    def computeStateExpansionFromAction(self, action, isViewpointValid):
        """
            Computes the child node from this and an action.
            The reward is also returned.
            TODO: This is the bottleneck
        """

        new_viewpoint = self.state.viewpoint.copyPose()
        new_viewpoint.x += action[0]
        new_viewpoint.y += action[1]
        new_viewpoint.theta += action[2]
        # Normalize theta between -pi and pi
        new_viewpoint.theta = wrapAngle(new_viewpoint.theta)

        if not isViewpointValid(new_viewpoint):
            return None, None

        # Update budget with correct travel cost
        new_budget = self.state.remaining_budget - 1
        new_beliefs = []
        reward = 0
        for belief in self.state.beliefs:
            new_belief = belief.copyBelief()
            new_beliefs.append(new_belief)
            
            if ("KalmanFilter" in new_belief.__class__.__name__ and  new_belief.getEntropy() < -15) or ("Semantic" in new_belief.__class__.__name__ and new_belief.getEntropy() < 0.23):
                    continue
            
            is_update = new_belief.simulateUpdate(new_viewpoint)
            # If there is no update, then reward is related to the distance
            # between the viewpoint and the target but smaller

            if not is_update:
                # distance = np.linalg.norm(np.array([new_viewpoint.x, new_viewpoint.y]) - np.array([belief.position[0], belief.position[1]]))
                # direction_angle = np.arctan2(belief.position[1] - new_viewpoint.y, belief.position[0] - new_viewpoint.x)
                # angular_distance = abs(direction_angle - new_viewpoint.theta)
                # utility = 1/(distance + angular_distance + 0.01)*new_belief.getEntropy()
                continue
            else:    
                prev_entropy = belief.getEntropy()
                new_entropy = new_belief.getEntropy()
                # TODO: reward is the utility normalized by travel time (action cost)
                utility = (prev_entropy - new_entropy)
            
                # If the pose to the targt is too close, utility is 0
                if np.linalg.norm(np.array([new_viewpoint.x, new_viewpoint.y]) - np.array([belief.position[0], belief.position[1]])) < 0.1:
                    utility = 0
            
            reward += utility
        
        # Add cost as the distance between the viewpoints in x, y and theta
        # cost = 0.1*(np.linalg.norm(np.array([action[0], action[1]])) + 1)
        # reward = reward #/ cost
        return IPPState(new_budget, new_viewpoint, new_beliefs), reward
        
    def computeExpandedChildrenAllActions(self, max_node_id, isViewpointValid):
        """
            Compute the children in a non-destructive way.
            This means that the max_node_id and self.children are not modified.
            It also means that they should be modified above
        """
        children = []
        c_max_node_id = max_node_id

        for action in self.actions:
            new_state, reward = self.computeStateExpansionFromAction(action, isViewpointValid)
            if new_state is None:
                continue
            # The reward has to be corrected by the gamma value
            discount = self.gamma ** (self.depth + 1)
            correct_reward = self.reward + discount * reward
            child = MCTSNode(max_node_id, self.depth+1, self, new_state, self.actions, correct_reward, self.gamma)
            
            children.append(child)

        return children

    def computeExpandedChildOneAction(self, action, max_node_id, isViewpointValid):
        """
            Compute the children in a non-destructive way.
            This means that the max_node_id and self.children are not modified.
            It also means that they should be modified above
        """
        new_state, reward = self.computeStateExpansionFromAction(action, isViewpointValid)
        if new_state is None:
            return None
        # The reward has to be corrected by the gamma value
        discount = self.gamma ** (self.depth + 1)
        correct_reward = self.reward + discount * reward
        child = MCTSNode(max_node_id, self.depth+1, self, new_state, self.actions, correct_reward, self.gamma)
        return child

    def expandAllActions(self, max_node_id, isViewpointValid):
        """
            Compute the children in a destructive way.
            This means that the max_node_id and self.children are modified.
            Children are directly integrated.
        """
        children = []
        for action in self.actions:
            # print("Action: {}".format(action))
            # print("State: {}".format(self.state.viewpoint))
            new_state, reward = self.computeStateExpansionFromAction(action, isViewpointValid)
            if new_state is None:
                continue
            # The reward has to be corrected by the gamma value
            discount = self.gamma ** (self.depth + 1)
            correct_reward = self.reward + discount * reward
            child = MCTSNode(max_node_id, self.depth+1, self, new_state, self.actions, correct_reward, self.gamma)
            max_node_id += 1
            self.addChild(child)
        
        return max_node_id

    def selectUctChild(self, c: float = 2.0):
        """
            Selects the child with the highest UCT value
            In case that more than one has the same, pick
            the first with the highest. This is: exclusive
            greater when comparing values
            N: is the total number of simulations of parent
            n: is the number of simulations of child node if it was visited
        """
        # Init with first node
        max_children = [self.children[0]]
        if max_children[0].visits == 0:
            max_uct = np.inf
        else:
            max_uct = max_children[0].value + c * np.sqrt(np.log(self.visits) / (max_children[0].visits))

        for child in self.children:
            if child.visits == 0:
                uct = np.inf
            else:
                uct = child.value + c * np.sqrt(np.log(self.visits) / (child.visits+1))
            if uct > max_uct:
                max_children = [child]
                max_uct = uct
            elif uct == max_uct:
                max_children.append(child)

        # Select randomly one of the children with the highest UCT
        child = random.choice(max_children)
        
        return child
        
    def selectBestChild(self):
        """
            Selects the child with the highest value
        """
        # Init with first node
        max_children = [self.children[0]]
        max_value = max_children[0].value

        for child in self.children:
            value = child.value
            if value > max_value:
                max_children = [child]
                max_value = value
            elif value == max_value:
                max_children.append(child)

        # Select randomly one of the children with the highest value
        child = random.choice(max_children)
        
        return child

class MonteCarloTreeSearchPlanner(BasePlanner):
    """
        Plan next goal based on Monte Carlo Tree Search
    """

    def __init__(self, agent, low_level_planner, beliefs, budget, env):
        super(MonteCarloTreeSearchPlanner, self).__init__(agent)
        # Members
        self.agent = agent # The planner controls one agent
        self.low_level_planner = low_level_planner # It communicates with a low level planner
        self.beliefs = beliefs # It can plan to optimize several beliefs
        self.budget = budget # The budget of the planner
        self.next_goal = None # The next goal to reach
        self.nodes = {} # The nodes of the tree 
        self.max_node_id = 0 # The maximum node id of the tree
        self.visualization = lines.Line2D([], [], color='black')

        # Get the function isViewpointValid from env
        self.isViewpointValid = env.isPositionValid

        # Set MCTS parameters
        self.horizon_length = 3 # Num of rollouts. NOT USED right now
        self.exploration_constant = 2.0 # Exploration constant for UCT. NOT USED right now
        self.gamma = 0.95 # Discount factor for the reward estimation
        self.eps_greedy_prob = 0.3 # Probability of selecting a random action # TODO: is it in rollout or traversal?  # atm random used since much faster

        # Define actions
        # For each pose, the action is a tuple (x, y, theta)
        # which are possible movements from a pose
        self.actions = []
        resolution = 1
        max_reach = 1
        self.action_space_x = np.arange(-max_reach, max_reach + resolution, resolution)
        self.action_space_y = np.arange(-max_reach, max_reach + resolution, resolution)
        angular_resolution = np.pi/2
        self.action_space_theta = np.arange(-np.pi + angular_resolution, np.pi + angular_resolution, angular_resolution)

        # Create a list of all possible actions
        for x in self.action_space_x:
            for y in self.action_space_y:
                for theta in self.action_space_theta:
                    self.actions.append((x, y, theta))
        # Delete not moving action
        self.actions.remove((0, 0, 0))
        print("Searching in {} actions".format(len(self.actions)))

    def plan(self, num_steps):
        # Init
        self.step = 0
        self.nodes = {}
        self.max_node_id = 0

        # Start tree with root node at current state
        root = MCTSNode(self.max_node_id, 0, None, IPPState(self.budget, self.agent.pose, self.beliefs), self.actions, 0, self.gamma)
        self.current_node = root
        self.max_node_id += 1
        self.nodes = {root.node_id: root}

        # Expand the root node before any step
        self.current_node.expandAllActions(self.max_node_id, self.isViewpointValid)
        # Register all nodes in the tree
        for child in self.current_node.children:
            self.nodes[child.node_id] = child
        # Start MCTS
        for step in tqdm(range(num_steps)):
            self.planStep()
            self.step += 1

        # Select best child node as action
        self.next_goal = root.selectBestChild().state.viewpoint
        self.updateVisualization()
        # Target chosen goal pose
        self.low_level_planner.setGoal(self.next_goal)

    def planStep(self):
        # Simulate
        # MCTS Steps
        # 1. Selection
        # 2. Expansion
        # 3. Simulation
        # 4. Backpropagation
        # Start from the top and select
        # TIME THIS
        while not self.current_node.isLeaf():
            self.current_node = self.current_node.selectUctChild()
        # TODO: Check if depth is equal to budget. Then, it's terminal state
        # Expansion OR rollout
        # The value for the resulting simulation. It is added in the backpropagation phase
        value = 0
        if self.current_node.visits == 0:
            # Simulation
            value = self.rollout(self.current_node)
        else:
            self.max_node_id = self.current_node.expandAllActions(self.max_node_id, self.isViewpointValid)
            # Register all nodes in the tree
            for child in self.current_node.children:
                self.nodes[child.node_id] = child
            self.current_node = self.current_node.selectUctChild()
            # Simulation
            value = self.rollout(self.current_node)
        # Backpropagate
        while self.current_node.parent != None:
            self.current_node.visits += 1
            self.current_node.value += (value - self.current_node.value) / self.current_node.visits
            self.current_node = self.current_node.parent

        # Update root node
        self.current_node.visits += 1
        self.current_node.value += (value - self.current_node.value) / self.current_node.visits
        return

    def update(self):

        if self.budget <= 0:
            print("Budget exhausted")
            self.low_level_planner.stop()
            # Shut down the program
            exit(0)
            return
        
        # Check if the robot has reached the next goal
        x = self.next_goal.x
        y = self.next_goal.y
        theta = self.next_goal.theta
        # Consider angles next to -pi and pi
        theta2 = theta + 2 * np.pi
        if np.linalg.norm(np.array([x, y]) - np.array([self.agent.pose.x, self.agent.pose.y])) < 0.1 and (abs(theta - self.agent.pose.theta) < 0.1 or abs(theta2 - self.agent.pose.theta) < 0.1):
            # If the robot has reached the goal, plan again for the next goal
            self.budget -= 1
            self.plan(100)
            self.low_level_planner.update()
            return True
        else:
            # Otherwise, continue executing the low-level controller
            self.low_level_planner.update()
            return False

    def rollout(self,node):
        """
            Rollout actually uses nodes instead of states because
            reward and actions are stored in them
        """
        current_node_rollout = node
        value = 0
        horizon_step = 0
        while not current_node_rollout.isTerminal() and horizon_step < self.horizon_length:
            # Pick a random valid action
            # action_id = np.random.choice(len(self.actions))
            next_node = self.eps_greedy_policy(current_node_rollout, self.isViewpointValid)
            if next_node is None:
                break
            current_node_rollout = next_node
            horizon_step += 1
        value = current_node_rollout.reward
        return value

    def eps_greedy_policy(self, node, isViewpointValid):
        if np.random.uniform(0, 1) >= self.eps_greedy_prob:
            next_node = self.greedy_action_node(node, isViewpointValid)
            if next_node is None:
                return None
        else:
            tried_actions = {}
            next_action = np.random.choice(len(self.actions))
            tried_actions[next_action] = True
            next_state, reward = node.computeStateExpansionFromAction(self.actions[next_action], isViewpointValid)
            # Try actions to find a valid one
            while (next_state is None) or (not isViewpointValid(next_state.viewpoint)):
                while next_action in tried_actions:
                    next_action = np.random.choice(len(self.actions))
                    tried_actions[next_action] = True
                    if (len(tried_actions) == len(self.actions)):
                        return None
                next_state, reward = node.computeStateExpansionFromAction(self.actions[next_action], self.isViewpointValid)

            # If no valid actions, return None
            if not self.isViewpointValid(next_state.viewpoint):
                return None
            else:
                next_node = node.computeExpandedChildOneAction(self.actions[next_action], 0, isViewpointValid)
        
        return next_node

    def greedy_action_node(self, node, isViewpointValid):
        children = node.computeExpandedChildrenAllActions(self.max_node_id, isViewpointValid)
        # Shuffle children to add randomness in case
        # that max reward is the same for several children
        np.random.shuffle(children)

        max_reward = -np.inf
        for id, child in enumerate(children):
            if child.reward > max_reward:
                max_reward = child.reward
                max_reward_idx = id
        return children[max_reward_idx]

    def updateVisualization(self):
        # Plot the tree for the previous plan

        # Get the root node
        root = self.nodes[0]
        # Get the all the lines to plot
        lines_x = []
        lines_y = []
        for node in self.nodes.values():
            if node.parent is not None:
                lines_x.append([node.parent.state.viewpoint.x, node.state.viewpoint.x])
                lines_y.append([node.parent.state.viewpoint.y, node.state.viewpoint.y])

        # Add points and splines to the visualization
        self.visualization.set_xdata(lines_x)
        self.visualization.set_ydata(lines_y)

class GreedyPlanner(BasePlanner):
    def __init__(self, agent, low_level_planner, beliefs, budget, env, is_visualize=False):
        super(GreedyPlanner, self).__init__(agent)
        # Members
        self.agent = agent # The planner controls one agent
        self.low_level_planner = low_level_planner # It communicates with a low level planner
        self.beliefs = beliefs # It can plan to optimize several beliefs
        self.budget = budget # The budget of the planner
        self.next_goal = None # The next goal to reach
        self.sample_number = 100

        self.fig = None
        self.ax = None
        self.configured = False

        self.is_visualize = is_visualize
        # Get the function isViewpointValid from env
        self.isViewpointValid = env.isPositionValid

        # Define actions
        # For each pose, the action is a tuple (x, y, theta)
        # which are possible movements from a pose
        self.actions = []
        resolution = 1
        self.max_reach = 2
        self.action_space_x = np.arange(-self.max_reach, self.max_reach + resolution, resolution)
        self.action_space_y = np.arange(-self.max_reach, self.max_reach + resolution, resolution)
        angular_resolution = np.pi/2
        self.action_space_theta = np.arange(-np.pi/2, np.pi + angular_resolution, angular_resolution)

        # Create a list of all possible actions
        # for x in self.action_space_x:
        #     for y in self.action_space_y:
        #         for theta in self.action_space_theta:
        #             self.actions.append((x, y, theta))
        
        # sampling the action space
        self.sampleActionSpace(self.sample_number)

        # Delete not moving action
        # self.actions.remove((0, 0, 0))

    def sampleActionSpace(self, n_samples):
        return self.sampleActionSpaceRandom(n_samples)
    
    def sampleActionSpaceRandom(self, n_samples):
        self.actions.clear()
        for i in range(n_samples):
            self.actions.append((np.random.uniform(-self.max_reach, self.max_reach), np.random.uniform(-self.max_reach, self.max_reach), np.random.uniform(-np.pi/2, np.pi)))
        # Create a list of all possible actions
        # for x in self.action_space_x:
        #     for y in self.action_space_y:
        #         for theta in self.action_space_theta:
        #             self.actions.append((x, y, theta))

    def sampleActionSpaceAround(self, n_samples, angles_around=4, distances_around=3):
        """
            This sampling strategy loops over the beliefs and samples around the target
            If the target is far away, it samples viewpoints in the region close to the target
        """
        self.actions.clear()
        for belief in self.beliefs:
            target_position = np.array([belief.position[0], belief.position[1]])
            target_distance = np.linalg.norm(target_position - np.array([self.agent.pose.x, self.agent.pose.y]))
            if target_distance < np.sqrt(2) * target_distance:
                # Sample around the target position
                angles = np.linspace(-np.pi, np.pi, angles_around+1)
                distances = np.linspace(1.0, 4.0, distances_around+1)
                for angle in angles:
                    for distance in distances:
                        x_t = target_position[0] + distance * np.cos(angle)
                        y_t = target_position[1] + distance * np.sin(angle)
                        theta_t = angle
                        # Compute the action to reach that viewpoints
                        x = x_t - self.agent.pose.x
                        y = y_t - self.agent.pose.y
                        theta = wrapAngle(theta_t - self.agent.pose.theta - np.pi)
                        self.actions.append((x, y, theta))
            else:
                # Sample near the maximum reach in the direction of the target
                direction = np.arctan2(target_position[1] - self.agent.pose.y, target_position[0] - self.agent.pose.x)
                sampling_center = np.array([self.agent.pose.x, self.agent.pose.y]) + np.array([0.7 * self.max_reach * np.cos(direction), 0.7 * self.max_reach * np.sin(direction)])

                for i in range(10):
                    distance = i * 0.1
                    x_t = sampling_center[0] + np.random.uniform(-1.0, 1.0)
                    y_t = sampling_center[1] + np.random.uniform(-1.0, 1.0)
                    while not self.isViewpointValid(Pose2D(x_t, y_t, 0)):
                        x_t = sampling_center[0] + np.random.uniform(-1.0, 1.0)
                        y_t = sampling_center[1] + np.random.uniform(-1.0, 1.0)
                    theta_t = wrapAngle(direction + np.random.uniform(-np.pi/4, np.pi/4))
                    # Compute the action to reach that viewpoints
                    x = x_t - self.agent.pose.x
                    y = y_t - self.agent.pose.y
                    theta = wrapAngle(theta_t - self.agent.pose.theta)
                    self.actions.append((x, y, theta))

        print("Searching in {} actions".format(len(self.actions)))

    def computeStateExpansionFromAction(self, action):
        """
            Computes the child node from this and an action.
            The reward is also returned.
        """
        new_viewpoint = self.agent.pose.copyPose()

        new_viewpoint.x += action[0]
        new_viewpoint.y += action[1]
        new_viewpoint.theta += action[2]
        # Normalize theta between -pi and pi
        new_viewpoint.theta = wrapAngle(new_viewpoint.theta)

        new_beliefs = []
        reward = 0

        if not self.isViewpointValid(new_viewpoint):
            return None, reward
        
        num_updates = 0
        for belief in self.beliefs:
            new_belief = belief.copyBelief()
            new_beliefs.append(new_belief)
            belief_type = new_belief.__class__.__name__
            if 'Kalman' in belief_type and new_belief.getEntropy() < -12:
                continue
            elif 'Semantic' in belief_type and new_belief.getEntropy() < 0.23: 
                continue

            utility = 0
            is_update = new_belief.simulateUpdate(new_viewpoint)
            if not is_update:
            # If there is no update, then reward is related to the distance
            # between the viewpoint and the target but smaller
                distance = np.linalg.norm(np.array([new_viewpoint.x, new_viewpoint.y]) - np.array([belief.position[0], belief.position[1]]))
                if distance < 0.4:
                    continue
                direction_angle = np.arctan2(belief.position[1] - new_viewpoint.y, belief.position[0] - new_viewpoint.x)
                angular_distance = abs(direction_angle - new_viewpoint.theta)
                utility = distance/(distance)**2 * 0.00001
                #continue
            else:
                num_updates += 1
                prev_entropy = belief.getEntropy()
                new_entropy = new_belief.getEntropy()
                utility = (prev_entropy - new_entropy)

                # If the pose to the target is too close, utility is 0
                if np.linalg.norm(np.array([new_viewpoint.x, new_viewpoint.y]) - np.array([belief.position[0], belief.position[1]])) < 0.1:
                    utility = 0

            #print("Updates: {}, Utility: {}".format(num_updates, utility))
            reward += utility
        
        # Add cost as the distance between the viewpoints in x, y and theta
        # cost = 0.1*(np.linalg.norm(np.array([action[0], action[1]])) + 1)
        # reward /= cost

        return IPPState(0, new_viewpoint, new_beliefs), reward

    def plan(self):
        # sampling the action space
        self.sampleActionSpace(self.sample_number)

        # Check the reward for each action
        self.future_states = {}
        for action in self.actions:
            new_state, reward = self.computeStateExpansionFromAction(action)
            if new_state is None or not self.isViewpointValid(new_state.viewpoint):
                continue
            self.future_states[action] = (reward, new_state)

        #print("Found {} possible states".format(len(self.future_states)))
        # for state in self.future_states.values():
        #     print(state[0], state[1].viewpoint.x, state[1].viewpoint.y, state[1].viewpoint.theta)

        # Select best next viewpoint comparing the reward
        max_reward = -np.inf
        for action, (reward, state) in self.future_states.items():
            if reward > max_reward:
                max_reward = reward
                self.next_goal = state.viewpoint
        
        # print("Selected state: {}".format(self.next_goal))
        # print("With reward {}".format(max_reward))
        if self.is_visualize:
            self.visualize()
            
        # Target chosen goal pose
        self.low_level_planner.setGoal(self.next_goal)
        self.low_level_planner.update()

    def update(self):
        # One step was made, so we need to update the budget
        if self.budget <= 0:
            print("Budget exhausted")
            self.low_level_planner.stop()
            return
        
        # Check if the robot has reached the next goal
        x = self.next_goal.x
        y = self.next_goal.y
        theta = self.next_goal.theta
        # Consider angles next to -pi and pi
        theta2 = theta + 2 * np.pi
        if np.linalg.norm(np.array([x, y]) - np.array([self.agent.pose.x, self.agent.pose.y])) < 0.1 and (abs(theta - self.agent.pose.theta) < 0.1 or abs(theta2 - self.agent.pose.theta) < 0.1):
            # If the robot has reached the goal, plan again for the next goal
            self.budget -= 1
            self.plan()
            self.low_level_planner.update()
            return True
        else:
            # Otherwise, continue executing the low-level controller
            self.low_level_planner.update()
            return False
        
    def visualize(self):
        # Plot the computed states with a label that shows: x, y, theta and reward
        # Displace slightly the viewpoint in the direction of the theta
        # Displace the label in the direction of the theta and draw an arrow in the direction of the theta
        # The information is stored in self.future_states
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()

        # Clear previous plots
        self.ax.clear()
        x = []
        y = []
        arrows = []
        c = []
        # Get 100 best states
        best_future_states = sorted(self.future_states.items(), key=lambda x: x[1][0], reverse=True)[:100]
        
        for action, (reward, state) in best_future_states:
            state_x = state.viewpoint.x
            state_y = state.viewpoint.y
            state_theta = state.viewpoint.theta
            # Displace the viewpoint
            state_x += 0.01 * np.cos(state_theta)
            state_y += 0.01 * np.sin(state_theta)
            # Displace the label
            label_state_x = state_x + 0.1 * np.cos(state_theta)
            label_state_y = state_y + 0.1 * np.sin(state_theta)
            self.ax.text(label_state_x, label_state_y, "{:.2f}".format(reward))
            # Draw an arrow
            arrow = self.ax.arrow(state_x, state_y, 0.1 * np.cos(state_theta), 0.1 * np.sin(state_theta))
            x.append(state_x)
            y.append(state_y)
            arrows.append(arrow)
            c.append(reward)

        self.ax.scatter(x, y, c=c)
        # Plot targets from beliefs and their numbers in red bold text
        for belief in self.beliefs:
            self.ax.scatter(belief.position[0], belief.position[1], color='red')
            self.ax.text(belief.position[0], belief.position[1], str(belief.id), color='red', weight='bold')

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("Planner")
        self.fig.show()

# class GreedyTrajectoryPlannerNode(object):
#     """
#         Stores the value, depth and parent of a node in the tree
#     """
#     def __init__(self, value, depth, parent):
#         self.value = value
#         self.depth = depth
#         self.parent = parent
#         self.children = []

class LowLevelPlanner(BasePlanner):
    """
        Low level planner plans the speed of the agent
        to reach the goal with constant speed
    """

    def __init__(self, agent):
        super(LowLevelPlanner, self).__init__(agent)

        self.goal = None
        self.v = 0.2
        self.w = 0.0
        self.kv = 0.5
        self.kw = 1.0

    def setGoal(self, goal):
        self.goal = goal

    def plan(self):
        # Calculate the goal angle
        e_x = self.goal.x - self.agent.pose.x
        e_y = self.goal.y - self.agent.pose.y
        e_d = np.sqrt(e_x**2 + e_y**2)
        e_th_goal = self.goal.theta - self.agent.pose.theta
        # Take the shortest angle if it is negative
        if e_th_goal < -np.pi:
            e_th_goal += 2 * np.pi
        elif e_th_goal > np.pi:
            e_th_goal -= 2 * np.pi
        e_th_d = np.arctan2(e_y, e_x) - self.agent.pose.theta
        # Take the shortest angle if it is negative
        if e_th_d < -np.pi:
            e_th_d += 2 * np.pi
        elif e_th_d > np.pi:
            e_th_d -= 2 * np.pi
        # If we are far, bear to the goal
        if abs(e_d) > 0.1:
            if e_th_d > 0.05:
                self.v = 0.0
                self.w = self.kw * e_th_d
            else:
                self.v = self.kv * e_d
                self.w = self.kw * e_th_d
        else:
            # If we are close, align with the goal
            self.v = 0.0
            self.w = self.kw * e_th_goal
        
    def stop(self):
        self.v = 0.0
        self.w = 0.0

    def update(self):
        self.plan()
        self.agent.dynamic.setVelocity(self.v, self.w)

class PosesLowLevelPlanner(BasePlanner):
    """
        Low level planner that moves the agent to the specified goal
    """

    def __init__(self, agent):
        super(PosesLowLevelPlanner, self).__init__(agent)

        self.goal = None
        self.agent.dynamic.setVelocity(0.0, 0.0)

    def setGoal(self, goal):
        self.goal = goal

    def plan(self):
        pass
        
    def stop(self):
        self.v = 0.0
        self.w = 0.0

    def update(self):
        if self.agent.pose == self.goal:
            return
        else:
            self.agent.setPose(self.goal.x, self.goal.y, self.goal.theta)

class RandomViewpointsPlanner(BasePlanner):
    """
        Random planner generates 10 viewpoints in the map
        and moves the robot directly to them with agent.setPosition.
    """
    
    def __init__(self, agent, env):
        super(RandomViewpointsPlanner, self).__init__(agent)

        self.width = env.width
        self.height = env.height
        self.viewpoints = []
        self.viewpoint = 0
        self.num_waypoints = 10

        self.visualization = plt.scatter([], [], color='black', s=1)

    def plan(self):
        self.viewpoint = 0

        # Generate 10 viewpoints
        self.viewpoints = []
        self.rotations = []
        for i in range(self.num_waypoints):
            x = (np.random.rand() - 0.5) * (self.width)
            y = (np.random.rand() - 0.5)* (self.height)
            theta = np.random.rand() * 2 * np.pi - np.pi
            self.viewpoints.append((x, y))
            self.rotations.append(theta)

        self.updateVisualization()
        self.agent.setPose(x, y, theta)

    def update(self):
        self.viewpoint += 1
        if self.viewpoint < len(self.viewpoints):
            # Set the goal for the low level planner
            x, y = self.viewpoints[self.viewpoint]
            theta = self.rotations[self.viewpoint]
            self.agent.setPose(x, y, theta)
        else:
            # Replan
            self.plan()

    def updateVisualization(self):
        # Add points and splines to the visualization
        self.visualization.set_offsets(self.viewpoints)
