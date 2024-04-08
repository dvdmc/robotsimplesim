import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
from copy import copy

from basics import Pose
plt.ion()
class Visualizer(object):

    def __init__(self, height, width, env):
        self.height = height
        self.width = width
        self.env = env

        self.fig, self.ax = plt.subplots()
        # set axis limits and labels
        self.ax.set_xlim([-self.width/2, self.width/2])
        self.ax.set_ylim([-self.height/2, self.height/2])
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        # set axis aspect ratio
        self.ax.set_aspect('equal')
        self.plotted_measurements = []
        self.annotated_measurements = []
        self.semantic_measurements = []

    def addAgent(self, agent):
        if agent.perception is not None: # Add the perception patch first so it is behind the agent
            self.ax.add_patch(agent.perception.visualization)
        self.ax.add_patch(agent.visualization)

    def addTarget(self, target):
        self.ax.add_patch(target.visualization)

    def addOccluder(self, occluder):
        self.ax.add_patch(occluder.visualization)

    def addBelief(self, belief):
        if isinstance(belief.visualization, matplotlib.patches.Patch):
            self.ax.add_patch(belief.visualization)
        elif isinstance(belief.visualization, list):
            for patch in belief.visualization:
                if isinstance(patch, matplotlib.patches.Rectangle):
                    self.ax.add_patch(patch)
                elif isinstance(patch, lines.Line2D):
                    self.ax.add_line(patch)

    def addPlanner(self, planner):
        if isinstance(planner.visualization, lines.Line2D):
            self.ax.add_line(planner.visualization)
        else:
            self.ax.add_collection(planner.visualization)
        planner.is_visualized = True # This is required because this method can use scatter

    def addUtility(self, utility):
        # self.ax.add_patch(utility.visualization)
        utility.is_visualized = True # This is required because this method can use imshow

    def addLight(self, light):
        self.ax.add_patch(light.visualization)
        # Add the body line as a reference
        self.ax.add_line(lines.Line2D([light.v_A[0], light.v_C[0]], [light.v_A[1], light.v_C[1]], color='k', linewidth=1.0, linestyle='--'))

    def plotMeasurement(self, position: Pose, label: str, color: str):
        # Plot a measurement
        self.plotted_measurements.append(self.ax.scatter(position.translation[0], position.translation[1], color=color))
        self.annotated_measurements.append(self.ax.annotate(label, xy=(position.translation[0], position.translation[1]), xytext=(7,7), textcoords='offset pixels'))
        return len(self.plotted_measurements) - 1
    
    def plotSemantic(self, position, label, color):
        # Plot a measurement
        self.semantic_measurements.append(self.ax.text(position.translation[0], position.translation[1], label, fontsize=6, bbox=dict(boxstyle='square, pad=-0.01', fc=color, ec='none')))
        return len(self.semantic_measurements) - 1
    
    def removeMeasurement(self, index):
        # Delete a plotted measurement
        self.plotted_measurements[index].remove()
        self.annotated_measurements[index].remove()

    def removeSemantic(self, index):
        # Delete a plotted semantic measurement 
        self.semantic_measurements[index].remove()
        
    def update(self):
        # Check the position of the agents and targets and update the scale of the plot
        # positions are updated in the agents and targets. Here we only update the scale

        # Check max and min of x and y positions
        x_min = np.inf
        x_max = -np.inf
        y_min = np.inf
        y_max = -np.inf
        for agent in self.env.agents:
            x_min = min(x_min, agent.pose.translation[0])
            x_max = max(x_max, agent.pose.translation[0])
            y_min = min(y_min, agent.pose.translation[1])
            y_max = max(y_max, agent.pose.translation[1])
        for target in self.env.targets:
            x_min = min(x_min, target.pose.translation[0])
            x_max = max(x_max, target.pose.translation[0])
            y_min = min(y_min, target.pose.translation[1])
            y_max = max(y_max, target.pose.translation[1])

        # Compare with self.height and self.width
        x_min = min(x_min, -self.height/2)
        x_max = max(x_max, self.height/2)
        y_min = min(y_min, -self.width/2)
        y_max = max(y_max, self.width/2)

        # Set the limits of the plot
        self.ax.set_xlim([x_min - 1, x_max + 1])
        self.ax.set_ylim([y_min - 1, y_max + 1])

    def saveImage(self, filename):
        self.fig.savefig(filename)

class VariableInspector(object):
    """
        This class receives some variables with names to plot.
        It generates a subplot for each variable and plots it.
    """

    def __init__(self, names, logger):
        self.names = names
        # If names are more than 3, create two columns
        if len(names) > 3:
            self.fig, self.axes = plt.subplots(len(names), 1)
        else:
            self.fig, self.axes = plt.subplots(len(names)/2+1, 2)
        self.logger = logger
        self.lines = [lines.Line2D([],[]) for _ in range(len(names))]

        # Check that all data is in the logger. Otherwise output an error
        for name in names:
            if name not in self.logger.names:
                print("ERROR: Variable {} not in logger".format(name))
                exit(1)

        if len(names) == 1:
            self.axes = [self.axes]

        for i, name in enumerate(names):
            self.axes[i].set_title(name)
            self.axes[i].add_line(self.lines[i])

    def update(self):
        # Update the plot with the data in the logger
        for name in self.names:
            data = copy(self.logger.data[self.logger.names.index(name)])
            x_data = copy(self.logger.x_data[self.logger.names.index(name)])
            self.lines[self.names.index(name)].set_data(x_data, data)
            # Adjust the limits of the plot
            self.axes[self.names.index(name)].set_xlim([0, len(data)])
            self.axes[self.names.index(name)].set_ylim([min(data), max(data) + 0.5])

    def saveImage(self, filename):
        self.fig.savefig(filename)