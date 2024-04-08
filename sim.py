import numpy as np
import matplotlib.pyplot as plt
import os
import errno
import datetime
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.stats import chi2

from basics import Pose2D
from objects import Agent2D, Target2D, Occluder2D, Target2DSemantic, PlaneLight2D
from environment import Environment2D, Environment2DRoughness
from dynamics import , RoughnessTerrainDynamicModel
from perception import RangeBearingFoVObservationModel, SemanticFoVObservationModel
from belief import ExtendedKalmanFilter, ExtendedKalmanFilterWithUtility, UtilityEstimationGrid, SemanticBayes, SemanticBayesWithUtility
from planner import RandomPlanner, RandomPlannerTrajectory, LowLevelPlanner,PosesLowLevelPlanner, MonteCarloTreeSearchPlanner, GreedyPlanner
from logger import Logger
from visualization import Visualizer, VariableInspector
from priors import ConstantPrior, DistanceBasedPrior

def getValidRandomPosition(env, margin):
    position = Pose2D(np.random.uniform(-4, 4), np.random.uniform(-4, 4), np.random.uniform(-np.pi, np.pi))
    while not env.isPositionValid(position, margin):
        position = Pose2D(np.random.uniform(-4, 4), np.random.uniform(-4, 4), np.random.uniform(-np.pi, np.pi))
    return position

def getValidBorderPosition(env):
    # Chose one of the four borders randomly
    border = np.random.randint(0, 4)
    while border in env.occupied_borders:
        border = np.random.randint(0, 4)
    if border == 0:
        # Left border
        position = Pose2D(-env.width / 2, np.random.uniform(-env.height / 2, env.height / 2), 0)
    elif border == 1:
        # Right border
        position = Pose2D(env.width / 2, np.random.uniform(-env.height / 2, env.height / 2), -np.pi)
    elif border == 2:
        # Top border
        position = Pose2D(np.random.uniform(-env.width / 2, env.width / 2), env.height / 2, -np.pi / 2)
    elif border == 3:
        # Bottom border
        position = Pose2D(np.random.uniform(-env.width / 2, env.width / 2), -env.height / 2, np.pi / 2)
    
    env.addOccupiedBorder(border)
    return position

def getValidPositionAround(env, position, distance, margin):
    """
        Get a random position around the given position at a given distance
    """
    angle = np.random.uniform(-np.pi, np.pi)
    new_position = Pose2D(position.x + distance * np.cos(angle), position.y + distance * np.sin(angle), 0)
    while not env.isPositionValid(new_position, margin):
        angle = np.random.uniform(-np.pi, np.pi)
        new_position = Pose2D(position.x + distance * np.cos(angle), position.y + distance * np.sin(angle), 0)
    return new_position

"""
    Configuration: the posible baselines are: [metric,semantic]_[basic,utility,oracle]
"""
def simulation(exp_name, planner_name, seed):
    # Global conf
    MANUAL = False
    SAVE_IMGS = False
    SAVE_UTILITIES = False
    VISUALIZE = False
    VISUALIZE_PLANNER = False
    SAVE_RESULTS = True

    # Global variables
    GLOBAL_IDS = 1

    # Get current directory
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'results', exp_name)
    # Get log name wiht the current time and date
    now = datetime.datetime.now()
    log_name = now.strftime("%Y_%m_%d_%H_%M_%S_seed_" + str(seed))
    save_log_file = os.path.join(save_path, log_name + '.csv')

    # Try create the directory if it does not exist
    try:
        print("Creating directory: ", save_path)
        os.makedirs(save_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    if SAVE_IMGS or SAVE_UTILITIES:
        try:
            print("Creating directory: ", os.path.join(save_path, log_name))
            os.makedirs(os.path.join(save_path, log_name))
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            
    # Create the environment
    env = Environment2D(10, 10, 0.1)
    
    if VISUALIZE:
        vis = Visualizer(env.width, env.height, env)
    steps = 0

    occluders = []
    # Add 10 occluders randomly in the environment
    if not "around_occluders" in exp_name:
        for i in range(15):
            position = Pose2D(np.random.uniform(-4, 4), np.random.uniform(-4, 4), 0)
            while not env.isPositionValid(position, 0.1):
                position = Pose2D(np.random.uniform(-4, 4), np.random.uniform(-4, 4), 0)
            occluders.append(Occluder2D(GLOBAL_IDS, position, 0.5))
            GLOBAL_IDS += 1
            env.addOccluder(occluders[i])

    if 'semantic' in exp_name:
        num_classes = 3
        cmap = cm.get_cmap('Spectral')
        colors = [cmap(i / num_classes) for i in range(num_classes)]

    # Add two lights in the environment, near the border of the map, pointing towards the inside and far from each other
    for i in range(2):
        position = getValidBorderPosition(env)
        light = PlaneLight2D(GLOBAL_IDS, position, size=2.0, angle=np.pi / 8, extent=5)
        GLOBAL_IDS += 1
        env.addLight(light)

    # Create the agents
    agent1 = Agent2D(0, Pose2D(-3,-3,0), 0.2)
    agent1.setDynamic(BasicVelocityDynamicModel(agent1, 0.0, 0.0, env.dt))
    env.addAgent(agent1)

    # Add targets randomly in the environment
    targets = []
    # occluders = []
    if 'semantic' in exp_name:
        for i in range(5):

            # Targets of class 1
            position = getValidRandomPosition(env, 0.2)
            targets.append(Target2DSemantic(GLOBAL_IDS, position, 0.1, 1, colors[1]))
            GLOBAL_IDS += 1
            env.addTarget(targets[-1])
            env.addOccluder(targets[-1]) # Also add other targets as occluders

            # Targets of class 2
            position = getValidRandomPosition(env, 0.2)
            targets.append(Target2DSemantic(GLOBAL_IDS, position, 0.1, 2, colors[2]))
            GLOBAL_IDS += 1
            env.addTarget(targets[-1])
            env.addOccluder(targets[-1]) # Also add other targets as occluders

    elif 'metric' in exp_name:    
        for i in range(10):

            position = getValidRandomPosition(env, 0.4)
            targets.append(Target2D(GLOBAL_IDS, position, 0.3))
            GLOBAL_IDS += 1
            env.addTarget(targets[-1])
            env.addOccluder(targets[-1]) # Also add other targets as occluders

    if "around_occluders" in exp_name:
        for target in targets:
            # Add two occluders for each target around its position
            for j in range(1):
                position = getValidPositionAround(env, target.pose, 1.0, 0.2)
                occluders.append(Occluder2D(GLOBAL_IDS, position, 0.5))
                GLOBAL_IDS += 1
                env.addOccluder(occluders[-1])
                # One smaller
                position = getValidPositionAround(env, target.pose, 1.0, 0.2)
                occluders.append(Occluder2D(GLOBAL_IDS, position, 0.2))
                GLOBAL_IDS += 1
                env.addOccluder(occluders[-1])

    sigma_distance = 0.3
    sigma_angle = 0.1
    if 'semantic' in exp_name:
        agent1.setPerception(SemanticFoVObservationModel(agent1, num_classes, np.pi * 60 / 180, 4.0, 1.0))
    elif 'metric' in exp_name:
        agent1.setPerception(RangeBearingFoVObservationModel(agent1, np.pi * 60 / 180, 4.0, 1.0, sigma_distance, sigma_angle))
    else:
        print('Error: unknown baseline')
        exit()

    # We add one belief for each agent
    beliefs = {}
    utilities = {}
    for i in range(len(targets)):
        if 'semantic' in exp_name:
            # Uninformative prior
            prior = np.ones(num_classes)
            prior = prior / np.sum(prior)
            if 'basic' in exp_name:
                target_position = np.array([targets[i].pose.x, targets[i].pose.y])
                belief = SemanticBayes(num_classes, prior, targets[i].id, target_position, agent1.perception, colors)
            elif 'utility' in exp_name:
                target_position = np.array([targets[i].pose.x, targets[i].pose.y])
                perceptual_prior = DistanceBasedPrior(target_position, 5.0, 3)
                utility = UtilityEstimationGrid(env.width, env.height, perceptual_prior, 0.1, True)
                utilities[targets[i].id] = utility
                belief = SemanticBayesWithUtility(num_classes, prior, targets[i].id, target_position, agent1.perception, colors, utility)
            else:
                print('Error: unknown baseline')
                exit()
        elif 'metric' in exp_name:
            # Initial x is close to each target but with random offset
            position = np.array([targets[i].pose.x + np.random.uniform(-0.25, 0.25), targets[i].pose.y + np.random.uniform(-0.25, 0.25)])
            P = np.diag([1.0, 1.0])
            Q = np.diag([0.0, 0.0])
            R = np.diag([sigma_distance**2, sigma_angle**2])
            if 'basic' in exp_name:
                belief = ExtendedKalmanFilter(position, P, targets[i].id, agent1.perception, Q, R, env.dt)
            elif 'utility' in exp_name:
                if 'gt_occ' in exp_name:
                    use_num_factors = False
                else:
                    use_num_factors = True
                perceptual_prior = ConstantPrior(1.0)
                utility = UtilityEstimationGrid(env.width, env.height, perceptual_prior, 0.1, use_num_factors)
                utilities[targets[i].id] = utility
                belief = ExtendedKalmanFilterWithUtility(position, P, targets[i].id, agent1.perception, Q, R, env.dt, utility)
        else:
            print('Error: unknown baseline')
            exit()
        beliefs[targets[i].id] = belief
        env.addBelief(belief)

    if 'utility' in exp_name:
        # Add occluders and lights to all target utilities
        for target in targets:
            if 'metric' in exp_name:
                utilities[target.id].addTarget(beliefs[target.id].position)
                for occluder in occluders:
                    # Add occluder factors only if the target belief is in the occluder range + a margin
                    dist_to_occluder = np.sqrt((occluder.pose.x - beliefs[target.id].position[0])**2 + (occluder.pose.y - beliefs[target.id].position[1])**2)
                    if dist_to_occluder < agent1.perception.max_range + 0.5:
                        utilities[target.id].addOccluder(occluder)
                for other_target in targets:
                    if other_target.id != target.id:
                        # Add occluder factors only if the target belief is in the occluder range + a margin
                        dist_to_occluder = np.sqrt((beliefs[other_target.id].position[0] - beliefs[target.id].position[0])**2 + (beliefs[other_target.id].position[1] - beliefs[target.id].position[1])**2)
                        if dist_to_occluder < agent1.perception.max_range + 0.5 * 2:
                            utilities[target.id].addOccluder(other_target)
                for light in env.lights:
                    # Add light factors only if the target is in the light range
                    is_in_light_area = light.contains(Pose2D(beliefs[target.id].position[0], beliefs[target.id].position[1],0), 1.0)
                    if is_in_light_area:
                        utilities[target.id].addLight(light)
            elif 'semantic' in exp_name:
                target_position = np.array([beliefs[target.id].position[0], beliefs[target.id].position[1]])
                utilities[target.id].addTarget(target_position)
                for occluder in occluders:
                    # Add occluder factors only if the target belief is in the occluder range + a margin
                    dist_to_occluder = np.sqrt((occluder.pose.x - target_position[0])**2 + (occluder.pose.y - target_position[1])**2)
                    if dist_to_occluder < agent1.perception.max_range + 0.5:
                        utilities[target.id].addOccluder(occluder)
                for other_target in targets:
                    if other_target.id != target.id:
                        other_target_position = np.array([beliefs[other_target.id].position[0], beliefs[other_target.id].position[1]])
                        # Add occluder factors only if the target belief is in the occluder range + a margin
                        dist_to_occluder = np.sqrt((other_target_position[0] - target_position[0])**2 + (other_target_position[1] - target_position[1])**2)
                        if dist_to_occluder < agent1.perception.max_range + 0.5 * 2:
                            utilities[target.id].addOccluder(other_target)
                for light in env.lights:
                    # Add light factors only if the target is in the light range
                    is_in_light_area = light.contains(Pose2D(target_position[0], target_position[1],0), 1.0)
                    if is_in_light_area:
                        utilities[target.id].addLight(light)

    low_level_planner1 = PosesLowLevelPlanner(agent1)
    
    if 'greedy' in planner_name:
        planner1 = GreedyPlanner(agent1, low_level_planner1, list(beliefs.values()), 70, env, VISUALIZE_PLANNER)
    elif 'mcts' in planner_name:
        planner1 = MonteCarloTreeSearchPlanner(agent1, low_level_planner1, list(beliefs.values()), 70, env)
    else :
        print('Error: unknown planner')
        exit()

    ### Configure the visualization
    if VISUALIZE:
        for light in env.lights:
            vis.addLight(light)
        for occluder in occluders:
            vis.addOccluder(occluder)
        for belief in beliefs.values():
            vis.addBelief(belief)
        for target in targets:
            vis.addTarget(target) # Add the targets  after the beliefs so that they are drawn on top
        vis.addAgent(agent1)
        
    total_entropy = 0
    if 'metric' in exp_name:
        for belief in beliefs.values():
            total_entropy += 15 # This is the minimum entropy considered
    elif 'semantic' in exp_name:
        for belief in beliefs.values():
            total_entropy += belief.getEntropy()

    ### METRICS
    names = ['entropy']
    if 'semantic' in exp_name:
        names.append('accuracy')
        names.append('accuracy_th')
        names.append('avg_confidence')
    elif 'metric' in exp_name:
        names.append('error_x')
        names.append('error_y')
        names.append('error')
        names.append('rmse')
        names.append('joint_nees')
        names.append('avg_nees')
        names.append('sum_nees')

    names.append('partial_occlusions_hit')
    names.append('total_occlusions_hit')
    names.append('lights_hit')

    logger = Logger(names, save_log_file)

    if VISUALIZE:
        inspector = VariableInspector(names, logger)

    # Gather METRICS
    logger.addData(total_entropy, 'entropy')
    # Get initial errors
    errors = []
    errors_xy = []

    # For joint nees we need to accumulate the errors and uncertainties for all targets
    errors_xy_single = []
    P_single = []
    nees_per_target = []
    avg_conf_per_target = []
    acc_th_per_target = []

    for target in targets:
        belief = beliefs[target.id]
        errors.append(belief.getError(target))
        if 'semantic' in exp_name:
            acc_th_per_target.append(belief.getErrorTH(target))
            avg_conf_per_target.append(belief.getConfidence(target))
        elif 'metric' in exp_name:
            error_xy = belief.getErrorXY(target)
            errors_xy.append(error_xy)
            errors_xy_single.append(error_xy[0])
            errors_xy_single.append(error_xy[1])
            P_single.append(belief.P[0,0])
            P_single.append(belief.P[1,1])
            nees_per_target.append(belief.getNEES(target))

    errors = np.array(errors)
    if 'semantic' in exp_name:
        # Compute the semantic evaluation metrics
        correct = len(targets) - np.count_nonzero(errors)
        accuracy = correct / len(targets)
        logger.addData(accuracy, 'accuracy')
        avg_confidence = np.mean(avg_conf_per_target)
        logger.addData(avg_confidence, 'avg_confidence')
        correct_th = len(targets) - np.count_nonzero(acc_th_per_target)
        accuracy_th = correct_th / len(targets)
        logger.addData(accuracy_th, 'accuracy_th')
    elif 'metric' in exp_name:
        # Compute the metric evaluation metrics
        errors_xy = np.array(errors_xy)
        # Get all the errors in a single array and all the uncertainties in a single diagonal matrix
        errors_xy_single = np.array(errors_xy_single)
        P_single = np.diag(np.array(P_single))
        joint_nees = errors_xy_single.T @ np.linalg.inv(P_single) @ errors_xy_single
        # For the consistency we have to evaluate the chi_square in num_targets * 2 dimensions with alpha = 0.05
        # chi_squared = chi2.ppf(0.95, len(targets) * 2)
        # consistency_ratio = nees / chi_squared
        # Get the number of consistent targets
        avg_nees = np.mean(nees_per_target)

        logger.addData(np.sum(errors), 'error')
        logger.addData(np.sqrt(np.sum(errors**2)), 'rmse')
        logger.addData(np.sum(errors_xy[:,0]), 'error_x')
        logger.addData(np.sum(errors_xy[:,1]), 'error_y')
        logger.addData(joint_nees, 'joint_nees')
        logger.addData(avg_nees, 'avg_nees')
        logger.addData(np.sum(nees_per_target), 'sum_nees')

    logger.addData(0, 'partial_occlusions_hit')
    logger.addData(0, 'total_occlusions_hit')
    logger.addData(0, 'lights_hit')

    if 'greedy' in planner_name:
        planner1.plan()
    elif 'mcts' in planner_name:
        planner1.plan(100)

    if SAVE_UTILITIES:
        # Save utilities
        for key, belief in beliefs.items():
            if 'utility' in exp_name:
                belief.utility.visualize()
                plt.savefig(os.path.join(save_path, log_name, 'utility_%d.png' % key))
                plt.close()

    pbar = tqdm(total=planner1.budget)
    while planner1.budget > 0:
        env.update()
        # Predict for all beliefs
        for belief in beliefs.values():
            belief.predict()
        take_measurement = planner1.update()
        entropy_reduction = 0
        if take_measurement:
            pbar.update(1)
            for agent in env.agents:
                measurements = agent.perception.observe(env)
                measurement_pose = Pose2D(agent1.pose.x, agent1.pose.y, agent1.pose.theta)
                
                for m in measurements:
                    if 'semantic' in exp_name:
                        z = m.probabilities
                        if (z < 0).any():
                            print("WEIRD MEASUREMENT!")
                            continue
                    elif 'metric' in exp_name:
                        z = np.array([m.noisy_distance, m.noisy_bearing])
                        if z[0] < 0:
                            print("WEIRD MEASUREMENT!")
                            continue
                    prev_entropy = beliefs[m.id].getEntropy()
                    beliefs[m.id].update(z, measurement_pose)
                    entropy_reduction += prev_entropy - beliefs[m.id].getEntropy()

                if 'utility' in exp_name:
                   for utility in utilities.values():
                    utility.addMeasurementPose(measurement_pose)

        # Gather new entropy
        total_entropy -= entropy_reduction

        # Gather METRICS
        logger.addData(total_entropy, 'entropy')
        # Get initial errors
        errors = []
        errors_xy = []

        # For joint nees we need to accumulate the errors and uncertainties for all targets
        errors_xy_single = []
        P_single = []
        nees_per_target = []
        avg_conf_per_target = []
        acc_th_per_target = []

        for target in targets:
            belief = beliefs[target.id]
            errors.append(belief.getError(target))
            if 'semantic' in exp_name:
                acc_th_per_target.append(belief.getErrorTH(target))
                avg_conf_per_target.append(belief.getConfidence(target))
            elif 'metric' in exp_name:
                error_xy = belief.getErrorXY(target)
                errors_xy.append(error_xy)
                errors_xy_single.append(error_xy[0])
                errors_xy_single.append(error_xy[1])
                P_single.append(belief.P[0,0])
                P_single.append(belief.P[1,1])
                nees_per_target.append(belief.getNEES(target))

        errors = np.array(errors)
        if 'semantic' in exp_name:
            # Compute the semantic evaluation metrics
            correct = len(targets) - np.count_nonzero(errors)
            accuracy = correct / len(targets)
            logger.addData(accuracy, 'accuracy')
            avg_confidence = np.mean(avg_conf_per_target)
            logger.addData(avg_confidence, 'avg_confidence')
            correct_th = len(targets) - np.count_nonzero(acc_th_per_target)
            accuracy_th = correct_th / len(targets)
            logger.addData(accuracy_th, 'accuracy_th')

        elif 'metric' in exp_name:
            # Compute the metric evaluation metrics
            errors_xy = np.array(errors_xy)
            # Get all the errors in a single array and all the uncertainties in a single diagonal matrix
            errors_xy_single = np.array(errors_xy_single)
            P_single = np.diag(np.array(P_single))
            joint_nees = errors_xy_single.T @ np.linalg.inv(P_single) @ errors_xy_single
            # For the consistency we have to evaluate the chi_square in num_targets * 2 dimensions with alpha = 0.05
            # chi_squared = chi2.ppf(0.95, len(targets) * 2)
            # consistency_ratio = nees / chi_squared
            # Get the number of consistent targets
            avg_nees = np.mean(nees_per_target)

            logger.addData(np.sum(errors), 'error')
            logger.addData(np.sqrt(np.sum(errors**2)), 'rmse')
            logger.addData(np.sum(errors_xy[:,0]), 'error_x')
            logger.addData(np.sum(errors_xy[:,1]), 'error_y')
            logger.addData(joint_nees, 'joint_nees')
            logger.addData(avg_nees, 'avg_nees')
            logger.addData(np.sum(nees_per_target), 'sum_nees')

        logger.addData(agent.perception.partial_occlusions_hit, 'partial_occlusions_hit')
        logger.addData(agent.perception.total_occlusions_hit, 'total_occlusions_hit')
        logger.addData(agent.perception.lights_hit, 'lights_hit')

        if VISUALIZE:
            vis.update()
            inspector.update()
            # Save image
            if SAVE_IMGS:
                vis.saveImage(os.path.join(save_path, log_name, '%04d.png' % steps))
                inspector.saveImage(os.path.join(save_path, log_name, 'inspector.png'))

        if SAVE_RESULTS:
            logger.saveData()
        steps += 1
        # Pause until user presses a key without cv2
        if MANUAL:
            input("Press Enter to continue...")
        else:
            plt.pause(0.01)
    
    pbar.close()
    print('Simulation finished')

if __name__ == '__main__':
    # Perform 10 simulations
    
    # Get a fixed seed for the simulation so that it is repeatable for all baselines
    fix_seed =  np.random.randint(0, 1000) # Code version developed on 645. Check 68 for backlight noise
    # Time experiment
    start = datetime.datetime.now()
    for i in range(10):
        # Set the seed for the simulation
        # print('Simulation %d' % i) # Explore seed: SEED: 170 there are 3 observations with light in utility and 1 in basic, why?
        # np.random.seed(fix_seed + i)
        # print('SEED: %d' % (fix_seed + i))
        # simulation('metric_utility_gt_occ', 'greedy')
        # plt.close('all')

        np.random.seed(fix_seed + i)
        print('SEED: %d' % (fix_seed + i))
        simulation('metric_utility', 'greedy', fix_seed + i)
        plt.close('all')

        np.random.seed(fix_seed + i)
        simulation('metric_basic', 'greedy', fix_seed + i)
        plt.close('all')

    print('All simulations finished')
    end = datetime.datetime.now()
    print('Time: ', end - start)