import numpy as np
import matplotlib.pyplot as plt
import random, os
from tqdm import tqdm
from roomba_class import Roomba


# ### Setup Environment

def seed_everything(seed: int):
    """Seed everything for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def is_obstacle(position):
    """Check if the position is outside the grid boundaries (acting as obstacles)."""
    x, y = position
    return x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT

def setup_environment(seed=111):
    """Setup function for grid and direction definitions."""
    global GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS
    GRID_WIDTH = 10
    GRID_HEIGHT = 10
    HEADINGS = ['N', 'E', 'S', 'W']
    MOVEMENTS = {
        'N': (0, -1),
        'E': (1, 0),
        'S': (0, 1),
        'W': (-1, 0),
    }
    print("Environment setup complete with a grid of size {}x{}.".format(GRID_WIDTH, GRID_HEIGHT))
    seed_everything(seed)
    return GRID_WIDTH, GRID_HEIGHT, HEADINGS, MOVEMENTS


# ### Sensor Movements

def simulate_roomba(T, movement_policy,sigma):
    """
    Simulate the movement of a Roomba robot for T time steps and generate noisy observations.

    Parameters:
    - T (int): The number of time steps for which to simulate the Roomba's movement.
    - movement_policy (str): The movement policy dictating how the Roomba moves.
                             Options may include 'straight_until_obstacle' or 'random_walk'.
    - sigma (float): The standard deviation of the Gaussian noise added to the true position 
                     to generate noisy observations.

    Returns:
    - tuple: A tuple containing three elements:
        1. true_positions (list of tuples): A list of the true positions of the Roomba 
                                            at each time step as (x, y) coordinates.
        2. headings (list): A list of headings of the Roomba at each time step.
        3. observations (list of tuples): A list of observed positions with added Gaussian noise,
                                          each as (obs_x, obs_y).
    """
    # Start at the center
    start_pos = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
    start_heading = random.choice(HEADINGS)
    roomba = Roomba(MOVEMENTS, HEADINGS,is_obstacle,start_pos, start_heading, movement_policy)

    true_positions = []
    observations = []
    headings = []

    print(f"Simulating Roomba movement for policy: {movement_policy}")
    for _ in tqdm(range(T), desc="Simulating Movement"):
        position = roomba.move()
        heading = roomba.heading
        true_positions.append(position)
        headings.append(heading)

        # Generate noisy observation
        noise = np.random.normal(0, sigma, 2)
        observed_position = (position[0] + noise[0], position[1] + noise[1])
        observations.append(observed_position)

    return true_positions, headings, observations


# ### Implement Functions

def emission_probability(state, observation,sigma):
    """
    Calculate the emission probability in log form for a given state and observation using a Gaussian distribution.

    Parameters:
    - state (tuple): The current state represented as (position, heading), 
                     where position is a tuple of (x, y) coordinates.
    - observation (tuple): The observed position as a tuple (obs_x, obs_y).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - float: The log probability of observing the given observation from the specified state.
    """
    ###### YOUR CODE HERE ######
    x_cur, y_cur = state[0][0], state[0][1]
    x_obs, y_obs = observation[0], observation[1]
    diff = (x_obs - x_cur)**2 + (y_obs - y_cur)**2
    log_prob = -0.5 * (diff / sigma**2 + 2*np.log(2 * np.pi * sigma**2))
    # log_prob = -0.5 * (diff / sigma**2 + np.log(2 * np.pi * sigma**2))
    return log_prob
    pass

def transition_probability(prev_state, curr_state, movement_policy):
    """
    Calculate the transition probability in log form between two states based on a given movement policy.

    Parameters:
    - prev_state (tuple): The previous state represented as (position, heading),
                          where position is a tuple of (x, y) coordinates and heading is a direction.
    - curr_state (tuple): The current state represented as (position, heading),
                          similar to prev_state.
    - movement_policy (str): The movement policy that dictates how transitions are made. 
                             Options are 'straight_until_obstacle' and 'random_walk'.

    Returns:
    - float: The log probability of transitioning from prev_state to curr_state given the movement policy.
             Returns 0.0 (log(1)) for certain transitions, -inf (log(0)) for impossible transitions,
             and a uniform log probability for equal transitions in the case of random walk.
    """
    ###### YOUR CODE HERE ######
    x_prev, y_prev, heading_prev = prev_state[0][0], prev_state[0][1], prev_state[1]
    x_curr, y_curr, heading_curr = curr_state[0][0], curr_state[0][1], curr_state[1]
    x_h , y_h = MOVEMENTS[heading_prev][0], MOVEMENTS[heading_prev][1]
    x_exp, y_exp = x_prev + x_h, y_prev + y_h

    if(is_obstacle((x_curr, y_curr))):
        return -np.inf

    if movement_policy == 'random_walk':    
        if not is_obstacle((x_exp, y_exp)) and (x_curr, y_curr) == (x_exp, y_exp):
            return np.log(1)    
        if is_obstacle((x_exp, y_exp)):
            valid_steps = []
            for h in HEADINGS:
                x_hp, y_hp = MOVEMENTS[h][0], MOVEMENTS[h][1]
                x_exp_h, y_exp_h = x_prev + x_hp, y_prev + y_hp
                if not is_obstacle((x_exp_h, y_exp_h)):
                    valid_steps.append((x_exp_h, y_exp_h, h))
            if(len(valid_steps) == 0):
                return -np.inf
            return np.log(1/len(valid_steps))
        else:
            return -np.inf
        

    
    elif movement_policy == 'straight_until_obstacle':
        if not is_obstacle((x_exp, y_exp)) and (x_curr, y_curr) == (x_exp, y_exp):
            return np.log(1)
        elif is_obstacle((x_exp, y_exp)):
        # # elif (x_curr,y_curr) == (x_exp, y_exp) and is_obstacle((x_curr, y_curr)):
        #     valid_steps = []
        #     for h in HEADINGS:
        #         x_hp, y_hp = MOVEMENTS[h][0], MOVEMENTS[h][1]
        #         x_exp_h, y_exp_h = x_prev + x_hp, y_prev + y_hp
        #         if not is_obstacle((x_exp_h, y_exp_h)):
        #             valid_steps.append((x_exp_h, y_exp_h, h))
        #     if(len(valid_steps) == 0):
        #         return -np.inf
            # return np.log(1/len(valid_steps))
            return -np.inf
        elif (x_curr, y_curr) == (x_prev, y_prev) and not is_obstacle((x_exp, y_exp)):
            return np.log(1)
        else:
            return -np.inf
    pass

# ### Viterbi Algorithm
def viterbi(observations, start_state, movement_policy,states,sigma):
    """
    Perform the Viterbi algorithm to find the most likely sequence of states given a series of observations.

    Parameters:
    - observations (list of tuples): A list of observed positions, each as a tuple (obs_x, obs_y).
    - start_state (tuple): The initial state represented as (position, heading),
                           where position is a tuple of (x, y) coordinates.
    - movement_policy (str): The movement policy that dictates how transitions are made.
                             Options are 'straight_until_obstacle' and 'random_walk'.
    - states (list of tuples): A list of all possible states, each represented as (position, heading).
    - sigma (float): The standard deviation of the Gaussian distribution representing observation noise.

    Returns:
    - list of tuples: The most probable sequence of states that could have led to the given observations.
    """
    ###### YOUR CODE HERE ######
    n_obs = len(observations)
    n_states = len(states)

    dp = []
    backpointer = []
    for i in range(n_states):
        dp.append([0]*n_obs)
        backpointer.append([0]*n_obs)

    for i in range(n_obs):
        for j in range(n_states):
            if(i == 0):
                # viterbi[j][0] = emission_probability(start_state, observations[i], sigma)
                dp[j][0] = emission_probability(start_state, observations[i], sigma)
                backpointer[j][0] = -1
            
            else:
                dp[j][i] = -np.inf
                for k in range(n_states):
                    prob_transition = transition_probability(states[k], states[j], movement_policy)
                    prob_emission = emission_probability(states[j], observations[i], sigma)
                    prob = dp[k][i-1] + prob_transition + prob_emission

                    if prob > dp[j][i]:
                        dp[j][i] = prob
                        backpointer[j][i] = k
    
    best_prob = -np.inf
    best_state = 0
    for i in range(n_states):
        if dp[i][n_obs-1] > best_prob:
            best_prob = dp[i][n_obs-1]
            best_state = i
    
    best_path = [best_state]

    for i in range(n_obs-1, 0, -1):
        best_state = backpointer[best_state][i]
        best_path.append(best_state)

    best_path = best_path[::-1]
    
    # output = [start_state]
    output = []
    for i in best_path:
        output.append(states[i])
    return output

    pass


# ### Evaluation (DO NOT CHANGE THIS)
def getestimatedPath(policy, results, states, sigma):
    """
    Estimate the path of the Roomba using the Viterbi algorithm for a specified policy.

    Parameters:
    - policy (str): The movement policy used during simulation, such as 'random_walk' or 'straight_until_obstacle'.
    - results (dict): A dictionary containing simulation results for different policies. Each policy's data includes:
                      - 'true_positions': List of true positions of the Roomba at each time step.
                      - 'headings': List of headings of the Roomba at each time step.
                      - 'observations': List of noisy observations at each time step.
    - states (list of tuples): A list of all possible states (position, heading) for the Hidden Markov Model.
    - sigma (float): The standard deviation of the Gaussian noise used in the emission probability.

    Returns:
    - tuple: A tuple containing:
        1. true_positions (list of tuples): The list of true positions from the simulation.
        2. estimated_path (list of tuples): The most likely sequence of states estimated by the Viterbi algorithm.
    """
    print(f"\nProcessing policy: {policy}")
    data = results[policy]
    observations = data['observations']
    start_state = (data['true_positions'][0], data['headings'][0])
    estimated_path = viterbi(observations, start_state, policy, states, sigma)
    return data['true_positions'], estimated_path


def evaluate_viterbi(estimated_path, true_positions, T,policy):
    """
    Evaluate the accuracy of the Viterbi algorithm's estimated path compared to the true path.
    """
    correct = 0
    for true_pos, est_state in zip(true_positions, estimated_path):
        if true_pos == est_state[0]:
            correct += 1
    accuracy = correct / T * 100
    # data['accuracy'] = accuracy
    # print('estimated_path:',estimated_path)
    # print('true_positions:',true_positions)

    print(f"Tracking accuracy for {policy.replace('_', ' ')} policy: {accuracy:.2f}%")


def plot_results(true_positions, observations, estimated_path, policy):
    """
    Plot the true and estimated paths of the Roomba along with the noisy observations.
    The function plots and saves the graphs of the true and estimated paths.
    """
    # Extract coordinates
    true_x = [pos[0] for pos in true_positions]
    true_y = [pos[1] for pos in true_positions]
    obs_x = [obs[0] for obs in observations]
    obs_y = [obs[1] for obs in observations]
    est_x = [state[0][0] for state in estimated_path]
    est_y = [state[0][1] for state in estimated_path]

    # Identify start and end positions
    start_true = true_positions[0]
    end_true = true_positions[-1]
    start_est = estimated_path[0][0]
    end_est = estimated_path[-1][0]

    # Plotting
    plt.figure(figsize=(10, 10))

    # True Path Plot
    plt.subplot(2, 1, 1)
    plt.plot(true_x, true_y, 'g-', label='True Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the true path
    plt.scatter(*start_true, c='b', marker='o', s=100, label='True Start', edgecolors='black')
    plt.scatter(*end_true, c='purple', marker='X', s=100, label='True End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - True Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)

    # Estimated Path Plot
    plt.subplot(2, 1, 2)
    plt.plot(est_x, est_y, 'b--', label='Estimated Path', linewidth=2)
    plt.scatter(obs_x, obs_y, c='r', s=10, label='Observations')

    # Mark start and end positions on the estimated path
    plt.scatter(*start_est, c='b', marker='o', s=100, label='Estimated Start', edgecolors='black')
    plt.scatter(*end_est, c='purple', marker='X', s=100, label='Estimated End', edgecolors='black')

    plt.title(f'Roomba Path Tracking ({policy.replace("_", " ").title()} Policy) - Estimated Path')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid(True)
    
    fname = f"{policy.replace('_', ' ')}_Policy_Roomba_Path_Tracking.png"
    plt.savefig(fname)


# import csv 
# import pandas as pd
# import os

if __name__ == "__main__":
    # 1. Set up the environment, including grid size, headings, and movements.
    # seed  = 111
    # seed = 150
    seed = 125
    setup_environment(seed)
    sigma = 1.0  # Observation noise standard deviation
    T = 50       # Number of time steps

    # Simulate for both movement policies
    policies = ['random_walk', 'straight_until_obstacle']
    results = {}

    # 2. Loop through each movement policy and simulate the Roomba's movement:
    #    - Generate true positions, headings, and noisy observations.
    #    - Store the results in the dictionary.
    for policy in policies:
        true_positions, headings, observations = simulate_roomba(T, policy,sigma)
        results[policy] = {
            'true_positions': true_positions,
            'headings': headings,
            'observations': observations
        }

    # 3. Define the HMM components
    #   - A list (states) containing all possible states of the Roomba, where each state is represented as a tuple ((x, y), h)
    #   - x, y: The position on the grid.
    #   - h: The heading or direction (e.g., 'N', 'E', 'S', 'W').
    states = []
    ###### YOUR CODE HERE ######
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            for heading in HEADINGS:
                states.append(((i, j), heading))
    # print(states)
    # 4. Loop through each policy to estimate the Roomba's path using the Viterbi algorithm:
    #    - Retrieve the true positions and estimated path.
    #    - Evaluate the accuracy of the Viterbi algorithm.
    #    - Plot the true and estimated paths along with the observations.

    # #code to make csv
    # df = {
    #     'seed value': [seed,seed],
    #     'policy name': [],
    #     'estimated_path': [],
    # }

    #---------------

    for policy in policies:
        true_positions, estimated_path = getestimatedPath(policy,results,states,sigma)
        evaluate_viterbi(estimated_path, true_positions, T,policy)
        plot_results(true_positions, observations, estimated_path, policy)


    # #code to make csv
    #     df['policy name'].append(policy)
    #     df['estimated_path'].append(estimated_path)
    # df_new = pd.DataFrame(df)
    # file_exists = os.path.exists('estimated_paths.csv')
    # df_new.to_csv('estimated_paths.csv',index=False, mode='a', header=not file_exists)
