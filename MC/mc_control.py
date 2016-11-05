#%%
%matplotlib inline

import gym
import matplotlib
import numpy as np
import sys
import operator

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

#%%
env = BlackjackEnv()

#%%
def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    def policy_fn(state):
        probs = np.ones_like(Q[state]) * epsilon / nA
        probs[np.argmax(Q[state])] += 1 - epsilon

        return probs

    return policy_fn

#%%
def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    def generate_episode():
        T = 100
        steps = []
        state = env.reset()
        for t in range(T):
            action = np.random.choice(np.arange(env.nA), p=policy(state))
            n_state, reward, done, _ = env.step(action)
            steps.append((state, action, reward))
            if done:
                break
            state = n_state
        return steps

    # Implement this!


    for _ in range(num_episodes):
        episodes = generate_episode()

        states_actions = list(map(operator.itemgetter(0, 1), episodes))
        # print(episodes)
        for state, action in set(states_actions):
            # print(state)
            # find t of first occurence
            t = states_actions.index((state, action))
            # print(t)
            rewards = list(map(operator.itemgetter(2), episodes[t:]))
            # print(rewards)
            total_reward = sum(map(lambda i: discount_factor ** i[0] * i[1], enumerate(rewards)))
            # print('total', total_reward)

            returns_count[(state, action)] += 1
            returns_sum[(state, action)] += total_reward

        for state, action in returns_sum.keys():
            Q[state][action] = returns_sum[(state, action)] / returns_count[(state, action)]

        policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)



    return Q, policy

#%%
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)

#%%
# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")
