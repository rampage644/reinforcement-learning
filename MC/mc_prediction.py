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
def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm. Calculates the value function
    for a given policy using sampling.

    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.

    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final value function
    V = defaultdict(float)

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

    for _ in range(num_episodes):
        episodes = generate_episode()

        states = list(map(operator.itemgetter(0), episodes))
        # print(episodes)
        for state in set(states):
            # print(state)
            # find t of first occurence
            t = states.index(state)
            # print(t)
            rewards = list(map(operator.itemgetter(2), episodes[t:]))
            # print(rewards)
            total_reward = sum(map(lambda i: discount_factor ** i[0] * i[1], enumerate(rewards)))
            # print('total', total_reward)

            returns_count[state] += 1
            returns_sum[state] += total_reward
        # print('-'*40)

    for state in returns_sum.keys():
        V[state] = returns_sum[state] / returns_count[state]
    return V

#%%
def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])


#%%
V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

#%%
V_500k = mc_prediction(sample_policy, env, num_episodes=500000)
plotting.plot_value_function(V_500k, title="500,000 Steps")


