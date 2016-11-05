#%%

import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.gridworld import GridworldEnv

#%%
pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

#%%
def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """


    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    while True:
        diff = np.ones_like(V)
        for s, v in enumerate(V):
            V[s] = np.max([p * (r + discount_factor * V[sp]) for a in range(env.nA) for p, sp, r, _ in env.P[s][a]])
            diff[s] = max(theta, abs(v - V[s]))
        if np.all(diff <= theta):
            break

    policy = [np.argmax([p * (r + discount_factor * V[sp])
                         for a in range(env.nA)
                         for p, sp, r, _ in env.P[s][a]])
              for s, v in enumerate(V)]
    return np.array(policy), V

#%%
policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(policy, env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

#%%
# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)