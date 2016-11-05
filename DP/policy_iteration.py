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
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    diff = np.ones_like(V)
    while True:
        # for all states in V(s)
        V_ = np.zeros_like(V)
        for s, v in enumerate(V):
          # for all actions avaliable
          for a in range(env.nA):
            for p, sprime, r, done in env.P[s][a]:
              V_[s] += policy[s][a] * p * (r + discount_factor * V[sprime])
          diff[s] = max([theta, abs(V_[s] - V[s])])
        V = V_

        if np.all(diff <= theta):
          break
    return np.array(V)

#%%
def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = None

    while True:
        Vs = policy_eval_fn(policy, env, discount_factor)
        stable = True
        for s, v in enumerate(Vs):
            old_action = policy[s]
            q = [p * (r + discount_factor * Vs[sprime]) for a in range(env.nA) for p, sprime, r, _ in env.P[s][a]]
            policy[s] = np.zeros_like(policy[s])
            policy[s][np.argmax(q)] = 1
            if np.all(old_action != policy[s]):
                stable = False
        if stable:
            V = policy_eval_fn(policy, env, discount_factor)
            break

    return policy, V

#%%
policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
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
