#%%

%matplotlib inline

import gym
import itertools
import matplotlib
import numpy as np
import operator
import sys
import tensorflow as tf
import collections

if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

matplotlib.style.use('ggplot')

#%%
env = CliffWalkingEnv()

#%%
class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, learning_rate=0.01, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], name='state')
            self.action = tf.placeholder(tf.int32, name='action')
            self.target = tf.placeholder(tf.float32, name='target')

            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=int(env.action_space.n),
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            )

            self.action_probs = tf.squeeze(tf.nn.softmax(self.output_layer))
            self.selected_action = tf.gather(self.action_probs, self.action)

            self.loss = -tf.log(self.selected_action) * self.target
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run([self.action_probs], feed_dict={self.state: state})

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.state: state,
            self.action: action,
            self.target: target
        })
        return loss


#%%
class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.state = tf.placeholder(tf.int32, [], name='state')
            self.target = tf.placeholder(tf.float32, name='target')

            state_one_hot = tf.one_hot(self.state, int(env.observation_space.n))
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(state_one_hot, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer
            )

            self.value = tf.squeeze(self.output_layer)

            self.loss = tf.squared_difference(self.value, self.target)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        value = sess.run(self.value, feed_dict={
            self.state: state
        })
        return value

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        loss, _ = sess.run([self.loss, self.train_op], feed_dict={
            self.state: state,
            self.target: target
        })
        return loss


#%%
def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
    """
    REINFORCE (Monte Carlo Policy Gradient) Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a baseline
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    for i_episode in range(num_episodes):
        # Reset the environment and pick the fisrst action
        state = env.reset()

        # One step in the environment
        I = 1
        for t in itertools.count():

            # Take a step
            action_probs = np.squeeze(estimator_policy.predict(state))
            action = np.random.choice(int(env.action_space.n), p=action_probs)
            n_state, reward, done, _ = env.step(action)

            # Update statistics
            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            # calculate delta
            td_target = reward + discount_factor * estimator_value.predict(n_state)
            td_error = td_target - estimator_value.predict(state)

            # update critic
            estimator_value.update(state, td_target)

            # update actor
            estimator_policy.update(state, td_error, action)


            # Print out which step we're on, useful for debugging.
            print("\rStep {} @ Episode {}/{} ({})".format(
                    t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")
            # sys.stdout.flush()

            if done:
                break

            state = n_state
            I *= discount_factor


    return stats

#%%
tf.reset_default_graph()

global_step = tf.Variable(0, name="global_step", trainable=False)
policy_estimator = PolicyEstimator()
value_estimator = ValueEstimator()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need to learn a good
    # policy may vary. ~2000-5000 seemed to work well for me.
    stats = actor_critic(env, policy_estimator, value_estimator, 5000, discount_factor=1.0)

#%%
plotting.plot_episode_stats(stats, smoothing_window=25)


