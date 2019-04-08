# Copyright 2018 The trfl Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""TensorFlow ops for expressing common types of RL policies."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import tensorflow as tf
import tensorflow_probability as tfp


def epsilon_greedy(action_values, epsilon):
  """Computes an epsilon-greedy distribution over actions.

  This returns a categorical distribution over a discrete action space. It is
  assumed that the trailing dimension of `action_values` is of length A, i.e.
  the number of actions. It is also assumed that actions are 0-indexed.

  This policy does the following:

  - With probability 1 - epsilon, take the action corresponding to the highest
  action value, breaking ties uniformly at random.
  - With probability epsilon, take an action uniformly at random.

  Args:
    action_values: A Tensor of action values with any rank >= 1 and dtype float.
      Shape can be flat ([A]), batched ([B, A]), a batch of sequences
      ([T, B, A]), and so on.
    epsilon: A scalar Tensor (or Python float) with value between 0 and 1.

  Returns:
    policy: tfp.distributions.Categorical distribution representing the policy.
  """
  with tf.name_scope("epsilon_greedy", values=[action_values, epsilon]):

    # Convert inputs to Tensors if they aren't already.
    action_values = tf.convert_to_tensor(action_values)
    epsilon = tf.convert_to_tensor(epsilon, dtype=action_values.dtype)

    # We compute the action space dynamically.
    num_actions = tf.cast(tf.shape(action_values)[-1], action_values.dtype)

    # Dithering action distribution.
    dither_probs = 1 / num_actions * tf.ones_like(action_values)

    # Greedy action distribution, breaking ties uniformly at random.
    max_value = tf.reduce_max(action_values, axis=-1, keepdims=True)
    greedy_probs = tf.cast(tf.equal(action_values, max_value),
                           action_values.dtype)
    greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

    # Epsilon-greedy action distribution.
    probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

    # Make the policy object.
    policy = tfp.distributions.Categorical(probs=probs)

  return policy
