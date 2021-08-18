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

import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def epsilon_greedy(action_values, epsilon, legal_actions_mask=None):
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
    legal_actions_mask: An optional one-hot tensor having the shame shape and
      dtypes as `action_values`, defining the legal actions:
      legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
      If not provided, all actions will be considered legal and
      `tf.ones_like(action_values)`.

  Returns:
    policy: tfp.distributions.Categorical distribution representing the policy.
  """
  with tf.name_scope("epsilon_greedy",
                     values=[action_values, epsilon, legal_actions_mask]):

    # Convert inputs to Tensors if they aren't already.
    action_values = tf.convert_to_tensor(action_values)
    epsilon = tf.convert_to_tensor(epsilon, dtype=action_values.dtype)

    # We compute the action space dynamically.
    num_actions = tf.cast(tf.shape(action_values)[-1], action_values.dtype)

    if legal_actions_mask is None:
      # Dithering action distribution.
      dither_probs = 1 / num_actions * tf.ones_like(action_values)
      # Greedy action distribution, breaking ties uniformly at random.
      max_value = tf.reduce_max(action_values, axis=-1, keepdims=True)
      greedy_probs = tf.cast(tf.equal(action_values, max_value),
                             action_values.dtype)
    else:
      legal_actions_mask = tf.convert_to_tensor(legal_actions_mask)
      # Dithering action distribution.
      dither_probs = 1 / tf.reduce_sum(
          legal_actions_mask, axis=-1, keepdims=True) * legal_actions_mask
      masked_action_values = tf.where(tf.equal(legal_actions_mask, 1),
                                      action_values,
                                      tf.fill(tf.shape(action_values), -np.inf))
      # Greedy action distribution, breaking ties uniformly at random.
      max_value = tf.reduce_max(masked_action_values, axis=-1, keepdims=True)
      greedy_probs = tf.cast(
          tf.equal(action_values * legal_actions_mask, max_value),
          action_values.dtype)

    greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

    # Epsilon-greedy action distribution.
    probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

    # Make the policy object.
    policy = tfp.distributions.Categorical(probs=probs)

  return policy
