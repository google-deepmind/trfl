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
"""Tensorflow ops for common Distributional RL learning rules.

Distributions are taken to be categorical over a support of 'N' distinct atoms,
which are always specified in ascending order.

These ops define state/action value distribution learning rules for discrete,
scalar, action spaces. Actions must be represented as indices in the range
`[0, K)` where `K` is the number of distinct actions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf
from trfl import base_ops
from trfl import distribution_ops

Extra = collections.namedtuple("dist_value_extra", ["target"])

_l2_project = distribution_ops.l2_project


def _slice_with_actions(embeddings, actions):
  """Slice a Tensor.

  Take embeddings of the form [batch_size, num_actions, embed_dim]
  and actions of the form [batch_size, 1], and return the sliced embeddings
  like embeddings[:, actions, :].

  Args:
    embeddings: Tensor of embeddings to index.
    actions: int Tensor to use as index into embeddings

  Returns:
    Tensor of embeddings indexed by actions
  """
  shape = tuple(t.value for t in embeddings.get_shape())
  batch_size, num_actions = shape[0], shape[1]

  # Values are the 'values' in a sparse tensor we will be setting
  act_indx = tf.cast(actions, tf.int64)[:, None]
  values = tf.reshape(tf.cast(tf.ones(tf.shape(actions)), tf.bool), [-1])

  # Create a range for each index into the batch
  act_range = tf.range(0, batch_size, dtype=tf.int64)[:, None]
  # Combine this into coordinates with the action indices
  indices = tf.concat([act_range, act_indx], 1)

  actions_mask = tf.SparseTensor(indices, values, [batch_size, num_actions])
  actions_mask = tf.stop_gradient(
      tf.sparse_tensor_to_dense(actions_mask, default_value=False))
  sliced_emb = tf.boolean_mask(embeddings, actions_mask)
  return sliced_emb


def categorical_dist_qlearning(atoms_tm1,
                               logits_q_tm1,
                               a_tm1,
                               r_t,
                               pcont_t,
                               atoms_t,
                               logits_q_t,
                               name="CategoricalDistQLearning"):
  """Implements Distributional Q-learning as TensorFlow ops.

  The function assumes categorical value distributions parameterized by logits.

  See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
  Dabney and Munos. (https://arxiv.org/abs/1707.06887).

  Args:
    atoms_tm1: 1-D tensor containing atom values for first timestep,
      shape `[num_atoms]`.
    logits_q_tm1: Tensor holding logits for first timestep in a batch of
      transitions, shape `[B, num_actions, num_atoms]`.
    a_tm1: Tensor holding action indices, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    atoms_t: 1-D tensor containing atom values for second timestep,
      shape `[num_atoms]`.
    logits_q_t: Tensor holding logits for second timestep in a batch of
      transitions, shape `[B, num_actions, num_atoms]`.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`: a tensor containing the values that `q_tm1` at actions
        `a_tm1` are regressed towards, shape `[B, num_atoms]`.

  Raises:
    ValueError: If the tensors do not have the correct rank or compatibility.
  """
  # Rank and compatibility checks.
  assertion_lists = [[logits_q_tm1, logits_q_t], [a_tm1, r_t, pcont_t],
                     [atoms_tm1, atoms_t]]
  base_ops.wrap_rank_shape_assert(assertion_lists, [3, 1, 1], name)

  # Categorical distributional Q-learning op.
  with tf.name_scope(
      name,
      values=[
          atoms_tm1, logits_q_tm1, a_tm1, r_t, pcont_t, atoms_t, logits_q_t
      ]):

    with tf.name_scope("target"):
      # Scale and shift time-t distribution atoms by discount and reward.
      target_z = r_t[:, None] + pcont_t[:, None] * atoms_t[None, :]

      # Convert logits to distribution, then find greedy action in state s_t.
      q_t_probs = tf.nn.softmax(logits_q_t)
      q_t_mean = tf.reduce_sum(q_t_probs * atoms_t, 2)
      pi_t = tf.argmax(q_t_mean, 1, output_type=tf.int32)

      # Compute distribution for greedy action.
      p_target_z = _slice_with_actions(q_t_probs, pi_t)

      # Project using the Cramer distance
      target = tf.stop_gradient(_l2_project(target_z, p_target_z, atoms_tm1))

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logit_qa_tm1, labels=target)

    return base_ops.LossOutput(loss, Extra(target))


def categorical_dist_double_qlearning(atoms_tm1,
                                      logits_q_tm1,
                                      a_tm1,
                                      r_t,
                                      pcont_t,
                                      atoms_t,
                                      logits_q_t,
                                      q_t_selector,
                                      name="CategoricalDistDoubleQLearning"):
  """Implements Distributional Double Q-learning as TensorFlow ops.

  The function assumes categorical value distributions parameterized by logits,
  and combines distributional RL with double Q-learning.

  See "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
  Hessel, Modayil, van Hasselt, Schaul et al.
  (https://arxiv.org/abs/1710.02298).

  Args:
    atoms_tm1: 1-D tensor containing atom values for first timestep,
      shape `[num_atoms]`.
    logits_q_tm1: Tensor holding logits for first timestep in a batch of
      transitions, shape `[B, num_actions, num_atoms]`.
    a_tm1: Tensor holding action indices, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    atoms_t: 1-D tensor containing atom values for second timestep,
      shape `[num_atoms]`.
    logits_q_t: Tensor holding logits for second timestep in a batch of
      transitions, shape `[B, num_actions, num_atoms]`.
    q_t_selector: Tensor holding another set of Q-values for second timestep
      in a batch of transitions, shape `[B, num_actions]`.
      These values are used for estimating the best action. In Double DQN they
      come from the online network.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: Tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`:  Tensor containing the values that `q_tm1` at actions
        `a_tm1` are regressed towards, shape `[B, num_atoms]` .

  Raises:
    ValueError: If the tensors do not have the correct rank or compatibility.
  """
  # Rank and compatibility checks.
  assertion_lists = [[logits_q_tm1, logits_q_t], [a_tm1, r_t, pcont_t],
                     [atoms_tm1, atoms_t], [q_t_selector]]
  base_ops.wrap_rank_shape_assert(assertion_lists, [3, 1, 1, 2], name)

  # Categorical distributional double Q-learning op.
  with tf.name_scope(
      name,
      values=[
          atoms_tm1, logits_q_tm1, a_tm1, r_t, pcont_t, atoms_t, logits_q_t,
          q_t_selector
      ]):

    with tf.name_scope("target"):
      # Scale and shift time-t distribution atoms by discount and reward.
      target_z = r_t[:, None] + pcont_t[:, None] * atoms_t[None, :]

      # Convert logits to distribution, then find greedy policy action in
      # state s_t.
      q_t_probs = tf.nn.softmax(logits_q_t)
      pi_t = tf.argmax(q_t_selector, 1, output_type=tf.int32)
      # Compute distribution for greedy action.
      p_target_z = _slice_with_actions(q_t_probs, pi_t)

      # Project using the Cramer distance
      target = tf.stop_gradient(_l2_project(target_z, p_target_z, atoms_tm1))

    logit_qa_tm1 = _slice_with_actions(logits_q_tm1, a_tm1)

    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logit_qa_tm1, labels=target)

    return base_ops.LossOutput(loss, Extra(target))


def categorical_dist_td_learning(atoms_tm1,
                                 logits_v_tm1,
                                 r_t,
                                 pcont_t,
                                 atoms_t,
                                 logits_v_t,
                                 name="CategoricalDistTDLearning"):
  """Implements Distributional TD-learning as TensorFlow ops.

  The function assumes categorical value distributions parameterized by logits.

  See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
  Dabney and Munos. (https://arxiv.org/abs/1707.06887).

  Args:
    atoms_tm1: 1-D tensor containing atom values for first timestep,
      shape `[num_atoms]`.
    logits_v_tm1: Tensor holding logits for first timestep in a batch of
      transitions, shape `[B, num_atoms]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    atoms_t: 1-D tensor containing atom values for second timestep,
      shape `[num_atoms]`.
    logits_v_t: Tensor holding logits for second timestep in a batch of
      transitions, shape `[B, num_atoms]`.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: Tensor containing the batch of losses, shape `[B]`.
    * `extra`: A namedtuple with fields:
        * `target`: Tensor containing the values that `v_tm1` are
        regressed towards, shape `[B, num_atoms]`.

  Raises:
    ValueError: If the tensors do not have the correct rank or compatibility.
  """
  # Rank and compatibility checks.
  assertion_lists = [[logits_v_tm1, logits_v_t], [r_t, pcont_t],
                     [atoms_tm1, atoms_t]]
  base_ops.wrap_rank_shape_assert(assertion_lists, [2, 1, 1], name)

  # Categorical distributional TD-learning op.
  with tf.name_scope(
      name, values=[atoms_tm1, logits_v_tm1, r_t, pcont_t, atoms_t,
                    logits_v_t]):

    with tf.name_scope("target"):
      # Scale and shift time-t distribution atoms by discount and reward.
      target_z = r_t[:, None] + pcont_t[:, None] * atoms_t[None, :]
      v_t_probs = tf.nn.softmax(logits_v_t)

      # Project using the Cramer distance
      target = tf.stop_gradient(_l2_project(target_z, v_t_probs, atoms_tm1))

    loss = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits_v_tm1, labels=target)

    return base_ops.LossOutput(loss, Extra(target))
