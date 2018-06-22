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
"""Tensorflow ops for common discrete-action value learning rules.

These ops define action value learning rules for discrete, scalar, action
spaces. Actions must be represented as indices in the range `[0, K)` where `K`
is the number of distinct actions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf
from trfl import base_ops
from trfl import sequence_ops

QExtra = collections.namedtuple(
    "qlearning_extra", ["target", "td_error"])
DoubleQExtra = collections.namedtuple(
    "double_qlearning_extra", ["target", "td_error", "best_action"])


def index_with_actions(q_values, actions, static_shapes=False):
  """Equivalent to `q_values[:, actions]`.

  This isn't available as a standalone op, and its impact on performance is
  unknown.

  Works with tensors whose shapes are unspecified or partially-specified,
  but this op will only do shape checking on shape information available at
  graph construction time.

  WARNING: When complete shape information is absent, certain shape
  incompatibilities may not be detected at runtime! (That is, no error will be
  raised.) This can be a source of insidious bugs. See unit tests for extensive
  demonstrations of the kinds of errors that may be missed (grep for the string
  `!!!DANGER!!!`).

  Args:
    q_values: tensor of shape `[B, num_actions]` or `[T, B, num_actions]`
    actions: tensor of shape `[B]` or `[T, B]` containing action indices.
    static_shapes: A boolean. If true, we assume all shapes are static

  Returns:
    Tensor of shape `[B]` or `[T, B]` containing values for the given actions.

  Raises: ValueError if q_values and actions have sizes that are known
    statically (i.e. during graph construction), and those sizes are not
    compatible (see shape descriptions in Args list above).
  """
  # In a simple case, this function can be drastically simplified.
  if static_shapes:
    num_actions = q_values.shape.as_list()[-1]
    one_hot_actions = tf.one_hot(actions, num_actions)
    qas = q_values * one_hot_actions
    return tf.reduce_sum(qas, axis=-1)

  # (The reason that incorrect behaviour is tolerated is that we don't want
  # to risk compromising performance by inserting ops capable of detecting
  # badly-formed inputs at runtime.)
  unspecified_shape_warning_template = (
      "index_with_actions cannot get complete shapes for both argument tensors "
      "\"%s\" and \"%s\" at construction time, and so can't check that their "
      "shapes are valid or compatible. Incorrect indexing behaviour may occur "
      "at runtime without error!")

  with tf.name_scope("index_with_actions", values=[q_values, actions]):
    q_values = tf.convert_to_tensor(q_values)
    actions = tf.convert_to_tensor(actions)

    # If all input shapes are known statically, obtain shapes of arguments and
    # perform compatibility checks. Otherwise, print a warning. The only check
    # we cannot perform statically (and do not attempt elsewhere) is making
    # sure that each action index in actions is in [0, num_actions).
    s_shape_q = q_values.get_shape()  # "s_shape_q" == "static shape q"
    s_shape_a = actions.get_shape()
    if s_shape_q and s_shape_a:  # note: rank-0 "[]" TensorShape is still True.
      try:
        assert (s_shape_q.ndims, s_shape_a.ndims) in [(2, 1), (3, 2)], (
            "Tensor arguments to index_with_actions \"{}\" and \"{}\" are not "
            "shaped in a way that corresponds to minibatch (2-D) or sequence-"
            "minibatch (3-D) indexing".format(q_values.name, actions.name))
        assert s_shape_q[:-1].is_compatible_with(s_shape_a), (
            "Tensor arguments to index_with_actions \"{}\" and \"{}\" have "
            "incompatible shapes of {} and {} respectively".format(
                q_values.name, actions.name, s_shape_q, s_shape_a))
      except AssertionError as e:
        raise ValueError(e)  # Convert AssertionError to ValueError.

    else:  # No shape information is known ahead of time.
      tf.logging.warning(
          unspecified_shape_warning_template, q_values.name, actions.name)

    # Some indexing operations require knowledge of the input tensor shapes and
    # of the rank of the action indices. If the static ranks and shapes are
    # fully known, we use those; otherwise we obtain this knowledge through
    # TF ops.
    s_shape_q_known = s_shape_q and s_shape_q.is_fully_defined()
    s_shape_a_known = s_shape_a and s_shape_a.is_fully_defined()

    shape_q = s_shape_q if s_shape_q_known else tf.shape(q_values)
    shape_a = s_shape_a if s_shape_a_known else tf.shape(actions)
    rank_a = s_shape_a.ndims if s_shape_a else tf.rank(actions)

    # Find the number of action indices, and the number of actions, using TF
    # ops if this is not known statically.
    num_indices = (s_shape_a.num_elements() if s_shape_a_known
                   else tf.reduce_prod(shape_a))
    num_actions = (s_shape_q[rank_a]
                   if s_shape_q and s_shape_a and s_shape_q[rank_a].value
                   else tf.gather(shape_q, rank_a))

    # Our most general method for indexing a Q values tensor of any rank is to
    # flatten the action indices tensor into a vector, then rescale its entries
    # to become indices into a flattened view of the Q values. If we know via
    # static shape information that the action indices are already a vector of
    # appropriate length, we can skip flattening the action indices.
    must_flatten_indices_to_index = not s_shape_a_known or s_shape_a.ndims > 1

    # Perform indexing at last.
    flattened_q = tf.reshape(q_values, [-1])  # Flatten Q values for indexing.
    if must_flatten_indices_to_index:
      actions = tf.reshape(actions, [-1])  # Flatten actions for indexing.
    indexes = tf.range(0, num_indices) * num_actions + actions
    indexed_q = tf.gather(flattened_q, indexes)
    if must_flatten_indices_to_index:
      indexed_q = tf.reshape(indexed_q, shape_a)  # Unflatten indexing result.
    return indexed_q


def qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t, name="QLearning"):
  """Implements the Q-learning loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  the target `r_t + pcont_t * max q_t`.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node65.html).

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape [B x num_actions].
    a_tm1: Tensor holding action indices, shape [B].
    r_t: Tensor holding rewards, shape [B].
    pcont_t: Tensor holding pcontinue values, shape [B].
    q_t: Tensor holding Q-values for second timestep in a batch of
      transitions, shape [B x num_actions].
    name: name to prefix ops created within this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
        * `td_error`: batch of temporal difference errors, shape [B].
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t], [a_tm1, r_t, pcont_t]], [2, 1], name)

  # Q-learning op.
  with tf.name_scope(name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t]):

    # Build target and select head to update.
    with tf.name_scope("target"):
      target = tf.stop_gradient(
          r_t + pcont_t * tf.reduce_max(q_t, axis=1))
    qa_tm1 = index_with_actions(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, QExtra(target, td_error))


def double_qlearning(
    q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector,
    name="DoubleQLearning"):
  """Implements the double Q-learning loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  the target `r_t + pcont_t * q_t_value[argmax q_t_selector]`.

  See "Double Q-learning" by van Hasselt.
  (https://papers.nips.cc/paper/3964-double-q-learning.pdf).

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape [B x num_actions].
    a_tm1: Tensor holding action indices, shape [B].
    r_t: Tensor holding rewards, shape [B].
    pcont_t: Tensor holding pcontinue values, shape [B].
    q_t_value: Tensor of Q-values for second timestep in a batch of transitions,
      used to estimate the value of the best action, shape [B x num_actions].
    q_t_selector: Tensor of Q-values for second timestep in a batch of
      transitions used to estimate the best action, shape [B x num_actions].
    name: name to prefix ops created within this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B]
        * `td_error`: batch of temporal difference errors, shape [B]
        * `best_action`: batch of greedy actions wrt `q_t_selector`, shape [B]
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t_value, q_t_selector], [a_tm1, r_t, pcont_t]], [2, 1], name)

  # double Q-learning op.
  with tf.name_scope(
      name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector]):

    # Build target and select head to update.
    best_action = tf.argmax(q_t_selector, 1, output_type=tf.int32)
    double_q_bootstrapped = index_with_actions(q_t_value, best_action)
    target = tf.stop_gradient(r_t + pcont_t * double_q_bootstrapped)
    qa_tm1 = index_with_actions(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(
        loss, DoubleQExtra(target, td_error, best_action))


def persistent_qlearning(
    q_tm1, a_tm1, r_t, pcont_t, q_t, action_gap_scale=0.5,
    name="PersistentQLearning"):
  """Implements the persistent Q-learning loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  `r_t + pcont_t * [(1-action_gap_scale) max q_t + action_gap_scale qa_t]`

  See "Increasing the Action Gap: New Operators for Reinforcement Learning"
  by Bellemare, Ostrovski, Guez et al. (https://arxiv.org/abs/1512.04860).

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape [B x num_actions].
    a_tm1: Tensor holding action indices, shape [B].
    r_t: Tensor holding rewards, shape [B].
    pcont_t: Tensor holding pcontinue values, shape [B].
    q_t: Tensor holding Q-values for second timestep in a batch of
      transitions, shape [B x num_actions].
      These values are used for estimating the value of the best action. In
      DQN they come from the target network.
    action_gap_scale: coefficient in [0, 1] for scaling the action gap term.
    name: name to prefix ops created within this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
        * `td_error`: batch of temporal difference errors, shape [B].
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t], [a_tm1, r_t, pcont_t]], [2, 1], name)
  base_ops.assert_arg_bounded(action_gap_scale, 0, 1, name, "action_gap_scale")

  # persistent Q-learning op.
  with tf.name_scope(name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t]):

    # Build target and select head to update.
    with tf.name_scope("target"):
      max_q_t = tf.reduce_max(q_t, axis=1)
      qa_t = index_with_actions(q_t, a_tm1)
      corrected_q_t = (1 - action_gap_scale) * max_q_t + action_gap_scale * qa_t
      target = tf.stop_gradient(r_t + pcont_t * corrected_q_t)
    qa_tm1 = index_with_actions(q_tm1, a_tm1)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, QExtra(target, td_error))


def sarsa(q_tm1, a_tm1, r_t, pcont_t, q_t, a_t, name="Sarsa"):
  """Implements the SARSA loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  the target `r_t + pcont_t * q_t[a_t]`.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node64.html.)

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape [B x num_actions].
    a_tm1: Tensor holding action indices, shape [B].
    r_t: Tensor holding rewards, shape [B].
    pcont_t: Tensor holding pcontinue values, shape [B].
    q_t: Tensor holding Q-values for second timestep in a batch of
      transitions, shape [B x num_actions].
    a_t: Tensor holding action indices for second timestep, shape [B].
    name: name to prefix ops created within this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
        * `td_error`: batch of temporal difference errors, shape [B].
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t], [a_t, r_t, pcont_t]], [2, 1], name)

  # SARSA op.
  with tf.name_scope(name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t, a_t]):

    # Select head to update and build target.
    qa_tm1 = index_with_actions(q_tm1, a_tm1)
    qa_t = index_with_actions(q_t, a_t)
    target = tf.stop_gradient(r_t + pcont_t * qa_t)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, QExtra(target, td_error))


def sarse(
    q_tm1, a_tm1, r_t, pcont_t, q_t, probs_a_t, debug=False, name="Sarse"):
  """Implements the SARSE (Expected SARSA) loss as a TensorFlow op.

  The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
  the target `r_t + pcont_t * (sum_a probs_a_t[a] * q_t[a])`.

  See "A Theoretical and Empirical Analysis of Expected Sarsa" by Seijen,
  van Hasselt, Whiteson et al.
  (http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).

  Args:
    q_tm1: Tensor holding Q-values for first timestep in a batch of
      transitions, shape [B x num_actions].
    a_tm1: Tensor holding action indices, shape [B].
    r_t: Tensor holding rewards, shape [B].
    pcont_t: Tensor holding pcontinue values, shape [B].
    q_t: Tensor holding Q-values for second timestep in a batch of
      transitions, shape [B x num_actions].
    probs_a_t: Tensor holding action probabilities for second timestep,
      shape [B x num_actions].
    debug: Boolean flag, when set to True adds ops to check whether probs_a_t
      is a batch of (approximately) valid probability distributions.
    name: name to prefix ops created by this class.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
        * `td_error`: batch of temporal difference errors, shape [B].
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert(
      [[q_tm1, q_t, probs_a_t], [a_tm1, r_t, pcont_t]], [2, 1], name)

  # SARSE (Expected SARSA) op.
  with tf.name_scope(name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t, probs_a_t]):

    # Debug ops.
    deps = []
    if debug:
      cumulative_prob = tf.reduce_sum(probs_a_t, axis=1)
      almost_prob = tf.less(tf.abs(tf.subtract(cumulative_prob, 1.0)), 1e-6)
      deps.append(tf.Assert(
          tf.reduce_all(almost_prob),
          ["probs_a_t tensor does not sum to 1", probs_a_t]))

    # With dependency on possible debug ops.
    with tf.control_dependencies(deps):

      # Select head to update and build target.
      qa_tm1 = index_with_actions(q_tm1, a_tm1)
      target = tf.stop_gradient(
          r_t + pcont_t * tf.reduce_sum(tf.multiply(q_t, probs_a_t), axis=1))

      # Temporal difference error and loss.
      # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
      td_error = target - qa_tm1
      loss = 0.5 * tf.square(td_error)
      return base_ops.LossOutput(loss, QExtra(target, td_error))


def qlambda(
    q_tm1, a_tm1, r_t, pcont_t, q_t, lambda_, static_shapes=False,
    name="GeneralizedQLambda"):
  """Implements Peng's and Watkins' Q(lambda) loss as a TensorFlow op.

  This function is general enough to implement both Peng's and Watkins'
  Q-lambda algorithms.

  See "Reinforcement Learning: An Introduction" by Sutton and Barto.
  (http://incompleteideas.net/book/ebook/node78.html).

  Args:
    q_tm1: `Tensor` holding a sequence of Q-values starting at the first
      timestep; shape `[T, B, num_actions]`
    a_tm1: `Tensor` holding a sequence of action indices, shape `[T, B]`
    r_t: Tensor holding a sequence of rewards, shape `[T, B]`
    pcont_t: `Tensor` holding a sequence of pcontinue values, shape `[T, B]`
    q_t: `Tensor` holding a sequence of Q-values for second timestep;
      shape `[T, B, num_actions]`. In a target network setting,
      this quantity is often supplied by the target network.
    lambda_: a scalar or `Tensor` of shape `[T, B]`
      specifying the ratio of mixing between bootstrapped and MC returns;
      if lambda_ is the same for all time steps then the function implements
      Peng's Q-learning algorithm; if lambda_ = 0 at every sub-optimal action
      and a constant otherwise, then the function implements Watkins'
      Q-learning algorithm. Generally lambda_ can be a Tensor of any values
      in the range [0, 1] supplied by the user.
    static_shapes: A boolean, defaulting to false. If true, we assume all
      shapes are static and pipe this to child functions.
    name: a name of the op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[T, B]`.
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[T, B]`.
        * `td_error`: batch of temporal difference errors, shape `[T, B]`.
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert([[q_tm1, q_t]], [3], name)
  if isinstance(lambda_, tf.Tensor) and lambda_.get_shape().ndims > 0:
    base_ops.wrap_rank_shape_assert([[a_tm1, r_t, pcont_t, lambda_]], [2], name)
  else:
    base_ops.wrap_rank_shape_assert([[a_tm1, r_t, pcont_t]], [2], name)

  # QLambda op.
  with tf.name_scope(name, values=[q_tm1, a_tm1, r_t, pcont_t, q_t]):

    # Build target and select head to update.
    with tf.name_scope("target"):
      state_values = tf.reduce_max(q_t, axis=2)
      target = sequence_ops.multistep_forward_view(
          r_t, pcont_t, state_values, lambda_, back_prop=False)
      target = tf.stop_gradient(target)
    qa_tm1 = index_with_actions(q_tm1, a_tm1, static_shapes)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, QExtra(target, td_error))
