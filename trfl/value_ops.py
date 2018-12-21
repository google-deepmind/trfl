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
"""TensorFlow ops for state value learning."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import tensorflow as tf
from trfl import base_ops
from trfl import sequence_ops


TDExtra = collections.namedtuple("td_extra", ["target", "td_error"])
TDLambdaExtra = collections.namedtuple(
    "td_lambda_extra", ["temporal_differences", "discounted_returns"])


def td_learning(v_tm1, r_t, pcont_t, v_t, name="TDLearning"):
  """Implements the TD(0)-learning loss as a TensorFlow op.

  The TD loss is `0.5` times the squared difference between `v_tm1` and
  the target `r_t + pcont_t * v_t`.

  See "Learning to Predict by the Methods of Temporal Differences" by Sutton.
  (https://link.springer.com/article/10.1023/A:1022633531479).

  Args:
    v_tm1: Tensor holding values at previous timestep, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    v_t: Tensor holding values at current timestep, shape `[B]`.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `v_tm1`, shape `[B]`.
        * `td_error`: batch of temporal difference errors, shape `[B]`.
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert([[v_tm1, v_t, r_t, pcont_t]], [1], name)

  # TD(0)-learning op.
  with tf.name_scope(name, values=[v_tm1, r_t, pcont_t, v_t]):

    # Build target.
    target = tf.stop_gradient(r_t + pcont_t * v_t)

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - v_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, TDExtra(target, td_error))


def generalized_lambda_returns(rewards,
                               pcontinues,
                               values,
                               bootstrap_value,
                               lambda_=1,
                               name="generalized_lambda_returns"):
  """Computes lambda-returns along a batch of (chunks of) trajectories.

  For lambda=1 these will be multistep returns looking ahead from each
  state to the end of the chunk, where bootstrap_value is used. If you pass an
  entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
  return / TD(1) target.

  For lambda=0 these are one-step TD(0) targets.

  For inbetween values of lambda these are lambda-returns / TD(lambda) targets,
  except that traces are always cut off at the end of the chunk, since we can't
  see returns beyond then. If you pass an entire trajectory with zeros for
  bootstrap_value though, then they're plain TD(lambda) targets.

  lambda can also be a tensor of values in [0, 1], determining the mix of
  bootstrapping vs further accumulation of multistep returns at each timestep.
  This can be used to implement Retrace and other algorithms. See
  `sequence_ops.multistep_forward_view` for more info on this. Another way to
  think about the end-of-chunk cutoff is that lambda is always effectively zero
  on the timestep after the end of the chunk, since at the end of the chunk we
  rely entirely on bootstrapping and can't accumulate returns looking further
  into the future.

  The sequences in the tensors should be aligned such that an agent in a state
  with value `V` transitions into another state with value `V'`, receiving
  reward `r` and pcontinue `p`. Then `V`, `r` and `p` are all at the same index
  `i` in the corresponding tensors. `V'` is at index `i+1`, or in the
  `bootstrap_value` tensor if `i == T`.

  Subtracting `values` from these lambda-returns will yield estimates of the
  advantage function which can be used for both the policy gradient loss and
  the baseline value function loss in A3C / GAE.

  Args:
    rewards: 2-D Tensor with shape `[T, B]`.
    pcontinues: 2-D Tensor with shape `[T, B]`.
    values: 2-D Tensor containing estimates of the state values for timesteps
      0 to `T-1`. Shape `[T, B]`.
    bootstrap_value: 1-D Tensor containing an estimate of the value of the
      final state at time `T`, used for bootstrapping the target n-step
      returns. Shape `[B]`.
    lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
    name: Customises the name_scope for this op.

  Returns:
    2-D Tensor with shape `[T, B]`
  """
  values.get_shape().assert_has_rank(2)
  rewards.get_shape().assert_has_rank(2)
  pcontinues.get_shape().assert_has_rank(2)
  bootstrap_value.get_shape().assert_has_rank(1)
  scoped_values = [rewards, pcontinues, values, bootstrap_value, lambda_]
  with tf.name_scope(name, values=scoped_values):
    if lambda_ == 1:
      # This is actually equivalent to the branch below, just an optimisation
      # to avoid unnecessary work in this case:
      return sequence_ops.scan_discounted_sum(
          rewards,
          pcontinues,
          initial_value=bootstrap_value,
          reverse=True,
          back_prop=False,
          name="multistep_returns")
    else:
      v_tp1 = tf.concat(
          axis=0, values=[values[1:, :],
                          tf.expand_dims(bootstrap_value, 0)])
      # `back_prop=False` prevents gradients flowing into values and
      # bootstrap_value, which is what you want when using the bootstrapped
      # lambda-returns in an update as targets for values.
      return sequence_ops.multistep_forward_view(
          rewards,
          pcontinues,
          v_tp1,
          lambda_,
          back_prop=False,
          name="generalized_lambda_returns")


def td_lambda(state_values,
              rewards,
              pcontinues,
              bootstrap_value,
              lambda_=1,
              name="BaselineLoss"):
  """Constructs a TensorFlow graph computing the L2 loss for sequences.

  This loss learns the baseline for advantage actor-critic models. Gradients
  for this loss flow through each tensor in `state_values`, but no other
  input tensors. The baseline is regressed towards the n-step bootstrapped
  returns given by the reward/pcontinue sequence.

  This function is designed for batches of sequences of data. Tensors are
  assumed to be time major (i.e. the outermost dimension is time, the second
  outermost dimension is the batch dimension). We denote the sequence length
  in the shapes of the arguments with the variable `T`, the batch size with
  the variable `B`, neither of which needs to be known at construction time.
  Index `0` of the time dimension is assumed to be the start of the sequence.

  `rewards` and `pcontinues` are the sequences of data taken directly from the
  environment, possibly modulated by a discount. `state_values` are the
  sequences of (typically learnt) estimates of the values of the states
  visited along a batch of trajectories.

  The sequences in the tensors should be aligned such that an agent in a state
  with value `V` that takes an action transitions into another state
  with value `V'`, receiving reward `r` and pcontinue `p`. Then `V`, `r`
  and `p` are all at the same index `i` in the corresponding tensors. `V'` is
  at index `i+1`, or in the `bootstrap_value` tensor if `i == T`.

  See "High-dimensional continuous control using generalized advantage
  estimation" by Schulman, Moritz, Levine et al.
  (https://arxiv.org/abs/1506.02438).

  Args:
    state_values: 2-D Tensor of state-value estimates with shape `[T, B]`.
    rewards: 2-D Tensor with shape `[T, B]`.
    pcontinues: 2-D Tensor with shape `[T, B]`.
    bootstrap_value: 1-D Tensor with shape `[B]`.
    lambda_: an optional scalar or 2-D Tensor with shape `[T, B]`.
    name: Customises the name_scope for this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * temporal_differences, Tensor of shape `[T, B]`
        * discounted_returns, Tensor of shape `[T, B]`
  """
  scoped_values = [state_values, rewards, pcontinues, bootstrap_value]
  with tf.name_scope(name, values=scoped_values):
    discounted_returns = generalized_lambda_returns(
        rewards, pcontinues, state_values, bootstrap_value, lambda_)
    temporal_differences = discounted_returns - state_values
    loss = 0.5 * tf.reduce_sum(
        tf.square(temporal_differences), axis=0, name="l2_loss")

    return base_ops.LossOutput(
        loss, TDLambdaExtra(
            temporal_differences=temporal_differences,
            discounted_returns=discounted_returns))


def qv_max(v_tm1, r_t, pcont_t, q_t, name="QVMAX"):
  """Implements the QVMAX learning loss as a TensorFlow op.

  The QVMAX loss is `0.5` times the squared difference between `v_tm1` and
  the target `r_t + pcont_t * max q_t`, where `q_t` is separately learned
  through QV learning (c.f. `action_value_ops.qv_learning`).

  See "The QV Family Compared to Other Reinforcement Learning Algorithms" by
  Wiering and van Hasselt (2009).
  (http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.713.1931)

  Args:
    v_tm1: Tensor holding values at previous timestep, shape `[B]`.
    r_t: Tensor holding rewards, shape `[B]`.
    pcont_t: Tensor holding pcontinue values, shape `[B]`.
    q_t: Tensor of action values at current timestep, shape `[B, num_actions]`.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `v_tm1`, shape `[B]`.
        * `td_error`: batch of temporal difference errors, shape `[B]`.
  """
  # Rank and compatibility checks.
  base_ops.wrap_rank_shape_assert([[v_tm1, r_t, pcont_t], [q_t]], [1, 2], name)

  # The QVMAX op.
  with tf.name_scope(name, values=[v_tm1, r_t, pcont_t, q_t]):

    # Build target.
    target = tf.stop_gradient(r_t + pcont_t * tf.reduce_max(q_t, axis=1))

    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - v_tm1
    loss = 0.5 * tf.square(td_error)
    return base_ops.LossOutput(loss, TDExtra(target, td_error))
