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
"""Tensorflow ops for multistep return evaluation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def _reverse_seq(sequence, sequence_lengths=None):
  """Reverse sequence along dim 0.

  Args:
    sequence: Tensor of shape [T, B, ...].
    sequence_lengths: (optional) tensor of shape [B]. If `None`, only reverse
      along dim 0.

  Returns:
    Tensor of same shape as sequence with dim 0 reversed up to sequence_lengths.
  """
  if sequence_lengths is None:
    return tf.reverse(sequence, [0])

  sequence_lengths = tf.convert_to_tensor(sequence_lengths)
  with tf.control_dependencies(
      [tf.assert_equal(sequence.shape[1], sequence_lengths.shape[0])]):
    return tf.reverse_sequence(
        sequence, sequence_lengths, seq_axis=0, batch_axis=1)


def scan_discounted_sum(sequence, decay, initial_value, reverse=False,
                        sequence_lengths=None, back_prop=True,
                        name="scan_discounted_sum"):
  """Evaluates a cumulative discounted sum along dimension 0.

    ```python
    if reverse = False:
      result[1] = sequence[1] + decay[1] * initial_value
      result[k] = sequence[k] + decay[k] * result[k - 1]
    if reverse = True:
      result[last] = sequence[last] + decay[last] * initial_value
      result[k] = sequence[k] + decay[k] * result[k + 1]
    ```

  Respective dimensions T, B and ... have to be the same for all input tensors.
  T: temporal dimension of the sequence; B: batch dimension of the sequence.

    if sequence_lengths is set then x1 and x2 below are equivalent:
    ```python
    x1 = zero_pad_to_length(
      scan_discounted_sum(
          sequence[:length], decays[:length], **kwargs), length=T)
    x2 = scan_discounted_sum(sequence, decays,
                             sequence_lengths=[length], **kwargs)
    ```

  Args:
    sequence: Tensor of shape `[T, B, ...]` containing values to be summed.
    decay: Tensor of shape `[T, B, ...]` containing decays/discounts.
    initial_value: Tensor of shape `[B, ...]` containing initial value.
    reverse: Whether to process the sum in a reverse order.
    sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
      (reversed and then) summed.
    back_prop: Whether to backpropagate.
    name: Sets the name_scope for this op.

  Returns:
    Cumulative sum with discount. Same shape and type as `sequence`.
  """
  # Note this can be implemented in terms of cumprod and cumsum,
  # approximately as (ignoring boundary issues and initial_value):
  #
  # cumsum(decay_prods * sequence) / decay_prods
  # where decay_prods = reverse_cumprod(decay)
  #
  # One reason this hasn't been done is that multiplying then dividing again by
  # products of decays isn't ideal numerically, in particular if any of the
  # decays are zero it results in NaNs.
  with tf.name_scope(name, values=[sequence, decay, initial_value]):
    if sequence_lengths is not None:
      # Zero out sequence and decay beyond sequence_lengths.
      with tf.control_dependencies(
          [tf.assert_equal(sequence.shape[0], decay.shape[0])]):
        mask = tf.sequence_mask(sequence_lengths, maxlen=sequence.shape[0],
                                dtype=sequence.dtype)
        mask = tf.transpose(mask)

      # Adding trailing dimensions to mask to allow for broadcasting.
      to_seq = mask.shape.dims + [1] * (sequence.shape.ndims - mask.shape.ndims)
      sequence *= tf.reshape(mask, to_seq)
      to_decay = mask.shape.dims + [1] * (decay.shape.ndims - mask.shape.ndims)
      decay *= tf.reshape(mask, to_decay)

    sequences = [sequence, decay]
    if reverse:
      sequences = [_reverse_seq(s, sequence_lengths) for s in sequences]

    summed = tf.scan(lambda a, x: x[0] + x[1] * a,
                     sequences,
                     initializer=tf.convert_to_tensor(initial_value),
                     parallel_iterations=1,
                     back_prop=back_prop)
    if reverse:
      summed = _reverse_seq(summed, sequence_lengths)
    return summed


def multistep_forward_view(rewards, pcontinues, state_values, lambda_,
                           back_prop=True, sequence_lengths=None,
                           name="multistep_forward_view_op"):
  """Evaluates complex backups (forward view of eligibility traces).

    ```python
    result[t] = rewards[t] +
        pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
    result[last] = rewards[last] + pcontinues[last]*state_values[last]
    ```

    This operation evaluates multistep returns where lambda_ parameter controls
    mixing between full returns and boostrapping. It is users responsibility
    to provide state_values. Depending on how state_values are evaluated this
    function can evaluate targets for Q(lambda), Sarsa(lambda) or some other
    multistep boostrapping algorithm.

    More information about a forward view is given here:
      http://incompleteideas.net/sutton/book/ebook/node74.html

    Please note that instead of evaluating traces and then explicitly summing
    them we instead evaluate mixed returns in the reverse temporal order
    by using the recurrent relationship given above.

    The parameter lambda_ can either be a constant value (e.g for Peng's
    Q(lambda) and Sarsa(_lambda)) or alternatively it can be a tensor containing
    arbitrary values (Watkins' Q(lambda), Munos' Retrace, etc).

    The result of evaluating this recurrence relation is a weighted sum of
    n-step returns, as depicted in the diagram below. One strategy to prove this
    equivalence notes that many of the terms in adjacent n-step returns
    "telescope", or cancel out, when the returns are summed.

    Below L3 is lambda at time step 3 (important: this diagram is 1-indexed, not
    0-indexed like Python). If lambda is scalar then L1=L2=...=Ln.
    g1,...,gn are discounts.

    ```
    Weights:  (1-L1)        (1-L2)*l1      (1-L3)*l1*l2  ...  L1*L2*...*L{n-1}
    Returns:    |r1*(g1)+     |r1*(g1)+      |r1*(g1)+          |r1*(g1)+
              v1*(g1)         |r2*(g1*g2)+   |r2*(g1*g2)+       |r2*(g1*g2)+
                            v2*(g1*g2)       |r3*(g1*g2*g3)+    |r3*(g1*g2*g3)+
                                           v3*(g1*g2*g3)               ...
                                                                |rn*(g1*...*gn)+
                                                              vn*(g1*...*gn)
    ```

  Args:
    rewards: Tensor of shape `[T, B]` containing rewards.
    pcontinues: Tensor of shape `[T, B]` containing discounts.
    state_values: Tensor of shape `[T, B]` containing state values.
    lambda_: Mixing parameter lambda.
        The parameter can either be a scalar or a Tensor of shape `[T, B]`
        if mixing is a function of state.
    back_prop: Whether to backpropagate.
    sequence_lengths: Tensor of shape `[B]` containing sequence lengths to be
      (reversed and then) summed, same as in `scan_discounted_sum`.
    name: Sets the name_scope for this op.

  Returns:
      Tensor of shape `[T, B]` containing multistep returns.
  """
  with tf.name_scope(name, values=[rewards, pcontinues, state_values]):
    # Regroup:
    #   result[t] = (rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]) +
    #               pcontinues[t]*lambda_*result[t + 1]
    # Define:
    #   sequence[t] = rewards[t] + pcontinues[t]*(1-lambda_)*state_values[t]
    #   discount[t] = pcontinues[t]*lambda_
    # Substitute:
    #   result[t] = sequence[t] + discount[t]*result[t + 1]
    # Boundary condition:
    #   result[last] = rewards[last] + pcontinues[last]*state_values[last]
    # Add and subtract the same quantity at BC:
    #   state_values[last] =
    #       lambda_*state_values[last] + (1-lambda_)*state_values[last]
    # This makes:
    #   result[last] =
    #       (rewards[last] + pcontinues[last]*(1-lambda_)*state_values[last]) +
    #       pcontinues[last]*lambda_*state_values[last]
    # Substitute in definitions for sequence and discount:
    #   result[last] = sequence[last] + discount[last]*state_values[last]
    # Define:
    #   initial_value=state_values[last]
    # We get the following recurrent relationship:
    #   result[last] = sequence[last] + decay[last]*initial_value
    #   result[k] = sequence[k] + decay[k] * result[k + 1]
    # This matches the form of scan_discounted_sum:
    #   result = scan_sum_with_discount(sequence, discount,
    #                                   initial_value = state_values[last])
    sequence = rewards + pcontinues * state_values * (1 - lambda_)
    discount = pcontinues * lambda_
    return scan_discounted_sum(sequence, discount, state_values[-1],
                               reverse=True, sequence_lengths=sequence_lengths,
                               back_prop=back_prop)
