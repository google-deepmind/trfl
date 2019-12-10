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
"""Tests for multistep_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v2 as tf
from trfl import sequence_ops


def get_n_step_backup(rewards, pcontinues, state_values, start, n):
  """Evaluates a single n-step backup (return) starting at position start.

    http://incompleteideas.net/sutton/book/ebook/node73.html (Eq. 7.1)

  Args:
    rewards: a list containing a sequence of rewards.
    pcontinues: a list containing a sequence of discounts.
    state_values: a list containing a sequence of state-values.
    start: position at which the n-Step return has to be evaluated.
    n: number of steps over which rewards are summed before adding the
        respective bootstrapped state-value.

  Returns:
    Sum of discounted rewards plus discounted bootstrapped value.
  """
  accumulator = 0.0
  k = 1.0
  for i in xrange(start, start + n):
    accumulator += k * rewards[i]
    k *= pcontinues[i]
  accumulator += k * state_values[start + n - 1]
  return accumulator


def get_complex_n_step_backup(rewards, pcontinues, state_values, start, n,
                              lambda_):
  """Evaluates a complex n=step backup (sum of lambda-weighted n-step backups).

    http://incompleteideas.net/sutton/book/ebook/node74.html (Eq. 7.3)

  Args:
    rewards: a list containing rewards.
    pcontinues: a list containing discounts.
    state_values: a list containing boostrapped state values.
    start: position at which the n-Step return has to be evaluated.
    n: number of steps over which rewards are summed before adding respective
        boostrapped state values.
    lambda_: mixing parameter lambda.

  Returns:
    A single complex backup.
  """
  accumulator = 0.0
  for t in xrange(1, n):
    value = get_n_step_backup(rewards, pcontinues, state_values, start, t)
    weight = (1 - lambda_) * (lambda_ ** (t - 1))
    accumulator += + value * weight
  value = get_n_step_backup(rewards, pcontinues, state_values, start, n)
  weight = lambda_ ** (n - 1)
  accumulator += value * weight
  return accumulator


def get_complex_n_step_backup_at_all_times(rewards, pcontinues, state_values,
                                           lambda_):
  """Evaluates complex n-step backups at all time-points.

  Args:
    rewards: a list containing rewards.
    pcontinues: a list containing discounts.
    state_values: a list containing bootstrapped state values.
    lambda_: mixing parameter lambda.

  Returns:
    A list containing complex backups at all times.
  """
  res = []
  length = len(rewards)
  for i in xrange(0, length):
    res.append(get_complex_n_step_backup(rewards, pcontinues, state_values, i,
                                         length - i, lambda_))
  return res


class ScanDiscountedSumTest(tf.test.TestCase):

  def testScanSumShapeInference(self):
    """scan_discounted_sum should support static shape inference."""
    sequence_in = tf.zeros(dtype=tf.float32, shape=[1647, 2001])
    decays_in = tf.zeros(dtype=tf.float32, shape=[1647, 2001])
    bootstrap = tf.zeros(dtype=tf.float32, shape=[2001])
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=False)
    self.assertAllEqual(result.get_shape(), [1647, 2001])

    # Let's do it again with higher-dimensional inputs.
    sequence_in = tf.zeros(dtype=tf.float32, shape=[4, 8, 15, 16, 23, 42])
    decays_in = tf.zeros(dtype=tf.float32, shape=[4, 8, 15, 16, 23, 42])
    bootstrap = tf.zeros(dtype=tf.float32, shape=[8, 15, 16, 23, 42])
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=False)
    self.assertAllEqual(result.get_shape(), [4, 8, 15, 16, 23, 42])

  def testScanSumShapeInferenceWithSeqLen(self):
    """scan_discounted_sum should support static shape inference."""
    sequence_in = tf.zeros(dtype=tf.float32, shape=[1647, 2001])
    decays_in = tf.zeros(dtype=tf.float32, shape=[1647, 2001])
    bootstrap = tf.zeros(dtype=tf.float32, shape=[2001])
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=False)
    self.assertAllEqual(result.get_shape(), [1647, 2001])

    # Let's do it again with higher-dimensional inputs.
    sequence_in = tf.zeros(dtype=tf.float32, shape=[4, 8, 15, 16, 23, 42])
    decays_in = tf.zeros(dtype=tf.float32, shape=[4, 8, 15, 16, 23, 42])
    bootstrap = tf.zeros(dtype=tf.float32, shape=[8, 15, 16, 23, 42])
    sequence_lengths = tf.zeros(dtype=tf.float32, shape=[8])
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=False,
                                              sequence_lengths=sequence_lengths)
    self.assertAllEqual(result.get_shape(), [4, 8, 15, 16, 23, 42])

  def testScanSumWithDecays(self):
    sequence = [[3, 1, 5, 2, 1], [-1.7, 1.2, 2.3, 0, 1]]
    decays = [[0.5, 0.9, 1.0, 0.1, 0.5], [0.9, 0.5, 0.0, 2, 0.8]]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32))
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32))
    bootstrap = tf.constant([0, 1.5], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=False)
    expected_result = tf.constant(
        [[3,
          3 * 0.9 + 1,
          (3 * 0.9 + 1) * 1.0 + 5,
          ((3 * 0.9 + 1) * 1.0 + 5) * 0.1 + 2,
          (((3 * 0.9 + 1) * 1.0 + 5) * 0.1 + 2) * 0.5 + 1],
         [-1.7 + 1.5 * 0.9,
          (-1.7 + 1.5 * 0.9) * 0.5 + 1.2,
          ((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3,
          (((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3) * 2 + 0,
          ((((-1.7 + 1.5 * 0.9) * 0.5 + 1.2) * 0.0 + 2.3) * 2 + 0) * 0.8 + 1,
         ]], dtype=tf.float32)
    self.assertAllClose(result.numpy(),
                        tf.transpose(expected_result).numpy())

  def testScanSumWithDecaysWithSeqLen(self):
    sequence = [[3, 1, 5, 2, 1], [-1.7, 1.2, 2.3, 0, 1]]
    decays = [[0.5, 0.9, 1.0, 0.1, 0.5], [0.9, 0.5, 0.0, 2, 0.8]]
    sequence_lengths = [0, 2]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32))
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32))
    bootstrap = tf.constant([0, 1.5], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(
        sequence_in, decays_in, bootstrap, reverse=False,
        sequence_lengths=sequence_lengths)
    expected_result = tf.constant(
        [[0, 0, 0, 0, 0],
         [-1.7 + 1.5 * 0.9, (-1.7 + 1.5 * 0.9) * 0.5 + 1.2, 0, 0, 0]],
        dtype=tf.float32)
    self.assertAllClose(result.numpy(),
                        tf.transpose(expected_result).numpy())

  def testScanSumEquivalenceWithSeqLen(self):
    sequence_lengths = [0, 2]
    bootstrap = tf.constant([0.5, 1.5], dtype=tf.float32)

    sequence = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    decays = [[.1, .2, .3, .4, .5], [.6, .7, .8, .9, .10]]

    eq_sequence = [[0, 0, 0, 0, 0], [6, 7, 0, 0, 0]]
    eq_decays = [[0, 0, 0, 0, 0], [.6, .7, 0, 0, 0]]

    eq_reverse_sequence = [[0, 0, 0, 0, 0], [7, 6, 0, 0, 0]]
    eq_reverse_decays = [[0, 0, 0, 0, 0], [.7, .6, 0, 0, 0]]

    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32))
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32))
    eq_sequence_in = tf.transpose(tf.constant(eq_sequence, dtype=tf.float32))
    eq_decays_in = tf.transpose(tf.constant(eq_decays, dtype=tf.float32))
    eq_reverse_sequence_in = tf.transpose(
        tf.constant(eq_reverse_sequence, dtype=tf.float32))
    eq_reverse_decays_in = tf.transpose(
        tf.constant(eq_reverse_decays, dtype=tf.float32))

    eq_result = sequence_ops.scan_discounted_sum(
        sequence_in, decays_in, bootstrap, reverse=False,
        sequence_lengths=sequence_lengths)
    exp_eq_result = sequence_ops.scan_discounted_sum(
        eq_sequence_in, eq_decays_in, bootstrap)

    eq_reverse_result = sequence_ops.scan_discounted_sum(
        sequence_in, decays_in, bootstrap, reverse=True,
        sequence_lengths=sequence_lengths)
    exp_eq_reverse_result = sequence_ops.scan_discounted_sum(
        eq_reverse_sequence_in, eq_reverse_decays_in, bootstrap)
    exp_eq_reverse_result = tf.reverse_sequence(
        exp_eq_reverse_result, sequence_lengths, seq_axis=0, batch_axis=1)

    self.assertAllClose(eq_result.numpy(),
                        exp_eq_result.numpy())
    self.assertAllClose(eq_reverse_result.numpy(),
                        exp_eq_reverse_result.numpy())

  def testScanSumWithDecaysReverse(self):
    sequence = [[3, 1, 5], [-1.7, 1.2, 2.3]]
    decays = [[0.5, 0.9, 1.0], [0.9, 0.5, 0.3]]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32))
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32))
    bootstrap = tf.constant([0, 1.5], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=True)
    expected_result = tf.constant(
        [[(5 * 0.9 + 1) * 0.5 + 3,
          5 * 0.9 + 1,
          5],
         [((2.3 + 0.3 * 1.5) * 0.5 + 1.2) * 0.9 - 1.7,
          (2.3 + 0.3 * 1.5) * 0.5 + 1.2,
          2.3 + 0.3 * 1.5,
         ]], dtype=tf.float32)
    self.assertAllClose(result.numpy(),
                        tf.transpose(expected_result).numpy())

  def testScanSumWithDecaysReverseWithSeqLen(self):
    sequence = [[3, 1, 5], [-1.7, 1.2, 2.3]]
    decays = [[0.5, 0.9, 1.0], [0.9, 0.5, 0.3]]
    sequence_lengths = [2, 0]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32))
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32))
    bootstrap = tf.constant([2.5, 1.5], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(
        sequence_in, decays_in, bootstrap, reverse=True,
        sequence_lengths=sequence_lengths)
    expected_result = tf.constant(
        [[(0.9 * 2.5 + 1) * 0.5 + 3, (0.9 * 2.5 + 1), 0], [0, 0, 0]],
        dtype=tf.float32)
    self.assertAllClose(result.numpy(),
                        tf.transpose(expected_result).numpy())

  def testScanSumWithDecaysReverse3D(self):
    """scan_discounted_sum vs. higher-dimensional arguments."""
    sequence = [[[3, 33], [1, 11], [5, 55]],
                [[-1.7, -17], [1.2, 12], [2.3, 23]]]
    decays = [[[0.5, 5], [0.9, 9], [1.0, 10]],
              [[0.9, 9], [0.5, 5], [0.3, 3]]]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32),
                               perm=[1, 0, 2])
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32),
                             perm=[1, 0, 2])
    bootstrap = tf.constant([[0, 0], [1.5, 15]], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(sequence_in, decays_in,
                                              bootstrap,
                                              reverse=True)
    expected_result = tf.constant(
        [[[(5 * 0.9 + 1) * 0.5 + 3,
           (55 * 9 + 11) * 5 + 33],
          [5 * 0.9 + 1,
           55 * 9 + 11],
          [5,
           55]],
         [[((2.3 + 0.3 * 1.5) * 0.5 + 1.2) * 0.9 - 1.7,
           ((23 + 3 * 15) * 5 + 12) * 9 - 17],
          [(2.3 + 0.3 * 1.5) * 0.5 + 1.2,
           (23 + 3 * 15) * 5 + 12],
          [2.3 + 0.3 * 1.5,
           23 + 3 * 15]]],
        dtype=tf.float32)
    self.assertAllClose(result.numpy(),
                        tf.transpose(expected_result,
                                     perm=[1, 0, 2]).numpy())

  def testScanSumWithDecaysReverse3DWithSeqLen(self):
    """scan_discounted_sum vs. higher-dimensional arguments."""
    sequence = [[[3, 33], [1, 11], [5, 55]],
                [[-1.7, -17], [1.2, 12], [2.3, 23]]]
    decays = [[[0.5, 5], [0.9, 9], [1.0, 10]],
              [[0.9, 9], [0.5, 5], [0.3, 3]]]
    sequence_lengths = [2, 0]
    # We use transpose because it is easier to define the input data in
    # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
    sequence_in = tf.transpose(tf.constant(sequence, dtype=tf.float32),
                               perm=[1, 0, 2])
    decays_in = tf.transpose(tf.constant(decays, dtype=tf.float32),
                             perm=[1, 0, 2])
    bootstrap = tf.constant([[0, 0], [1.5, 15]], dtype=tf.float32)
    result = sequence_ops.scan_discounted_sum(
        sequence_in, decays_in, bootstrap, reverse=True,
        sequence_lengths=sequence_lengths)
    expected_result = np.asarray(
        [[[1 * 0.5 + 3, 11 * 5 + 33], [1, 11], [0, 0]],
         [[0, 0], [0, 0], [0, 0]]], dtype=np.float32)
    self.assertAllClose(result.numpy(),
                        np.transpose(expected_result, axes=[1, 0, 2]))


class MultistepForwardViewTest(tf.test.TestCase):

  def testMultistepForwardView(self):
    # Define input data.
    rewards = [[1, 0, -1, 0, 1], [0.5, 0.8, -0.7, 0.0, 2.1]]
    pcontinues = [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]]
    state_values = [[3, 1, 5, -5, 3], [-1.7, 1.2, 2.3, 2.2, 2.7]]
    lambda_ = 0.75
    # Evaluate expected complex backups at all time-steps for both batches.
    expected_result = []
    for b in xrange(0, 2):
      expected_result.append(
          get_complex_n_step_backup_at_all_times(rewards[b], pcontinues[b],
                                                 state_values[b], lambda_))
    # Only partially-specify the input shapes - verifies that the
    # dynamically sized Tensors are handled correctly.
    @tf.function(input_signature=[
        tf.TensorSpec(dtype=tf.float32, shape=[None, None]),
        tf.TensorSpec(dtype=tf.float32, shape=[None, None]),
        tf.TensorSpec(dtype=tf.float32, shape=[None, None]),
    ])
    def f(state_values_pl, rewards_pl, pcontinues_pl):
      # We use transpose because it is easier to define the input data in
      # BxT (batch x time) form, while scan_discounted_sum assumes TxB form.
      state_values_in = tf.transpose(state_values_pl)
      rewards_in = tf.transpose(rewards_pl)
      pcontinues_in = tf.transpose(pcontinues_pl)
      # Evaluate complex backups.
      result = sequence_ops.multistep_forward_view(rewards_in, pcontinues_in,
                                                   state_values_in, lambda_)
      return result

    expected = tf.transpose(tf.constant(expected_result,
                                        dtype=tf.float32)).numpy()
    actual = f(state_values_pl=state_values, rewards_pl=rewards,
               pcontinues_pl=pcontinues)
    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
