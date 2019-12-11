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
"""Tests for vtrace_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from trfl import vtrace_ops


def _shaped_arange(*shape):
  """Runs np.arange, converts to float and reshapes."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _softmax(logits):
  """Applies softmax non-linearity on inputs."""
  return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def _ground_truth_calculation(discounts, log_rhos, rewards, values,
                              bootstrap_value, clip_rho_threshold,
                              clip_pg_rho_threshold):
  """Calculates the ground truth for V-trace in Python/Numpy."""
  vs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  clipped_rhos = rhos
  if clip_rho_threshold:
    clipped_rhos = np.minimum(rhos, clip_rho_threshold)
  clipped_pg_rhos = rhos
  if clip_pg_rho_threshold:
    clipped_pg_rhos = np.minimum(rhos, clip_pg_rho_threshold)

  # This is a very inefficient way to calculate the V-trace ground truth.
  # We calculate it this way because it is close to the mathematical notation of
  # V-trace.
  # v_s = V(x_s)
  #       + \sum^{T-1}_{t=s} \gamma^{t-s}
  #         * \prod_{i=s}^{t-1} c_i
  #         * \rho_t (r_t + \gamma V(x_{t+1}) - V(x_t))
  # Note that when we take the product over c_i, we write `s:t` as the notation
  # of the paper is inclusive of the `t-1`, but Python is exclusive.
  # Also note that np.prod([]) == 1.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    v_s = np.copy(values[s])  # Very important copy.
    for t in range(s, seq_len):
      v_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s:t - 1],
                                                    axis=0) * clipped_rhos[t] *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - values[t]))
    vs.append(v_s)
  vs = np.stack(vs, axis=0)
  pg_advantages = (
      clipped_pg_rhos * (rewards + discounts * np.concatenate(
          [vs[1:], bootstrap_value[None, :]], axis=0) - values))

  return vtrace_ops.VTraceReturns(vs=vs, pg_advantages=pg_advantages)


class LogProbsFromLogitsAndActionsTest(tf.test.TestCase,
                                       parameterized.TestCase):

  @parameterized.named_parameters(('Batch1', 1), ('Batch2', 2))
  def testLogProbsFromLogitsAndActions(self, batch_size):
    """Tests log_probs_from_logits_and_actions."""
    seq_len = 7
    num_actions = 3

    policy_logits = _shaped_arange(seq_len, batch_size, num_actions) + 10
    actions = np.random.randint(
        0, num_actions - 1, size=(seq_len, batch_size), dtype=np.int32)

    action_log_probs_tensor = vtrace_ops.log_probs_from_logits_and_actions(
        policy_logits, actions)

    # Ground Truth
    # Using broadcasting to create a mask that indexes action logits
    action_index_mask = actions[..., None] == np.arange(num_actions)

    def index_with_mask(array, mask):
      return array[mask].reshape(*array.shape[:-1])

    # Note: Normally log(softmax) is not a good idea because it's not
    # numerically stable. However, in this test we have well-behaved values.
    ground_truth_v = index_with_mask(
        np.log(_softmax(policy_logits)), action_index_mask)

    self.assertAllClose(ground_truth_v, action_log_probs_tensor)


class VtraceTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('Batch1', 1), ('Batch5', 5))
  def testVTrace(self, batch_size):
    """Tests V-trace against ground truth data calculated in python."""
    seq_len = 5

    values = {
        # Note that this is only for testing purposes using well-formed inputs.
        # In practice we'd be more careful about taking log() of arbitrary
        # quantities.
        'log_rhos':
            np.log((_shaped_arange(seq_len, batch_size)) / batch_size /
                   seq_len + 1),
        # T, B where B_i: [0.9 / (i+1)] * T
        'discounts':
            np.array([[0.9 / (b + 1)
                       for b in range(batch_size)]
                      for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,
        'clip_rho_threshold':
            3.7,
        'clip_pg_rho_threshold':
            2.2,
    }

    output = vtrace_ops.vtrace_from_importance_weights(**values)
    ground_truth = _ground_truth_calculation(**values)
    for a, b in zip(ground_truth, output):
      self.assertAllClose(a, b)

  @parameterized.named_parameters(('Batch1', 1), ('Batch2', 2))
  def testVTraceFromLogits(self, batch_size):
    """Tests V-trace calculated from logits."""
    seq_len = 5
    num_actions = 3
    clip_rho_threshold = None  # No clipping.
    clip_pg_rho_threshold = None  # No clipping.

    inputs = {
        'behaviour_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'target_policy_logits':
            _shaped_arange(seq_len, batch_size, num_actions),
        'actions':
            np.random.randint(0, num_actions - 1, size=(seq_len, batch_size)),
        'discounts':
            np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1)
                  for b in range(batch_size)]
                 for _ in range(seq_len)]),
        'rewards':
            _shaped_arange(seq_len, batch_size),
        'values':
            _shaped_arange(seq_len, batch_size) / batch_size,
        'bootstrap_value':
            _shaped_arange(batch_size) + 1.0,  # B
    }

    from_logits_output = vtrace_ops.vtrace_from_logits(
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold,
        **inputs)

    target_log_probs = vtrace_ops.log_probs_from_logits_and_actions(
        inputs['target_policy_logits'], inputs['actions'])
    behaviour_log_probs = vtrace_ops.log_probs_from_logits_and_actions(
        inputs['behaviour_policy_logits'], inputs['actions'])
    log_rhos = target_log_probs - behaviour_log_probs

    # Calculate V-trace using the ground truth logits.
    from_iw = vtrace_ops.vtrace_from_importance_weights(
        log_rhos=log_rhos,
        discounts=inputs['discounts'],
        rewards=inputs['rewards'],
        values=inputs['values'],
        bootstrap_value=inputs['bootstrap_value'],
        clip_rho_threshold=clip_rho_threshold,
        clip_pg_rho_threshold=clip_pg_rho_threshold)

    self.assertAllClose(from_iw.vs, from_logits_output.vs)
    self.assertAllClose(from_iw.pg_advantages,
                        from_logits_output.pg_advantages)
    self.assertAllClose(behaviour_log_probs,
                        from_logits_output.behaviour_action_log_probs)
    self.assertAllClose(target_log_probs,
                        from_logits_output.target_action_log_probs)
    self.assertAllClose(log_rhos, from_logits_output.log_rhos)

  def testHigherRankInputsForIW(self):
    """Checks support for additional dimensions in inputs."""
    inputs = {
        'log_rhos': tf.zeros(shape=[1, 1, 1], dtype=tf.float32),
        'discounts': tf.zeros(shape=[1, 1, 1], dtype=tf.float32),
        'rewards': tf.zeros(shape=[1, 1, 42], dtype=tf.float32),
        'values': tf.zeros(shape=[1, 1, 42], dtype=tf.float32),
        'bootstrap_value': tf.zeros(shape=[1, 42], dtype=tf.float32)
    }
    output = vtrace_ops.vtrace_from_importance_weights(**inputs)
    self.assertEqual(output.vs.shape.as_list()[-1], 42)

  def testInconsistentRankInputsForIW(self):
    """Test one of many possible errors in shape of inputs."""
    inputs = {
        'log_rhos': tf.zeros(dtype=tf.float32, shape=[1, 1, 1]),
        'discounts': tf.zeros(dtype=tf.float32, shape=[1, 1, 1]),
        'rewards': tf.zeros(dtype=tf.float32, shape=[1, 1, 42]),
        'values': tf.zeros(dtype=tf.float32, shape=[1, 1, 42]),
        # Should be [1, 42].
        'bootstrap_value': tf.zeros(dtype=tf.float32, shape=[1])
    }
    with self.assertRaisesRegexp(ValueError, 'must have rank 2'):
      vtrace_ops.vtrace_from_importance_weights(**inputs)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
