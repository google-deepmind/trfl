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
"""Tests for continuous_retrace_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

import numpy as np
import six
from six.moves import range
import tensorflow.compat.v1 as tf

from trfl import continuous_retrace_ops


def _shaped_arange(*shape):
  """Runs np.arange, converts to float and reshapes."""
  return np.arange(np.prod(shape), dtype=np.float32).reshape(*shape)


def _ground_truth_calculation(discounts, log_rhos, rewards, q_values, values,
                              bootstrap_value, lambda_):
  """Calculates the ground truth for Retrace in python/numpy."""
  qs = []
  seq_len = len(discounts)
  rhos = np.exp(log_rhos)
  cs = np.minimum(rhos, 1.0)
  cs *= lambda_
  # This is a very inefficient way to calculate the Retrace ground truth.
  values_t_plus_1 = np.concatenate([values, bootstrap_value[None, :]], axis=0)
  for s in range(seq_len):
    q_s = np.copy(q_values[s])  # Very important copy...
    delta = rewards[s] + discounts[s] * values_t_plus_1[s + 1] - q_values[s]
    q_s += delta
    for t in range(s + 1, seq_len):
      q_s += (
          np.prod(discounts[s:t], axis=0) * np.prod(cs[s + 1:t + 1], axis=0) *
          (rewards[t] + discounts[t] * values_t_plus_1[t + 1] - q_values[t]))
    qs.append(q_s)
  qs = np.stack(qs, axis=0)
  return qs


class ContinuousRetraceTest(tf.test.TestCase):

  def testSingleElem(self):
    """Tests Retrace with a single element batch and lambda set to 1.0."""
    batch_size = 1
    lambda_ = 1.0
    self._main_test(batch_size, lambda_)

  def testLargerBatch(self):
    """Tests Retrace with a larger batch."""
    batch_size = 2
    lambda_ = 1.0
    self._main_test(batch_size, lambda_)

  def testLowerLambda(self):
    """Tests Retrace with a lower lambda."""
    batch_size = 2
    lambda_ = 0.5
    self._main_test(batch_size, lambda_)

  def _main_test(self, batch_size, lambda_):
    """Tests Retrace against ground truth data calculated in python."""
    seq_len = 5
    # Create log_rhos such that rho will span from near-zero to above the
    # clipping thresholds. In particular, calculate log_rhos in [-2.5, 2.5),
    # so that rho is in approx [0.08, 12.2).
    log_rhos = _shaped_arange(seq_len, batch_size) / (batch_size * seq_len)
    log_rhos = 5 * (log_rhos - 0.5)  # [0.0, 1.0) -> [-2.5, 2.5).
    values = {
        "discounts":
            np.array(  # T, B where B_i: [0.9 / (i+1)] * T
                [[0.9 / (b + 1)
                  for b in range(batch_size)]
                 for _ in range(seq_len)]),
        "rewards":
            _shaped_arange(seq_len, batch_size),
        "q_values":
            _shaped_arange(seq_len, batch_size) / batch_size,
        "values":
            _shaped_arange(seq_len, batch_size) / batch_size,
        "bootstrap_value":
            _shaped_arange(batch_size) + 1.0,  # B
        "log_rhos":
            log_rhos
    }
    placeholders = {
        key: tf.placeholder(tf.float32, shape=val.shape)
        for key, val in six.iteritems(values)
    }
    placeholders = {
        k: tf.placeholder(dtype=p.dtype, shape=[None] * len(p.shape))
        for k, p in placeholders.items()
    }

    retrace_returns = continuous_retrace_ops.retrace_from_importance_weights(
        lambda_=lambda_, **placeholders)

    feed_dict = {placeholders[k]: v for k, v in values.items()}
    with self.test_session() as sess:
      retrace_outputvalues = sess.run(retrace_returns, feed_dict=feed_dict)

    ground_truth_data = _ground_truth_calculation(lambda_=lambda_, **values)

    self.assertAllClose(ground_truth_data, retrace_outputvalues.qs)


if __name__ == "__main__":
  tf.test.main()
