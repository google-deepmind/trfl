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
"""Tests for policy_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from trfl import policy_ops


class EpsilonGreedyTest(tf.test.TestCase):

  def testTieBreaking(self):
    num_actions = 4
    # Given some action values that are all equal:
    action_values = [1.1] * num_actions
    epsilon = 0.

    # We expect the policy to be a uniform distribution.
    expected = [1 / num_actions] * num_actions

    result = policy_ops.epsilon_greedy(action_values, epsilon).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)

  def testGreedy(self):
    # Given some action values with one largest value:
    action_values = [0.5, 0.99, 0.9, 1., 0.1, -0.1, -100.]

    # And zero epsilon:
    epsilon = 0.

    # We expect a deterministic greedy policy that chooses one action.
    expected = [0., 0., 0., 1., 0., 0., 0.]

    result = policy_ops.epsilon_greedy(action_values, epsilon).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)

  def testDistribution(self):
    # Given some action values and non-zero epsilon:
    action_values = [0.9, 1., 0.9, 0.1, -0.6]
    epsilon = 0.1

    # We expect a distribution that concentrates the right probabilities.
    expected = [0.02, 0.92, 0.02, 0.02, 0.02]

    result = policy_ops.epsilon_greedy(action_values, epsilon).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)

  def testBatched(self):
    # Given batched action values:
    action_values = [[1., 2., 3.],
                     [4., 5., 6.],
                     [6., 5., 4.],
                     [3., 2., 1.]]
    epsilon = 0.

    # We expect batched probabilities.
    expected = [[0., 0., 1.],
                [0., 0., 1.],
                [1., 0., 0.],
                [1., 0., 0.]]

    result = policy_ops.epsilon_greedy(action_values, epsilon).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)

  def testFloat64(self):
    # Given action values that are float 64:
    action_values = tf.convert_to_tensor([1., 2., 4., 3.], dtype=tf.float64)
    epsilon = 0.1

    expected = [0.025, 0.025, 0.925, 0.025]

    result = policy_ops.epsilon_greedy(action_values, epsilon).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)

  def testLegalActionsMask(self):
    action_values = [0.9, 1., 0.9, 0.1, -0.6]
    legal_actions_mask = [0., 1., 1., 1., 1.]
    epsilon = 0.1

    expected = [0.00, 0.925, 0.025, 0.025, 0.025]

    result = policy_ops.epsilon_greedy(action_values, epsilon,
                                       legal_actions_mask).probs
    with self.test_session() as sess:
      self.assertAllClose(sess.run(result), expected)


if __name__ == "__main__":
  tf.test.main()
