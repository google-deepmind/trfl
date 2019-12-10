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
"""Tests for pixel_control_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from trfl import pixel_control_ops


class PixelControlRewardsTest(tf.test.TestCase):
  """Test the `pixel_control_rewards` op."""

  def setUp(self):
    """Defines example data and expected result for the op."""
    super(PixelControlRewardsTest, self).setUp()

    # Configure.
    self._cell = 2
    obs_size = (5, 2, 4, 4, 3, 2)
    y = obs_size[2] // self._cell
    x = obs_size[3] // self._cell
    channels = np.prod(obs_size[4:])
    rew_size = (obs_size[0]-1, obs_size[1], x, y)

    # Input data.
    self._obs_np = np.random.uniform(size=obs_size)
    self._obs_tf = tf.constant(self._obs_np, dtype=tf.float32)

    # Expected pseudo-rewards.
    abs_diff = np.absolute(self._obs_np[1:] - self._obs_np[:-1])
    abs_diff = abs_diff.reshape((-1,) + obs_size[2:4] + (channels,))
    abs_diff = abs_diff.reshape((-1, y, self._cell, x, self._cell, channels))
    avg_abs_diff = abs_diff.mean(axis=(2, 4, 5))
    self._expected_pseudo_rewards = avg_abs_diff.reshape(rew_size)

  def testPixelControlRewards(self):
    """Compute pseudo rewards from observations."""
    pseudo_rewards_tf = pixel_control_ops.pixel_control_rewards(
        self._obs_tf, self._cell)
    self.assertAllClose(pseudo_rewards_tf, self._expected_pseudo_rewards)


class PixelControlLossTest(tf.test.TestCase):
  """Test the `pixel_control_loss` op."""

  def setUp(self):
    """Defines example data and expected result for the op."""
    super(PixelControlLossTest, self).setUp()

    # Observation shape is (2,2,3) (i.e., height 2, width 2, and 3 channels).
    # We will use no cropping, and a cell size of 1. We have num_actions = 3,
    # meaning our Q values should be (2,2,3). We will set the Q value equal to
    # the observation.
    self.seq_length = 3
    self.batch_size = 1
    self.discount = 0.9
    self.cell_size = 1
    self.scale = 1.0

    # Observations.
    obs1 = np.array([[[1, 2, 3], [3, 4, 5]], [[5, 6, 7], [7, 8, 9]]])
    obs2 = np.array([[[7, 8, 9], [1, 2, 3]], [[3, 4, 5], [5, 6, 7]]])
    obs3 = np.array([[[5, 6, 7], [7, 8, 9]], [[1, 2, 3], [3, 4, 5]]])
    obs4 = np.array([[[3, 4, 5], [5, 6, 7]], [[7, 8, 9], [1, 2, 3]]])

    # Actions.
    action1 = 0
    action2 = 1
    action3 = 2

    # Compute loss for constant discount.
    qa_tm1 = obs3[:, :, action3]
    reward3 = np.mean(np.abs(obs4 - obs3), axis=2)
    qmax_t = np.amax(obs4, axis=2)
    target = reward3 + self.discount * qmax_t
    error3 = target - qa_tm1

    qa_tm1 = obs2[:, :, action2]
    reward2 = np.mean(np.abs(obs3 - obs2), axis=2)
    target = reward2 + self.discount * target
    error2 = target - qa_tm1

    qa_tm1 = obs1[:, :, action1]
    reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
    target = reward1 + self.discount * target
    error1 = target - qa_tm1

    # Compute loss for episode termination with discount 0.
    qa_tm1 = obs1[:, :, action1]
    reward1 = np.mean(np.abs(obs2 - obs1), axis=2)
    target = reward1 + 0. * target
    error1_term = target - qa_tm1

    self.error = np.sum(
        np.square(error1) + np.square(error2) + np.square(error3)) * 0.5
    self.error_term = np.sum(
        np.square(error1_term) + np.square(error2) + np.square(error3)) * 0.5

    # Placeholder data.
    self.observations = np.expand_dims(
        np.stack([obs1, obs2, obs3, obs4], axis=0), axis=1)
    self.action_values = self.observations
    self.actions = np.stack(
        [np.array([action1]), np.array([action2]), np.array([action3])], axis=0)

    # Create ops to feed actions and rewards.
    self.observations_tf = tf.constant(self.observations, dtype=tf.float32)
    self.action_values_tf = tf.constant(self.action_values, dtype=tf.float32)
    self.actions_tf = tf.constant(self.actions, dtype=tf.int32)

  def testPixelControlLossScalarDiscount(self):
    """Compute loss for given observations, actions, values, scalar discount."""

    loss, _ = pixel_control_ops.pixel_control_loss(
        self.observations_tf, self.actions_tf, self.action_values_tf,
        self.cell_size, self.discount, self.scale)
    self.assertNear(loss, self.error, 1e-3)

  def testPixelControlLossTensorDiscount(self):
    """Compute loss for given observations, actions, values, tensor discount."""

    zero_discount = tf.zeros((1, self.batch_size))
    non_zero_discount = tf.tile(
        tf.reshape(self.discount, [1, 1]),
        [self.seq_length - 1, self.batch_size])
    tensor_discount = tf.concat([zero_discount, non_zero_discount], axis=0)
    loss, _ = pixel_control_ops.pixel_control_loss(
        self.observations_tf, self.actions_tf, self.action_values_tf,
        self.cell_size, tensor_discount, self.scale)
    self.assertNear(loss, self.error_term, 1e-3)

  def testPixelControlLossShapes(self):
    with self.assertRaisesRegexp(
        ValueError, "Pixel Control values are not compatible"):
      pixel_control_ops.pixel_control_loss(
          self.observations_tf, self.actions_tf,
          self.action_values_tf[:, :, :-1], self.cell_size, self.discount,
          self.scale)

  def testTensorDiscountShape(self):
    with self.assertRaisesRegexp(
        ValueError, "discount_factor must be a scalar or a tensor of rank 2"):
      tensor_discount = tf.tile(
          tf.reshape(self.discount, [1, 1, 1]),
          [self.seq_length, self.batch_size, 1])
      pixel_control_ops.pixel_control_loss(
          self.observations_tf, self.actions_tf,
          self.action_values_tf, self.cell_size, tensor_discount,
          self.scale)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
