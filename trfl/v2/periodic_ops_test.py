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
"""Tests for periodic_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from tensorflow.compat import v2 as tf
from trfl import periodic_ops

tf.enable_v2_behavior()


class PeriodicallyTest(tf.test.TestCase):
  """Tests function periodically."""

  def setUp(self):
    super(PeriodicallyTest, self).setUp()
    self.target = tf.Variable(0)
    self.counter = tf.Variable(np.iinfo(np.int64).max, dtype=tf.int64)
    self.periodic_update = functools.partial(
        periodic_ops.periodically,
        body=lambda: self.target.assign_add(1).op, counter=self.counter)

  def testPeriodically(self):
    """Tests that a function is called exactly every `period` steps."""

    # Compare all values at the end so the error messages (if any) show the
    # whole sequence.
    desired_values = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    periodic_values = []
    for _ in desired_values:
      self.periodic_update(period=3)
      periodic_values.append(self.target.numpy())
    self.assertEqual(desired_values, periodic_values)

  def testPeriodicallyGraphMode(self):
    """Tests that a function is called exactly every `period` steps."""

    @tf.function
    def periodic_update():
      return periodic_ops.periodically(
          period=3,
          body=lambda: self.target.assign_add(1).op,
          counter=self.counter)

    # Compare all values at the end so the error messages (if any) show the
    # whole sequence.
    desired_values = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
    periodic_values = []
    for _ in desired_values:
      periodic_update()
      periodic_values.append(self.target.numpy())
    self.assertEqual(desired_values, periodic_values)

  def testPeriodOne(self):
    """Tests that the function is called every time if period == 1."""
    for desired_value in range(1, 11):
      self.periodic_update(period=1)
      self.assertEqual(desired_value, self.target.numpy())

  def testPeriodNone(self):
    """Tests that the function is never called if period == None."""
    desired_value = 0
    for _ in range(1, 11):
      self.periodic_update(period=None)
      self.assertEqual(desired_value, self.target.numpy())

  def testFunctionNotCallable(self):
    """Tests value error when argument fn is not a callable."""
    self.assertRaises(TypeError, periodic_ops.periodically, body=1, period=2)


if __name__ == '__main__':
  tf.test.main()
