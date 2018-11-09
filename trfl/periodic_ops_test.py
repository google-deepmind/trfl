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

# Dependency imports
import tensorflow as tf
from trfl import periodic_ops


class PeriodicallyTest(tf.test.TestCase):
  """Tests function periodically."""

  def testPeriodically(self):
    """Tests that a function is called exactly every `period` steps."""
    target = tf.Variable(0)
    period = 3

    periodic_update = periodic_ops.periodically(
        body=lambda: target.assign_add(1).op, period=period)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      desired_values = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
      for desired_value in desired_values:
        sess.run(periodic_update)
        result = sess.run(target)
        self.assertEqual(desired_value, result)

  def testPeriodOne(self):
    """Tests that the function is called every time if period == 1."""
    target = tf.Variable(0)

    periodic_update = periodic_ops.periodically(
        body=lambda: target.assign_add(1).op, period=1)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for desired_value in range(1, 11):
        _, result = sess.run([periodic_update, target])
        self.assertEqual(desired_value, result)

  def testPeriodNone(self):
    """Tests that the function is never called if period == None."""
    target = tf.Variable(0)

    periodic_update = periodic_ops.periodically(
        body=lambda: target.assign_add(1).op, period=None)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      desired_value = 0
      for _ in range(1, 11):
        _, result = sess.run([periodic_update, target])
        self.assertEqual(desired_value, result)

  def testFunctionNotCallable(self):
    """Tests value error when argument fn is not a callable."""
    self.assertRaises(
        TypeError, periodic_ops.periodically, body=1, period=2)


if __name__ == '__main__':
  tf.test.main()
