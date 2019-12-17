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
"""Tests for target_update_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import tensorflow.compat.v2 as tf
from trfl import target_update_ops


class UpdateTargetVariablesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests function update_target_variables."""

  @parameterized.parameters({'use_locking': True}, {'use_locking': False})
  def testFullUpdate(self, use_locking):
    """Tests full update of the target variables from the source variables."""
    target_variables = [
        tf.Variable(tf.random.normal(shape=[1, 2])),
        tf.Variable(tf.random.normal(shape=[3, 4])),
    ]
    source_variables = [
        tf.Variable(tf.random.normal(shape=[1, 2])),
        tf.Variable(tf.random.normal(shape=[3, 4])),
    ]
    target_update_ops.update_target_variables(
        target_variables, source_variables, use_locking=use_locking)

    self.assertAllClose(target_variables[0], source_variables[0])
    self.assertAllClose(target_variables[1], source_variables[1])

  @parameterized.parameters({'use_locking': True}, {'use_locking': False})
  def testIncrementalUpdate(self, use_locking):
    """Tests incremental update of the target variables."""
    target_variable = tf.Variable(tf.random.normal(shape=[1, 2]))
    before_assign = tf.identity(target_variable)
    source_variable = tf.Variable(tf.random.normal(shape=[1, 2]))
    tau = .1
    target_update_ops.update_target_variables(
        [target_variable], [source_variable], tau=tau, use_locking=use_locking)
    self.assertAllClose(target_variable,
                        tau * source_variable + (1 - tau) * before_assign)

  def testIncompatibleLength(self):
    """Tests error when variable lists have unequal length."""
    with self.test_session():
      target_variables = [tf.Variable(tf.random.normal(shape=[1, 2]))]
      source_variables = [
          tf.Variable(tf.random.normal(shape=[1, 2])),
          tf.Variable(tf.random.normal(shape=[3, 4])),
      ]
      self.assertRaises(ValueError, target_update_ops.update_target_variables,
                        target_variables, source_variables)

  def testIncompatibleShape(self):
    """Tests error when variable lists have unequal shapes."""
    with self.test_session():
      target_variables = [
          tf.Variable(tf.random.normal(shape=[1, 2])),
          tf.Variable(tf.random.normal(shape=[1, 2])),
      ]
      source_variables = [
          tf.Variable(tf.random.normal(shape=[1, 2])),
          tf.Variable(tf.random.normal(shape=[3, 4])),
      ]
      self.assertRaises(ValueError, target_update_ops.update_target_variables,
                        target_variables, source_variables)

  def testInvalidTypeTau(self):
    """Tests error when tau has wrong type."""
    target_variables = [tf.Variable(tf.random.normal(shape=[1, 2]))]
    source_variables = [tf.Variable(tf.random.normal(shape=[1, 2]))]
    self.assertRaises(TypeError, target_update_ops.update_target_variables,
                      target_variables, source_variables, 1)

  def testInvalidRangeTau(self):
    """Tests error when tau is outside permitted range."""
    target_variables = [tf.Variable(tf.random.normal(shape=[1, 2]))]
    source_variables = [tf.Variable(tf.random.normal(shape=[1, 2]))]
    self.assertRaises(ValueError, target_update_ops.update_target_variables,
                      target_variables, source_variables, -0.1)
    self.assertRaises(ValueError, target_update_ops.update_target_variables,
                      target_variables, source_variables, 1.1)


class PeriodicTargetUpdateTest(tf.test.TestCase, parameterized.TestCase):
  """Tests function period_target_update."""

  @parameterized.parameters(
      {'use_locking': True, 'update_period': 1},
      {'use_locking': False, 'update_period': 1},
      {'use_locking': True, 'update_period': 3},
      {'use_locking': False, 'update_period': 3}
  )
  def testPeriodicTargetUpdate(self, use_locking, update_period):
    """Tests that the simple success case works as expected.

    This is an integration test. The periodically and update parts are
    unit-tested in the preceding.

    Args:
      use_locking: value for `periodic_target_update`'s `use_locking` argument.
      update_period: how often an update should happen.
    """
    target_variable = tf.Variable(tf.zeros([1, 2]))
    source_variable = tf.Variable(tf.random.normal([1, 2]))
    increment = tf.ones([1, 2])
    counter = tf.Variable(update_period, dtype=tf.int64)

    for step in range(3 * update_period):
      source_variable.assign_add(increment)
      target_update_ops.periodic_target_update(
          [target_variable],
          [source_variable],
          update_period=update_period,
          counter=counter,
          use_locking=use_locking)
      if step % update_period == 0:
        self.assertAllClose(target_variable, source_variable)
      else:
        self.assertNotAllClose(target_variable, source_variable)

if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()

