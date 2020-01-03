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
import tensorflow.compat.v1 as tf
from trfl import target_update_ops


class UpdateTargetVariablesTest(tf.test.TestCase, parameterized.TestCase):
  """Tests function update_target_variables."""

  @parameterized.parameters({'use_locking': True}, {'use_locking': False})
  def testFullUpdate(self, use_locking):
    """Tests full update of the target variables from the source variables."""
    target_variables = [
        tf.Variable(tf.random_normal(shape=[1, 2])),
        tf.Variable(tf.random_normal(shape=[3, 4])),
    ]
    source_variables = [
        tf.Variable(tf.random_normal(shape=[1, 2])),
        tf.Variable(tf.random_normal(shape=[3, 4])),
    ]
    updated = target_update_ops.update_target_variables(
        target_variables, source_variables, use_locking=use_locking)

    # Collect all the tensors and ops we want to evaluate in the session.
    vars_ops = target_variables + source_variables

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(updated)
      results = sess.run(vars_ops)
      # First target variable is updated with first source variable.
      self.assertAllClose(results[0], results[2])
      # Second target variable is updated with second source variable.
      self.assertAllClose(results[1], results[3])

  @parameterized.parameters({'use_locking': True}, {'use_locking': False})
  def testIncrementalUpdate(self, use_locking):
    """Tests incremental update of the target variables."""
    target_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
    source_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
    updated = target_update_ops.update_target_variables(
        target_variables, source_variables, tau=0.1, use_locking=use_locking)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      before_assign = sess.run(target_variables[0])
      sess.run(updated)
      results = sess.run([target_variables[0], source_variables[0]])
      self.assertAllClose(results[0], 0.1 * results[1] + 0.9 * before_assign)

  def testIncompatibleLength(self):
    """Tests error when variable lists have unequal length."""
    with self.test_session():
      target_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
      source_variables = [
          tf.Variable(tf.random_normal(shape=[1, 2])),
          tf.Variable(tf.random_normal(shape=[3, 4])),
      ]
      self.assertRaises(ValueError, target_update_ops.update_target_variables,
                        target_variables, source_variables)

  def testIncompatibleShape(self):
    """Tests error when variable lists have unequal shapes."""
    with self.test_session():
      target_variables = [
          tf.Variable(tf.random_normal(shape=[1, 2])),
          tf.Variable(tf.random_normal(shape=[1, 2])),
      ]
      source_variables = [
          tf.Variable(tf.random_normal(shape=[1, 2])),
          tf.Variable(tf.random_normal(shape=[3, 4])),
      ]
      self.assertRaises(ValueError, target_update_ops.update_target_variables,
                        target_variables, source_variables)

  def testInvalidTypeTau(self):
    """Tests error when tau has wrong type."""
    target_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
    source_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
    self.assertRaises(TypeError, target_update_ops.update_target_variables,
                      target_variables, source_variables, 1)

  def testInvalidRangeTau(self):
    """Tests error when tau is outside permitted range."""
    target_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
    source_variables = [tf.Variable(tf.random_normal(shape=[1, 2]))]
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
    target_variables = [tf.Variable(tf.zeros([1, 2]))]
    source_variables = [tf.Variable(tf.random_normal([1, 2]))]
    increment = tf.ones([1, 2])

    update_source_op = tf.assign_add(source_variables[0], increment)
    updated = target_update_ops.periodic_target_update(
        target_variables,
        source_variables,
        update_period=update_period,
        use_locking=use_locking)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      for step in range(3 * update_period):
        sess.run(update_source_op)
        sess.run(updated)
        targets, sources = sess.run([target_variables, source_variables])

        if step % update_period == 0:
          self.assertAllClose(targets, sources)
        else:
          self.assertNotAllClose(targets, sources)


if __name__ == '__main__':
  tf.test.main()

