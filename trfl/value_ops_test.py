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
"""Tests for value_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import tensorflow as tf
from trfl import value_ops

nest = tf.nest


class TDLearningTest(tf.test.TestCase):
  """Tests for ValueLearning."""

  def setUp(self):
    super(TDLearningTest, self).setUp()
    self.v_tm1 = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
    self.v_t = tf.constant([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=tf.float32)
    self.pcont_t = tf.constant(
        [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=tf.float32)
    self.r_t = tf.constant(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=tf.float32)
    self.value_learning = value_ops.td_learning(
        self.v_tm1, self.r_t, self.pcont_t, self.v_t)

  def testRankCheck(self):
    v_tm1 = tf.placeholder(tf.float32, [None, None])
    with self.assertRaisesRegexp(
        ValueError, 'TDLearning: Error in rank and/or compatibility check'):
      self.value_learning = value_ops.td_learning(
          v_tm1, self.r_t, self.pcont_t, self.v_t)

  def testCompatibilityCheck(self):
    pcont_t = tf.placeholder(tf.float32, [8])
    with self.assertRaisesRegexp(
        ValueError, 'TDLearning: Error in rank and/or compatibility check'):
      self.value_learning = value_ops.td_learning(
          self.v_tm1, self.r_t, pcont_t, self.v_t)

  def testTarget(self):
    """Tests that target value == r_t + pcont_t * v_t."""
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(self.value_learning.extra.target),
          [-1, -1, -1, -1, -0.5, 0, -1, 0, 1])

  def testTDError(self):
    """Tests that td_error == target_value - v_tm1."""
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(self.value_learning.extra.td_error),
          [-2, -2, -2, -2, -1.5, -1, -2, -1, 0])

  def testLoss(self):
    """Tests that loss == 0.5 * td_error^2."""
    with self.test_session() as sess:
      # Loss is 0.5 * td_error^2
      self.assertAllClose(
          sess.run(self.value_learning.loss),
          [2, 2, 2, 2, 1.125, 0.5, 2, 0.5, 0])

  def testGradVtm1(self):
    """Tests that the gradients of negative loss are equal to the td_error."""
    with self.test_session() as sess:
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      gradients = tf.gradients([-self.value_learning.loss], [self.v_tm1])
      grad_v_tm1 = sess.run(gradients[0])
      self.assertAllClose(grad_v_tm1, [-2, -2, -2, -2, -1.5, -1, -2, -1, 0])

  def testNoOtherGradients(self):
    """Tests no gradient propagates through things other than v_tm1."""
    # Gradients are only defined for v_tm1, not any other input.
    gradients = tf.gradients([self.value_learning.loss],
                             [self.v_t, self.r_t, self.pcont_t])
    self.assertEqual(gradients, [None] * len(gradients))


class TDLambdaTest(parameterized.TestCase, tf.test.TestCase):

  def _setUp_td_loss(self, gae_lambda=1, sequence_length=4, batch_size=2):
    t, b = sequence_length, batch_size
    self._state_values = tf.placeholder(tf.float32, shape=(t, b))
    self._rewards = tf.placeholder(tf.float32, shape=(t, b))
    self._pcontinues = tf.placeholder(tf.float32, shape=(t, b))
    self._bootstrap_value = tf.placeholder(tf.float32, shape=(b,))
    loss, (td, discounted_returns) = value_ops.td_lambda(
        state_values=self._state_values,
        rewards=self._rewards,
        pcontinues=self._pcontinues,
        bootstrap_value=self._bootstrap_value,
        lambda_=gae_lambda)
    self._loss = loss
    self._temporal_differences = td
    self._discounted_returns = discounted_returns

  @parameterized.parameters(
      (1,),
      (0.9,),)
  def testShapeInference(self, gae_lambda):
    sequence_length = 4
    batch_size = 2
    self._setUp_td_loss(
        gae_lambda, sequence_length=sequence_length, batch_size=batch_size)
    sequence_batch_shape = tf.TensorShape([sequence_length, batch_size])
    batch_shape = tf.TensorShape(batch_size)
    self.assertEqual(self._discounted_returns.get_shape(), sequence_batch_shape)
    self.assertEqual(self._temporal_differences.get_shape(),
                     sequence_batch_shape)
    self.assertEqual(self._loss.get_shape(), batch_shape)

  @parameterized.named_parameters(
      ('Length', None, 4),
      ('Batch', 5, None),
      ('BatchAndLength', None, None),)
  def testShapeInferenceDynamic(self, sequence_length, batch_size):
    self._setUp_td_loss(
        sequence_length=sequence_length, batch_size=batch_size, gae_lambda=1.)
    t, b = sequence_length, batch_size

    self.assertEqual(self._discounted_returns.get_shape().as_list(), [t, b])
    self.assertEqual(self._temporal_differences.get_shape().as_list(), [t, b])
    self.assertEqual(self._loss.get_shape().as_list(), [b])

  @parameterized.parameters(
      (1,),
      (0.9,),)
  def testInvalidGradients(self, gae_lambda):
    self._setUp_td_loss(gae_lambda=gae_lambda)
    ins = nest.flatten([self._rewards, self._pcontinues, self._bootstrap_value])
    outs = [None] * len(ins)

    self.assertAllEqual(tf.gradients(self._loss, ins), outs)

  def testGradientsLoss(self):
    self._setUp_td_loss()
    gradient = tf.gradients(self._loss, self._state_values)[0]
    self.assertEqual(gradient.get_shape(), self._state_values.get_shape())


class GeneralizedLambdaReturnsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(0.25, 0.5, 1)
  def testGeneralizedLambdaReturns(self, lambda_):
    """Tests the module-level function generalized_lambda_returns."""

    # Sequence length 2, batch size 1.
    state_values = tf.constant([[0.2], [0.3]], dtype=tf.float32)
    rewards = tf.constant([[0.4], [0.5]], dtype=tf.float32)
    pcontinues = tf.constant([[0.9], [0.8]], dtype=tf.float32)
    bootstrap_value = tf.constant([0.1], dtype=tf.float32)

    discounted_returns = value_ops.generalized_lambda_returns(
        rewards, pcontinues, state_values, bootstrap_value, lambda_)

    # Manually calculate the discounted returns.
    return1 = 0.5 + 0.8 * 0.1
    return0 = 0.4 + 0.9 * (lambda_ * return1 + (1 - lambda_) * 0.3)

    with self.test_session() as sess:
      self.assertAllClose(sess.run(discounted_returns), [[return0], [return1]])


class QVMAXTest(tf.test.TestCase):
  """Tests for the QVMAX loss."""

  def setUp(self):
    super(QVMAXTest, self).setUp()
    self.v_tm1 = tf.constant([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=tf.float32)
    self.pcont_t = tf.constant(
        [0, 0.5, 1, 0, 0.5, 1, 0, 0.5, 1], dtype=tf.float32)
    self.r_t = tf.constant(
        [-1, -1, -1, -1, -1, -1, -1, -1, -1], dtype=tf.float32)
    self.q_t = tf.constant(
        [[0, -1], [-2, 0], [0, -3], [1, 0], [1, 1],
         [0, 1], [1, 2], [2, -2], [2, 2]], dtype=tf.float32)
    self.loss_op, self.extra_ops = value_ops.qv_max(
        self.v_tm1, self.r_t, self.pcont_t, self.q_t)

  def testRankCheck(self):
    v_tm1 = tf.placeholder(tf.float32, [None, None])
    with self.assertRaisesRegexp(
        ValueError, 'QVMAX: Error in rank and/or compatibility check'):
      value_ops.qv_max(v_tm1, self.r_t, self.pcont_t, self.q_t)

  def testCompatibilityCheck(self):
    pcont_t = tf.placeholder(tf.float32, [8])
    with self.assertRaisesRegexp(
        ValueError, 'QVMAX: Error in rank and/or compatibility check'):
      value_ops.qv_max(self.v_tm1, self.r_t, pcont_t, self.q_t)

  def testTarget(self):
    """Tests that target value == r_t + pcont_t * max q_t."""
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(self.extra_ops.target),
          [-1, -1, -1, -1, -0.5, 0, -1, 0, 1])

  def testTDError(self):
    """Tests that td_error == target_value - v_tm1."""
    with self.test_session() as sess:
      self.assertAllClose(
          sess.run(self.extra_ops.td_error),
          [-2, -2, -2, -2, -1.5, -1, -2, -1, 0])

  def testLoss(self):
    """Tests that loss == 0.5 * td_error^2."""
    with self.test_session() as sess:
      # Loss is 0.5 * td_error^2
      self.assertAllClose(
          sess.run(self.loss_op),
          [2, 2, 2, 2, 1.125, 0.5, 2, 0.5, 0])

  def testGradVtm1(self):
    """Tests that the gradients of negative loss are equal to the td_error."""
    with self.test_session() as sess:
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      gradients = tf.gradients([-self.loss_op], [self.v_tm1])
      grad_v_tm1 = sess.run(gradients[0])
      self.assertAllClose(grad_v_tm1, [-2, -2, -2, -2, -1.5, -1, -2, -1, 0])

  def testNoOtherGradients(self):
    """Tests no gradient propagates through things other than v_tm1."""
    # Gradients are only defined for v_tm1, not any other input.
    gradients = tf.gradients([self.loss_op],
                             [self.q_t, self.r_t, self.pcont_t])
    self.assertEqual(gradients, [None] * len(gradients))


if __name__ == '__main__':
  tf.test.main()
