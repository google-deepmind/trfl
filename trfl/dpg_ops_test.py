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
"""Tests for dpg_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from trfl import dpg_ops


class DpgTest(tf.test.TestCase):
  """Tests for DpgLearning.
  """

  def setUp(self):
    """Sets up test scenario.

    a_tm1_max = s_tm1 * w_s + b_s
    q_tm1_max = a_tm1_max * w + b
    """
    super(DpgTest, self).setUp()
    self.s_tm1 = tf.constant([[0, 1, 0], [1, 1, 2]], dtype=tf.float32)
    self.w_s = tf.Variable(tf.random_normal([3, 2]), dtype=tf.float32)
    self.b_s = tf.Variable(tf.zeros([2]), dtype=tf.float32)
    self.a_tm1_max = tf.matmul(self.s_tm1, self.w_s) + self.b_s
    self.w = tf.Variable(tf.random_normal([2, 1]), dtype=tf.float32)
    self.b = tf.Variable(tf.zeros([1]), dtype=tf.float32)
    self.q_tm1_max = tf.matmul(self.a_tm1_max, self.w) + self.b
    self.loss, self.dpg_extra = dpg_ops.dpg(self.q_tm1_max, self.a_tm1_max)
    self.batch_size = self.a_tm1_max.get_shape()[0]

  def testDpgNoGradient(self):
    """Test case: q_tm1_max does not depend on a_tm1_max => exception raised.
    """
    with self.test_session():
      a_tm1_max = tf.constant([[0, 1, 0], [1, 1, 2]])
      q_tm1_max = tf.constant([[1], [0]])
      self.assertRaises(ValueError, dpg_ops.dpg, q_tm1_max, a_tm1_max)

  def testDpgDqda(self):
    """Tests the gradient qd/qa produced by the DPGLearner is correct."""
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      value_grad = np.transpose(self.w.eval())[0]
      for i in range(int(self.batch_size)):
        self.assertAllClose(self.dpg_extra.dqda.eval()[i], value_grad)

  def testDpgGradient(self):
    """Gradient of loss w.r.t. actor network parameter w_s is correct."""
    with self.test_session() as sess:
      weight_gradient = tf.gradients(self.loss, self.w_s)
      sess.run(tf.global_variables_initializer())
      value_dpg_gradient, value_s_tm1, value_w = sess.run(
          [weight_gradient[0], self.s_tm1, self.w])
      true_grad = self.calculateTrueGradient(value_w, value_s_tm1)
      self.assertAllClose(value_dpg_gradient, true_grad)

  def testDpgNoOtherGradients(self):
    """No gradient of loss w.r.t. parameters other than that of actor network.
    """
    with self.test_session():
      gradients = tf.gradients([self.loss], [self.q_tm1_max, self.w, self.b])
      self.assertListEqual(gradients, [None] * len(gradients))

  def testDpgDqdaClippingError(self):
    self.assertRaises(
        ValueError, dpg_ops.dpg,
        self.q_tm1_max, self.a_tm1_max, dqda_clipping=-10)

  def testDpgGradientClipping(self):
    """Tests the gradient qd/qa are clipped."""
    _, dpg_extra = dpg_ops.dpg(
        self.q_tm1_max, self.a_tm1_max, dqda_clipping=0.01)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      value_grad = np.transpose(self.w.eval())[0]
      for i in range(int(self.batch_size)):
        self.assertAllClose(dpg_extra.dqda.eval()[i],
                            np.clip(value_grad, -0.01, 0.01))
        self.assertTrue(np.greater(np.absolute(value_grad), 0.01).any())

  def testDpgGradientNormClipping(self):
    """Tests the gradient qd/qa are clipped using norm clipping."""
    _, dpg_extra = dpg_ops.dpg(
        self.q_tm1_max, self.a_tm1_max, dqda_clipping=0.01, clip_norm=True)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(int(self.batch_size)):
        self.assertAllClose(np.linalg.norm(dpg_extra.dqda.eval()[i]), 0.01)

  def testLossShape(self):
    self.assertEqual(self.loss.shape.as_list(), [self.batch_size])

  def calculateTrueGradient(self, value_w, value_s_tm1):
    """Calculates the true gradient over the batch.

    sum_k dq/dw_s = sum_k dq/da * da/dw_s
                  = w * sum_k da/dw_s

    Args:
      value_w: numpy.ndarray containing weights of the linear layer.
      value_s_tm1: state representation.

    Returns:
      The true_gradient of the test case.
    """
    dadws = np.zeros((value_w.shape[0],
                      np.product(self.w_s.get_shape().as_list())))
    for i in range(self.batch_size):
      dadws += np.vstack((np.hstack((value_s_tm1[i], np.zeros(3))),
                          np.hstack((np.zeros(3), value_s_tm1[i]))))
    true_grad = np.dot(np.transpose(value_w), dadws)
    true_grad = -np.transpose(np.reshape(
        true_grad, self.w_s.get_shape().as_list()[::-1]))
    return true_grad


if __name__ == '__main__':
  tf.test.main()
