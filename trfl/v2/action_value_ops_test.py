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
"""Tests for action_value_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
from trfl import action_value_ops as rl


class QLearningTest(tf.test.TestCase):

  def setUp(self):
    super(QLearningTest, self).setUp()
    self.q_tm1 = tf.constant(
        [[1, 1, 0],
         [1, 2, 0]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[0, 1, 0],
         [1, 2, 0]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1], dtype=tf.int32)
    self.pcont_t = tf.constant([0, 1], dtype=tf.float32)
    self.r_t = tf.constant([1, 1], dtype=tf.float32)

    self.output = rl.qlearning(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "QLearning: Error in rank and/or compatibility check"):
      rl.qlearning(q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t)

  def testCompatibilityCheck(self):
    a_tm1 = tf.constant([0, 0, 0], dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError, "QLearning: Error in rank and/or compatibility check"):
      rl.qlearning(self.q_tm1, a_tm1, self.r_t, self.pcont_t, self.q_t)

  def testTarget(self):
    self.assertAllClose(self.output.extra.target, [1, 3])

  def testTDError(self):
    self.assertAllClose(self.output.extra.td_error, [0, 1])

  def testLoss(self):
    self.assertAllClose(self.output.loss, [0, 0.5])  # Loss is 0.5 * td_error^2.

  def testGradQtm1(self):
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[0, 0, 0], [0, 1, 0]])

  def testNoOtherGradients(self):
    tensors = [self.q_t, self.r_t, self.a_tm1, self.pcont_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


class DoubleQLearningTest(tf.test.TestCase):
  # The test is written so that it calculates the same thing as QLearningTest:
  # The selector, despite having different values, select the same actions,
  # whose values are unchanged. (Other values are changed and larger.)

  def setUp(self):
    super(DoubleQLearningTest, self).setUp()
    self.q_tm1 = tf.constant(
        [[1, 1, 0],
         [1, 2, 0]],
        dtype=tf.float32)
    self.q_t_selector = tf.constant(
        [[2, 10, 1],
         [11, 20, 1]],
        dtype=tf.float32)
    self.q_t_value = tf.constant(
        [[99, 1, 98],
         [91, 2, 66]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1], dtype=tf.int32)
    self.pcont_t = tf.constant([0, 1], dtype=tf.float32)
    self.r_t = tf.constant([1, 1], dtype=tf.float32)

    self.output = rl.double_qlearning(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
        self.q_t_value, self.q_t_selector)

  def testRankCheck(self):
    q_t_selector = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError,
        "DoubleQLearning: Error in rank and/or compatibility check"):
      rl.double_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t_value,
          q_t_selector)

  def testCompatibilityCheck(self):
    r_t = tf.constant([0, 1, 0], dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError,
        "DoubleQLearning: Error in rank and/or compatibility check"):
      rl.double_qlearning(
          self.q_tm1, self.a_tm1, r_t, self.pcont_t, self.q_t_value,
          self.q_t_selector)

  def testDoubleQLearningBestAction(self):
    self.assertAllClose(self.output.extra.best_action, [1, 1])

  def testDoubleQLearningTarget(self):
    self.assertAllClose(self.output.extra.target, [1, 3])

  def testDoubleQLearningTDError(self):
    self.assertAllClose(self.output.extra.td_error, [0, 1])

  def testDoubleQLearningLoss(self):
    self.assertAllClose(self.output.loss, [0, 0.5])  # Loss is 0.5 * td_error^2.

  def testDoubleQLearningGradQtm1(self):
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.double_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t_value,
          self.q_t_selector).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[0, 0, 0], [0, 1, 0]])

  def testDoubleQLearningNoOtherGradients(self):
    tensors = [
        self.r_t, self.a_tm1, self.pcont_t, self.q_t_value, self.q_t_selector]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.double_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t_value,
          self.q_t_selector).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


class PersistentQLearningTest(tf.test.TestCase):

  def setUp(self):
    super(PersistentQLearningTest, self).setUp()
    self.action_gap_scale = 0.25

    self.q_tm1 = tf.constant(
        [[1, 2],
         [3, 4],
         [5, 6]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[11, 12],
         [20, 16],
         [-8, -4]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1, 1], dtype=tf.int32)
    self.pcont_t = tf.constant([0, 1, 0.5], dtype=tf.float32)
    self.r_t = tf.constant([3, 2, 7], dtype=tf.float32)

    self.output = rl.persistent_qlearning(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
        self.action_gap_scale)

  def testScalarCheck(self):
    action_gap_scale = 2
    with self.assertRaisesRegexp(
        ValueError,
        r"PersistentQLearning: action_gap_scale has to lie in \[0, 1\]\."):
      rl.persistent_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          action_gap_scale)

  def testCompatibilityCheck(self):
    r_t = tf.constant([0, 1], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError,
        "PersistentQLearning: Error in rank and/or compatibility check"):
      rl.persistent_qlearning(
          self.q_tm1, self.a_tm1, r_t, self.pcont_t, self.q_t,
          self.action_gap_scale)

  def testPersistentQLearningTarget(self):
    self.assertAllClose(self.output.extra.target, [3, 21, 5])

  def testPersistentQLearningTDError(self):
    self.assertAllClose(self.output.extra.td_error, [2, 17, -1])

  def testPersistentQLearningLoss(self):
    # Loss is 0.5 * td_error^2
    self.assertAllClose(self.output.loss, [2, 144.5, 0.5])

  def testPersistentQLearningGradQtm1(self):
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.persistent_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.action_gap_scale).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[2, 0], [0, 17], [0, -1]])

  def testPersistentQLearningNoOtherGradients(self):
    tensors = [self.r_t, self.a_tm1, self.pcont_t, self.q_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.persistent_qlearning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.action_gap_scale).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


class SarsaTest(tf.test.TestCase):
  """Tests for Sarsa learner."""

  def setUp(self):
    super(SarsaTest, self).setUp()
    self.q_tm1 = tf.constant(
        [[1, 1, 0],
         [1, 1, 0]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[0, 1, 0],
         [3, 2, 0]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1], dtype=tf.int32)
    self.a_t = tf.constant([1, 0], dtype=tf.int32)
    self.pcont_t = tf.constant([0, 1], dtype=tf.float32)
    self.r_t = tf.constant([1, 1], dtype=tf.float32)

    self.output = rl.sarsa(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, self.a_t)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "Sarsa: Error in rank and/or compatibility check"):
      rl.sarsa(q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, self.a_t)

  def testCompatibilityCheck(self):
    a_t = tf.constant([0, 1, 0], dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError, "Sarsa: Error in rank and/or compatibility check"):
      rl.sarsa(self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, a_t)

  def testTarget(self):
    """Tests that target value == r_t + pcont_t * q_t[a_t]."""
    self.assertAllClose(self.output.extra.target, [1, 4])

  def testTDError(self):
    """Tests that td_error = target - q_tm1[a_tm1]."""
    self.assertAllClose(self.output.extra.td_error, [0, 3])

  def testLoss(self):
    """Tests that loss == 0.5 * td_error^2."""
    # Loss is 0.5 * td_error^2
    self.assertAllClose(self.output.loss, [0, 4.5])

  def testGradQtm1(self):
    """Tests that the gradients of negative loss are equal to the td_error."""
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.sarsa(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.a_t).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[0, 0, 0], [0, 3, 0]])

  def testNoOtherGradients(self):
    """Tests no gradient propagates through any tensors other than q_tm1."""
    tensors = [self.q_t, self.r_t, self.a_tm1, self.pcont_t, self.a_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.sarsa(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.a_t).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


class SarseTest(tf.test.TestCase):
  """Tests for Sarse learner."""

  def setUp(self):
    super(SarseTest, self).setUp()
    self.q_tm1 = tf.constant(
        [[1, 1, 0.5],
         [1, 1, 3]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[1.5, 1, 2],
         [3, 2, 1]],
        dtype=tf.float32)
    self.probs_a_t = tf.constant(
        [[0.2, 0.5, 0.3],
         [0.3, 0.4, 0.3]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1], dtype=tf.int32)
    self.pcont_t = tf.constant([1, 1], dtype=tf.float32)
    self.r_t = tf.constant([4, 1], dtype=tf.float32)

    self.output = rl.sarse(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
        self.probs_a_t)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "Sarse: Error in rank and/or compatibility check"):
      rl.sarse(
          q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, self.probs_a_t)

  def testCompatibilityCheck(self):
    probs_a_t = tf.constant(
        [[0.2, 0.5], [0.3, 0.4]], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "Sarse: Error in rank and/or compatibility check"):
      rl.sarse(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, probs_a_t)

  def testIncorrectProbsTensor(self):
    probs_a_t = tf.constant(
        [[0.2, 0.5, 0.3], [0.3, 0.5, 0.3]], dtype=tf.float32)
    with self.assertRaisesRegexp(tf.errors.InvalidArgumentError,
                                 "probs_a_t tensor does not sum to 1"):
      rl.sarse(self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
               self.q_t, probs_a_t, debug=True)

  def testTarget(self):
    # target is r_t + sum_a (probs_a_t[a] * q_t[a])
    self.assertAllClose(self.output.extra.target, [5.4, 3])

  def testTDError(self):
    """Tests that td_error = target - q_tm1[a_tm1]."""
    self.assertAllClose(self.output.extra.td_error, [4.4, 2])

  def testLoss(self):
    """Tests that loss == 0.5 * td_error^2."""
    # Loss is 0.5 * td_error^2
    self.assertAllClose(self.output.loss, [9.68, 2])

  def testGradQtm1(self):
    """Tests that the gradients of negative loss are equal to the td_error."""
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.sarse(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.probs_a_t).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[4.4, 0, 0], [0, 2, 0]])

  def testNoOtherGradients(self):
    """Tests no gradient propagates through any tensors other than q_tm1."""
    tensors = [self.q_t, self.r_t, self.a_tm1, self.pcont_t, self.probs_a_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.sarse(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.probs_a_t).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None, None, None, None, None])


class QLambdaTest(tf.test.TestCase):

  def setUp(self):
    super(QLambdaTest, self).setUp()

    # Tensor dimensions below: TxBxA (time, batch id, action).
    self.q_tm1 = tf.constant(
        [[[1.1, 2.1], [2.1, 3.1]],
         [[-1.1, 1.1], [-1.1, 0.1]],
         [[3.1, -3.1], [-2.1, -1.1]]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[[1.2, 2.2], [4.2, 2.2]],
         [[-1.2, 0.2], [1.2, 1.2]],
         [[2.2, -1.2], [-1.2, -2.2]]],
        dtype=tf.float32)
    # Tensor dimensions below: TxB (time, batch id).
    self.a_tm1 = tf.constant(
        [[0, 1], [1, 0], [0, 0]], dtype=tf.int32)
    self.pcont_t = tf.constant(
        [[0.00, 0.88], [0.89, 1.00], [0.85, 0.83]], dtype=tf.float32)
    self.r_t = tf.constant(
        [[-1.3, 1.3], [-1.3, 5.3], [2.3, -3.3]], dtype=tf.float32)
    self.lambda_value = tf.constant(
        [[0.67, 0.68], [0.65, 0.69], [0.66, 0.64]], dtype=tf.float32)

    self.output = rl.qlambda(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
        self.q_t, self.lambda_value)

    # Evaluate target Q-values used for testing.
    # t20 is Target for timestep 2, batch 0
    self.t20 = 2.2 * 0.85 + 2.3
    self.t10 = (self.t20 * 0.65 + 0.2 * (1 - 0.65)) * 0.89 - 1.3
    self.t00 = (self.t10 * 0.67 + 2.2 * (1 - 0.67)) * 0.00 - 1.3
    self.t21 = -1.2 * 0.83 - 3.3
    self.t11 = (self.t21 * 0.69 + 1.2 * (1 - 0.69)) * 1.00 + 5.3
    self.t01 = (self.t11 * 0.68 + 4.2 * (1 - 0.68)) * 0.88 + 1.3

  def testRankCheck(self):
    lambda_ = tf.constant([[[1], [1], [0]]], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "QLambda: Error in rank and/or compatibility check"):
      rl.qlambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t, lambda_)

  def testCompatibilityCheck(self):
    r_t = tf.constant([0, 1], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "QLambda: Error in rank and/or compatibility check"):
      rl.qlambda(
          self.q_tm1, self.a_tm1, r_t, self.pcont_t, self.q_t,
          self.lambda_value)

  def testTarget(self):
    # Please note: the last two values of lambda_ are effectively ignored as
    # there is nothing to mix at the end of the sequence.
    self.assertAllClose(
        self.output.extra.target,
        [[self.t00, self.t01], [self.t10, self.t11], [self.t20, self.t21]])

  def testTDError(self):
    self.assertAllClose(
        self.output.extra.td_error,
        [[self.t00 - 1.1, self.t01 - 3.1], [self.t10 - 1.1, self.t11 + 1.1],
         [self.t20 - 3.1, self.t21 + 2.1]])

  def testLoss(self):
    self.assertAllClose(
        self.output.loss,
        [[0.5 * (self.t00 - 1.1)**2, 0.5 * (self.t01 - 3.1)**2],
         [0.5 * (self.t10 - 1.1)**2, 0.5 * (self.t11 + 1.1)**2],
         [0.5 * (self.t20 - 3.1)**2, 0.5 * (self.t21 + 2.1)**2]])

  def testGradQtm1(self):
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.qlambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.lambda_value).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1,
                        [[[self.t00 - 1.1, 0], [0, self.t01 - 3.1]],
                         [[0, self.t10 - 1.1], [self.t11 + 1.1, 0]],
                         [[self.t20 - 3.1, 0], [self.t21 + 2.1, 0]]])

  def testNoOtherGradients(self):
    tensors = [self.a_tm1, self.r_t, self.pcont_t, self.q_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.qlambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
          self.lambda_value).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None, None, None, None])


class PengsQLambdaTest(tf.test.TestCase):
  # These tests verify that GeneralizedQLambda operates as expected when
  # the lambda_ parameter is a constant. We compare against results
  # calculated by GeneralizedQLambda when lambda_ is a tensor whose entries
  # are all equal to the constant value. (The correct operation of this
  # configuration is tested by GeneralizedQLambdaTest.)

  def setUp(self):
    super(PengsQLambdaTest, self).setUp()
    self.lambda_scalar = 0.5
    self.lambda_vector = tf.constant(
        [[self.lambda_scalar, self.lambda_scalar],
         [self.lambda_scalar, self.lambda_scalar],
         [self.lambda_scalar, self.lambda_scalar]],
        dtype=tf.float32)

    # Tensor dimensions below: TxBxA (time, batch id, action).
    self.q_tm1 = tf.constant(
        [[[1.1, 2.1], [2.1, 3.1]],
         [[-1.1, 1.1], [-1.1, 0.1]],
         [[3.1, -3.1], [-2.1, -1.1]]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[[1.2, 2.2], [4.2, 2.2]],
         [[-1.2, 0.2], [1.2, 1.2]],
         [[2.2, -1.2], [-1.2, -2.2]]],
        dtype=tf.float32)
    # Tensor dimensions below: TxB (time, batch id).
    self.a_tm1 = tf.constant(
        [[0, 1], [1, 0], [0, 0]], dtype=tf.int32)
    self.pcont_t = tf.constant(
        [[0.00, 0.88], [0.89, 1.00], [0.85, 0.83]], dtype=tf.float32)
    self.r_t = tf.constant(
        [[-1.3, 1.3], [-1.3, 5.3], [2.3, -3.3]], dtype=tf.float32)

    # Evaluate trusted values by defining lambda_ as a tensor.
    self.output_reference = rl.qlambda(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.q_t,
        self.lambda_vector)
    # Evaluate values by defining lambda_ as a python number.
    self.output = rl.qlambda(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
        self.q_t, self.lambda_scalar)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "QLambda: Error in rank and/or compatibility check"):
      rl.qlambda(
          q_tm1, self.a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.lambda_scalar)

  def testCompatibilityCheck(self):
    a_tm1 = tf.constant([1, 1, 1, 0], dtype=tf.int32)
    with self.assertRaisesRegexp(
        ValueError, "QLambda: Error in rank and/or compatibility check"):
      rl.qlambda(
          self.q_tm1, a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.lambda_scalar)

  def testTarget(self):
    self.assertAllClose(
        self.output.extra.target, self.output_reference.extra.target)

  def testTDError(self):
    self.assertAllClose(
        self.output.extra.td_error, self.output_reference.extra.td_error)

  def testLoss(self):
    self.assertAllClose(self.output.loss, self.output_reference.loss)

  def testGradQtm1(self):
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(self.q_tm1)
      neg_loss = - rl.qlambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.lambda_scalar).loss
      neg_loss_reference = - rl.qlambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.lambda_scalar).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    grad_q_tm1_reference = tape.gradient(neg_loss_reference, self.q_tm1)
    self.assertAllClose(grad_q_tm1, grad_q_tm1_reference)


class SarsaLambdaTest(tf.test.TestCase):

  def setUp(self):
    super(SarsaLambdaTest, self).setUp()
    self.lambda_value = 0.65
    # Tensor dimensions below: TxBxA (time, batch id, action).
    self.q_tm1 = tf.constant(
        [[[1.1, 2.1], [2.1, 3.1]],
         [[-1.1, 1.1], [-1.1, 0.1]],
         [[3.1, -3.1], [-2.1, -1.1]]],
        dtype=tf.float32)
    self.q_t = tf.constant(
        [[[1.2, 2.2], [4.2, 2.2]],
         [[-1.2, 0.2], [1.2, 1.2]],
         [[2.2, -1.2], [-1.2, -2.2]]],
        dtype=tf.float32)
    # Tensor dimensions below: TxB (time, batch id).
    self.a_tm1 = tf.constant(
        [[0, 1], [1, 0], [0, 0]], dtype=tf.int32)
    self.pcont_t = tf.constant(
        [[0.00, 0.88], [0.89, 1.00], [0.85, 0.83]], dtype=tf.float32)
    self.r_t = tf.constant(
        [[-1.3, 1.3], [-1.3, 5.3], [2.3, -3.3]], dtype=tf.float32)
    self.a_t = tf.constant(
        [[1, 0], [0, 0], [0, 1]], dtype=tf.int32)
    self.output = rl.sarsa_lambda(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
        self.q_t, self.a_t, self.lambda_value)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "SarsaLambda: Error in rank and/or compatibility check"):
      rl.sarsa_lambda(
          q_tm1, self.a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.a_t, self.lambda_value)

  def testCompatibilityCheck(self):
    r_t = tf.constant([0, 1], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "SarsaLambda: Error in rank and/or compatibility check"):
      rl.sarsa_lambda(
          self.q_tm1, self.a_tm1, r_t, self.pcont_t,
          self.q_t, self.a_t, self.lambda_value)

  def testNoOtherGradients(self):
    """Tests no gradient propagates through any tensors other than q_tm1."""
    tensors = [
        self.q_t, self.r_t, self.a_tm1, self.pcont_t, self.a_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.sarsa_lambda(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t,
          self.q_t, self.a_t, self.lambda_value).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


class QVTest(tf.test.TestCase):
  """Tests for QV learner."""

  def setUp(self):
    super(QVTest, self).setUp()
    self.q_tm1 = tf.constant(
        [[1, 1, 0],
         [1, 1, 0]],
        dtype=tf.float32)
    self.a_tm1 = tf.constant([0, 1], dtype=tf.int32)
    self.pcont_t = tf.constant([0, 1], dtype=tf.float32)
    self.r_t = tf.constant([1, 1], dtype=tf.float32)
    self.v_t = tf.constant([1, 3], dtype=tf.float32)

    self.output = rl.qv_learning(
        self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.v_t)

  def testRankCheck(self):
    q_tm1 = tf.constant([1, 1, 0], dtype=tf.float32)
    with self.assertRaisesRegexp(
        ValueError, "QVLearning: Error in rank and/or compatibility check"):
      rl.qv_learning(q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.v_t)

  def testTarget(self):
    """Tests that target value == r_t + pcont_t * q_t[a_t]."""
    self.assertAllClose(self.output.extra.target, [1, 4])

  def testTDError(self):
    """Tests that td_error = target - q_tm1[a_tm1]."""
    self.assertAllClose(self.output.extra.td_error, [0, 3])

  def testLoss(self):
    """Tests that loss == 0.5 * td_error^2."""
    # Loss is 0.5 * td_error^2
    self.assertAllClose(self.output.loss, [0, 4.5])

  def testGradQtm1(self):
    """Tests that the gradients of negative loss are equal to the td_error."""
    with tf.GradientTape() as tape:
      tape.watch(self.q_tm1)
      # Take gradients of the negative loss, so that the tests here check the
      # values propogated during gradient _descent_, rather than _ascent_.
      neg_loss = - rl.qv_learning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.v_t).loss
    grad_q_tm1 = tape.gradient(neg_loss, self.q_tm1)
    self.assertAllClose(grad_q_tm1, [[0, 0, 0], [0, 3, 0]])

  def testNoOtherGradients(self):
    """Tests no gradient propagates through any tensors other than q_tm1."""
    tensors = [self.r_t, self.a_tm1, self.pcont_t, self.v_t]
    with tf.GradientTape() as tape:
      tape.watch(tensors)
      # Gradients are only defined for q_tm1, not any other input.
      # Bellman residual variants could potentially generate a gradient wrt q_t.
      neg_loss = - rl.qv_learning(
          self.q_tm1, self.a_tm1, self.r_t, self.pcont_t, self.v_t).loss
    gradients = tape.gradient(neg_loss, tensors)
    self.assertEqual(gradients, [None] * len(tensors))


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.test.main()
