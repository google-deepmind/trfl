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
"""Tests for dist_value_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow as tf
from trfl import dist_value_ops as rl


class CategoricalDistRLTest(tf.test.TestCase):
  """Abstract base class for Distributional RL value ops tests."""

  def setUp(self):
    super(CategoricalDistRLTest, self).setUp()

    # Define both state- and action-value transitions here for the different
    # learning rules tested in the subclasses.

    self.atoms_tm1 = tf.constant([0.5, 1.0, 1.5], dtype=tf.float32)
    self.atoms_t = tf.identity(self.atoms_tm1)

    self.logits_q_tm1 = tf.constant(
        [[[1, 1, 1], [0, 9, 9], [0, 9, 0], [0, 0, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[1, 1, 1], [0, 9, 9], [0, 0, 0], [0, 9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 9], [0, 0, 0], [9, -9, 0]]], dtype=tf.float32)
    self.logits_q_t = tf.constant(
        [[[1, 1, 1], [9, 0, 9], [1, 0, 0], [0, 0, 9]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[1, 1, 1], [9, 0, 9], [0, 0, 9], [1, 0, 0]],
         [[9, 9, 0], [9, 0, 0], [1, 1, 1], [9, -9, 0]],
         [[9, 9, 0], [9, 0, 0], [0, 9, 9], [9, -9, 0]]], dtype=tf.float32)
    # mean Q_t are approximately:
    #  1.0 1.0 0.5 1.5
    #  0.75 0.5 1.0 0.5
    #  1.0 1.0 1.5 0.5
    #  0.75 0.5 1.0 0.5
    #  0.75 0.5 1.25 0.5

    self.logits_v_tm1 = tf.constant(
        [[0, 9, 0],
         [9, 0, 9],
         [0, 9, 0],
         [9, 9, 0],
         [9, 0, 9]], dtype=tf.float32)
    self.logits_v_t = tf.constant(
        [[0, 0, 9],
         [1, 1, 1],
         [0, 0, 9],
         [1, 1, 1],
         [0, 9, 9]], dtype=tf.float32)

    self.a_tm1 = tf.constant([2, 1, 3, 0, 1], dtype=tf.int32)
    self.r_t = tf.constant([0.5, 0., 0.5, 0.8, -0.1], dtype=tf.float32)
    self.pcont_t = tf.constant([0.8, 1., 0.8, 0., 1.], dtype=tf.float32)

  def assertEachInputRankAndCompatibilityChecked(self, nt, inputs,
                                                 invalid_inputs, nt_name):
    """Check class constructor raises exception if an input tensor is invalid.

    Args:
      nt: namedtuple to be tested.
      inputs: list of (valid) inputs to class constructor.
      invalid_inputs: list of invalid alternative inputs. Should be of same
        length as `inputs`, so that each input can be swapped out for a broken
        input individually.
      nt_name: A string specifying the name of the namedtuple.
    """
    for i, alt_input in enumerate(invalid_inputs):
      broken_inputs = list(inputs)
      broken_inputs[i] = alt_input
      with self.assertRaisesRegexp(
          ValueError,
          "{}: Error in rank and/or compatibility check".format(nt_name)):
        nt(*broken_inputs)


class CategoricalDistQLearningTest(CategoricalDistRLTest):

  def setUp(self):
    super(CategoricalDistQLearningTest, self).setUp()

    self.inputs = [self.atoms_tm1, self.logits_q_tm1, self.a_tm1, self.r_t,
                   self.pcont_t, self.atoms_t, self.logits_q_t]
    self.qlearning = rl.categorical_dist_qlearning(*self.inputs)

  def testRankCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, ()) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_qlearning, self.inputs, alt_inputs,
        "CategoricalDistQLearning")

  def testCompatibilityCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, [1]) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_qlearning, self.inputs, alt_inputs,
        "CategoricalDistQLearning")

  def testTarget(self):
    with self.test_session() as sess:
      # Target is projected KL between r_t + pcont_t atoms_t and
      # probabilities corresponding to logits_q_tm1 [ a_tm1 ].
      expected = np.array([[0.0, 0.0, 1.0],
                           [1/3, 1/3, 1/3],
                           [0.0, 0.0, 1.0],
                           [0.4, 0.6, 0.0],
                           [0.1, 0.5, 0.4]])
      self.assertAllClose(
          sess.run(self.qlearning.extra.target), expected, atol=1e-3)

  def testLoss(self):
    with self.test_session() as sess:
      # Loss is CE between logits_q_tm1 [a_tm1] and target.
      expected = np.array([9.0, 3.69, 9.0, 0.69, 5.19])
      self.assertAllClose(sess.run(self.qlearning.loss), expected, atol=1e-2)

  def testGradQtm1(self):
    with self.test_session() as sess:
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      gradients = tf.gradients([-self.qlearning.loss], [self.logits_q_tm1])
      grad_q_tm1 = sess.run(gradients[0])
      # Correct gradient directions (including 0.0 for unused actions at t=tm1).
      expected = np.zeros_like(grad_q_tm1)
      expected[0, 2] = [-1, -1, 1]
      expected[1, 1] = [-1, 1, -1]
      expected[2, 3] = [-1, -1, 1]
      expected[3, 0] = [-1, 1, -1]
      expected[4, 1] = [-1, 1, -1]
      self.assertAllClose(np.sign(grad_q_tm1), expected)

  def testNoOtherGradients(self):
    # Gradients are only defined for logits_q_tm1, not any other input.
    # Bellman residual variants could potentially generate a gradient wrt q_t.
    gradients = tf.gradients([self.qlearning.loss],
                             [self.logits_q_t, self.r_t, self.a_tm1,
                              self.pcont_t, self.atoms_t, self.atoms_tm1])
    self.assertEqual(gradients, [None for _ in gradients])


class CategoricalDistDoubleQLearningTest(CategoricalDistRLTest):

  def setUp(self):
    super(CategoricalDistDoubleQLearningTest, self).setUp()

    self.q_t_selector = tf.constant(
        [[0, 2, 0, 5],
         [0, 1, 2, 1],
         [0, 2, 5, 0],
         [0, 1, 2, 1],
         [1, 2, 3, 1]], dtype=tf.float32)

    self.inputs = [
        self.atoms_tm1, self.logits_q_tm1, self.a_tm1, self.r_t, self.pcont_t,
        self.atoms_t, self.logits_q_t, self.q_t_selector]
    self.qlearning = rl.categorical_dist_double_qlearning(*self.inputs)

  def testRankCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, ()) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_double_qlearning, self.inputs, alt_inputs,
        "CategoricalDistDoubleQLearning")

  def testCompatibilityCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, [1]) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_double_qlearning, self.inputs, alt_inputs,
        "CategoricalDistDoubleQLearning")

  def testTarget(self):
    with self.test_session() as sess:
      # Target is projected KL between r_t + pcont_t atoms_t and
      # probabilities corresponding to logits_q_tm1 [ a_tm1 ].
      expected = np.array([[0.0, 0.0, 1.0],
                           [1/3, 1/3, 1/3],
                           [0.0, 0.0, 1.0],
                           [0.4, 0.6, 0.0],
                           [0.1, 0.5, 0.4]])
      self.assertAllClose(
          sess.run(self.qlearning.extra.target), expected, atol=1e-3)

  def testLoss(self):
    with self.test_session() as sess:
      # Loss is CE between logits_q_tm1 [a_tm1] and target.
      expected = np.array([9.0, 3.69, 9.0, 0.69, 5.19])
      self.assertAllClose(sess.run(self.qlearning.loss), expected, atol=1e-2)

  def testGradQtm1(self):
    with self.test_session() as sess:
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      gradients = tf.gradients([-self.qlearning.loss], [self.logits_q_tm1])
      grad_q_tm1 = sess.run(gradients[0])
      # Correct gradient directions (including 0.0 for unused actions at t=tm1).
      expected = np.zeros_like(grad_q_tm1)
      expected[0, 2] = [-1, -1, 1]
      expected[1, 1] = [-1, 1, -1]
      expected[2, 3] = [-1, -1, 1]
      expected[3, 0] = [-1, 1, -1]
      expected[4, 1] = [-1, 1, -1]
      self.assertAllClose(np.sign(grad_q_tm1), expected)

  def testNoOtherGradients(self):
    # Gradients are only defined for logits_q_tm1, not any other input.
    # Bellman residual variants could potentially generate a gradient wrt q_t.
    gradients = tf.gradients([self.qlearning.loss],
                             [self.logits_q_t, self.r_t, self.a_tm1,
                              self.pcont_t, self.atoms_t, self.atoms_tm1,
                              self.q_t_selector])
    self.assertEqual(gradients, [None for _ in gradients])


class CategoricalDistTDLearningTest(CategoricalDistRLTest):

  def setUp(self):
    super(CategoricalDistTDLearningTest, self).setUp()

    self.inputs = [self.atoms_tm1, self.logits_v_tm1, self.r_t, self.pcont_t,
                   self.atoms_t, self.logits_v_t]
    self.tdlearning = rl.categorical_dist_td_learning(*self.inputs)

  def testRankCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, ()) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_td_learning, self.inputs, alt_inputs,
        "CategoricalDistTDLearning")

  def testCompatibilityCheck(self):
    alt_inputs = [tf.placeholder(tf.float32, [1]) for _ in self.inputs]
    self.assertEachInputRankAndCompatibilityChecked(
        rl.categorical_dist_td_learning, self.inputs, alt_inputs,
        "CategoricalDistTDLearning")

  def testTarget(self):
    with self.test_session() as sess:
      # Target is projected KL between r_t + pcont_t atoms_t and
      # probabilities corresponding to logits_v_tm1.
      expected = np.array([[0.0, 0.0, 1.0],
                           [1/3, 1/3, 1/3],
                           [0.0, 0.0, 1.0],
                           [0.4, 0.6, 0.0],
                           [0.1, 0.5, 0.4]])
      self.assertAllClose(
          sess.run(self.tdlearning.extra.target), expected, atol=1e-3)

  def testLoss(self):
    with self.test_session() as sess:
      # Loss is CE between logits_v_tm1 and target.
      expected = np.array([9.0, 3.69, 9.0, 0.69, 5.19])
      self.assertAllClose(sess.run(self.tdlearning.loss), expected, atol=1e-2)

  def testGradVtm1(self):
    with self.test_session() as sess:
      # Take gradients of the negative loss, so that the tests here check the
      # values propagated during gradient _descent_, rather than _ascent_.
      gradients = tf.gradients([-self.tdlearning.loss], [self.logits_v_tm1])
      grad_v_tm1 = sess.run(gradients[0])
      # Correct gradient directions.
      expected = np.array([[-1, -1, 1],
                           [-1, 1, -1],
                           [-1, -1, 1],
                           [-1, 1, -1],
                           [-1, 1, -1]])
      self.assertAllClose(np.sign(grad_v_tm1), expected)

  def testNoOtherGradients(self):
    # Gradients are only defined for logits_v_tm1, not any other input.
    # Bellman residual variants could potentially generate a gradient wrt v_t.
    gradients = tf.gradients([self.tdlearning.loss],
                             [self.logits_v_t, self.r_t, self.pcont_t,
                              self.atoms_t, self.atoms_tm1])
    self.assertEqual(gradients, [None for _ in gradients])


if __name__ == "__main__":
  tf.test.main()
