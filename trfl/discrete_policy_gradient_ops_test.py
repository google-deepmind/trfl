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
"""Unit tests for discrete-action Policy Gradient functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf
import tree as nest
from trfl import discrete_policy_gradient_ops as pg_ops


class EntropyCostTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for discrete_policy_entropy op."""

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testEntropy(self, is_multi_actions):
    with self.test_session() as sess:
      # Large values check numerical stability through the logs
      policy_logits_np = np.array([[0, 1], [1, 2], [0, 2], [1, 1], [0, -1000],
                                   [0, 1000]])
      if is_multi_actions:
        num_action_components = 3
        policy_logits_nest = [tf.constant(policy_logits_np, dtype=tf.float32)
                              for _ in xrange(num_action_components)]
      else:
        num_action_components = 1
        policy_logits_nest = tf.constant(policy_logits_np, dtype=tf.float32)

      entropy_op = pg_ops.discrete_policy_entropy_loss(policy_logits_nest)
      entropy = entropy_op.extra.entropy
      self.assertEqual(entropy.get_shape(), tf.TensorShape(6))
      # Get these reference values in Torch with:
      #   c = nnd.EntropyCriterion()
      #   s = nn.LogSoftMax()
      #   result = c:forward(s:forward(logits))
      expected_entropy = num_action_components * np.array(
          [0.58220309, 0.58220309, 0.36533386, 0.69314718, 0, 0])
      self.assertAllClose(sess.run(entropy),
                          expected_entropy,
                          atol=1e-4)

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testGradient(self, is_multi_actions):
    with self.test_session() as sess:
      policy_logits_np = np.array([[0, 1], [1, 2], [0, 2], [1, 1], [0, -1000],
                                   [0, 1000]])
      if is_multi_actions:
        num_action_components = 3
        policy_logits_nest = [tf.constant(policy_logits_np, dtype=tf.float32)
                              for _ in xrange(num_action_components)]
      else:
        num_action_components = 1
        policy_logits_nest = tf.constant(policy_logits_np, dtype=tf.float32)

      entropy_op = pg_ops.discrete_policy_entropy_loss(policy_logits_nest)
      entropy = entropy_op.extra.entropy
      # Counterintuitively, the gradient->0 as policy->deterministic, that's why
      # the gradients for the large logit cases are `[0, 0]`. They should
      # strictly be >0, but they get truncated when we run out of precision.
      expected_gradients = np.array([[0.1966119, -0.1966119],
                                     [0.1966119, -0.1966119],
                                     [0.2099872, -0.2099872],
                                     [0, 0],
                                     [0, 0],
                                     [0, 0]])
      for policy_logits in nest.flatten(policy_logits_nest):
        gradients = tf.gradients(entropy, policy_logits)
        grad_policy_logits = sess.run(gradients[0])
        self.assertAllClose(grad_policy_logits,
                            expected_gradients,
                            atol=1e-4)

  @parameterized.named_parameters(('TwoActions', 2),
                                  ('FiveActions', 5),
                                  ('TenActions', 10),
                                  ('MixedMultiActions', [2, 5, 10]))
  def testNormalisation(self, num_actions):
    with self.test_session() as sess:
      if isinstance(num_actions, list):
        policy_logits = [tf.constant([[1.0] * n], dtype=tf.float32)
                         for n in num_actions]
      else:
        policy_logits = tf.constant(
            [[1.0] * num_actions], dtype=tf.float32)
      entropy_op = pg_ops.discrete_policy_entropy_loss(
          policy_logits, normalise=True)
      self.assertAllClose(sess.run(entropy_op.loss), [-1.0])

  @parameterized.named_parameters(
      ('Fixed', 5, 4, 3, False),
      ('DynamicLength', None, 4, 3, False),
      ('DynamicBatch', 5, None, 3, False),
      ('DynamicBatchAndLength', None, None, 3, False),
      ('DynamicAll', None, None, None, False),
      ('NormFixed', 5, 4, 3, True),
      ('NormDynamicLength', None, 4, 3, True),
      ('NormDynamicBatch', 5, None, 3, True),
      ('NormDynamicBatchAndLength', None, None, 3, True),
      ('NormDynamicAll', None, None, None, True))
  def testShapeInference3D(self, sequence_length, batch_size, num_actions,
                           normalise):
    T, B, A = sequence_length, batch_size, num_actions  # pylint: disable=invalid-name
    op = pg_ops.discrete_policy_entropy_loss(
        policy_logits=tf.placeholder(tf.float32, shape=[T, B, A]),
        normalise=normalise)
    op.extra.entropy.get_shape().assert_is_compatible_with([T, B])
    op.loss.get_shape().assert_is_compatible_with([T, B])

  @parameterized.named_parameters(
      ('Fixed2D', 4, 3, False),
      ('DynamicBatch2D', None, 3, False),
      ('DynamicAll2D', None, None, False),
      ('NormFixed2D', 4, 3, True),
      ('NormDynamicBatch2D', None, 3, True),
      ('NormDynamicAll2D', None, None, True))
  def testShapeInference2D(self, batch_size, num_actions, normalise):
    policy_logits = tf.placeholder(tf.float32, shape=[batch_size, num_actions])
    op = pg_ops.discrete_policy_entropy_loss(policy_logits, normalise=normalise)
    op.extra.entropy.get_shape().assert_is_compatible_with([batch_size])
    op.loss.get_shape().assert_is_compatible_with([batch_size])


@parameterized.named_parameters(('SingleAction', False),
                                ('MultiActions', True))
class DiscretePolicyGradientLossTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for discrete_policy_gradient_loss op."""

  def _setUpLoss(self, is_multi_actions):
    policy_logits_np = np.array([[[0, 1], [0, 1]],
                                 [[1, 1], [0, 100]]])
    actions_np = np.array([[0, 0],
                           [1, 1]], dtype=np.int32)

    if is_multi_actions:
      self._num_action_components = 3
      self._policy_logits_nest = [
          tf.constant(policy_logits_np, dtype=tf.float32)
          for _ in xrange(self._num_action_components)]
      self._actions_nest = [tf.constant(actions_np, dtype=tf.int32)
                            for _ in xrange(self._num_action_components)]
    else:
      self._num_action_components = 1
      self._policy_logits_nest = tf.constant(policy_logits_np, dtype=tf.float32)
      self._actions_nest = tf.constant(actions_np, dtype=tf.int32)

    self._action_values = tf.constant([[0, 1], [2, 1]], dtype=tf.float32)

    self._loss = pg_ops.discrete_policy_gradient_loss(
        self._policy_logits_nest, self._actions_nest, self._action_values)

  def testLoss(self, is_multi_actions):
    self._setUpLoss(is_multi_actions)
    with self.test_session() as sess:
      self.assertEqual(self._loss.get_shape(), tf.TensorShape(2))  # [B]
      self.assertAllClose(
          sess.run(self._loss),
          # computed by summing expected losses from DiscretePolicyGradientTest
          # over the two sequences of length two which I've split the batch
          # into:
          self._num_action_components * np.array([1.386294, 1.313262]))

  def testGradients(self, is_multi_actions):
    self._setUpLoss(is_multi_actions)
    with self.test_session() as sess:
      total_loss = tf.reduce_sum(self._loss)
      gradients = tf.gradients(
          [total_loss], nest.flatten(self._policy_logits_nest))
      grad_policy_logits_nest = sess.run(gradients)
      for grad_policy_logits in grad_policy_logits_nest:
        self.assertAllClose(grad_policy_logits,
                            [[[0, 0], [-0.731, 0.731]],
                             [[1, -1], [0, 0]]], atol=1e-4)
      dead_grads = tf.gradients(
          [total_loss],
          nest.flatten(self._actions_nest) + [self._action_values])
      for grad in dead_grads:
        self.assertIsNone(grad)


class DiscretePolicyGradientTest(tf.test.TestCase):
  """Tests for discrete_policy_gradient op."""

  def testLoss(self):
    with self.test_session() as sess:
      policy_logits = tf.constant([[0, 1], [0, 1], [1, 1], [0, 100]],
                                  dtype=tf.float32)
      action_values = tf.constant([0, 1, 2, 1], dtype=tf.float32)
      actions = tf.constant([0, 0, 1, 1], dtype=tf.int32)
      loss = pg_ops.discrete_policy_gradient(policy_logits, actions,
                                             action_values)
      self.assertEqual(loss.get_shape(), tf.TensorShape(4))

      # Calculate the targets with:
      #     loss = action_value*(-logits[action] + log(sum_a(exp(logits[a]))))
      #  The final case (with large logits), runs out of precision and gets
      #  truncated to 0, but isn't `nan`.
      self.assertAllClose(sess.run(loss), [0, 1.313262, 1.386294, 0])

  def testGradients(self):
    with self.test_session() as sess:
      policy_logits = tf.constant([[0, 1], [0, 1], [1, 1], [0, 100]],
                                  dtype=tf.float32)
      action_values = tf.constant([0, 1, 2, 1], dtype=tf.float32)
      actions = tf.constant([0, 0, 1, 1], dtype=tf.int32)
      loss = pg_ops.discrete_policy_gradient(policy_logits, actions,
                                             action_values)
      total_loss = tf.reduce_sum(loss)
      gradients = tf.gradients([total_loss], [policy_logits])
      grad_policy_logits = sess.run(gradients[0])
      #  The final case (with large logits), runs out of precision and gets
      #  truncated to 0, but isn't `nan`.
      self.assertAllClose(grad_policy_logits,
                          [[0, 0], [-0.731, 0.731], [1, -1], [0, 0]], atol=1e-4)

      self.assertAllEqual(tf.gradients([total_loss], [actions, action_values]),
                          [None, None])

  def testDynamicBatchSize(self):
    policy_logits = tf.placeholder(tf.float32, shape=[None, 3])
    action_values = tf.placeholder(tf.float32, shape=[None])
    actions = tf.placeholder(tf.int32, shape=[None])
    loss = pg_ops.discrete_policy_gradient(policy_logits, actions,
                                           action_values)
    self.assertEqual(loss.get_shape().as_list(), [None])
    gradients = tf.gradients(tf.reduce_sum(loss), [policy_logits])
    self.assertAllEqual(gradients[0].get_shape().as_list(), [None, 3])


class SequenceAdvantageActorCriticLossTest(parameterized.TestCase,
                                           tf.test.TestCase):

  @parameterized.named_parameters(
      ('SingleActionEntropyNormalise', False, True),
      ('SingleActionNoEntropyNormalise', False, False),
      ('MultiActionsEntropyNormalise', True, True),
      ('MultiActionsNoEntropyNormalise', True, False),
  )
  def testLossSequence(self, is_multi_actions, normalise_entropy):
    # A sequence of length 2, batch size 1, 3 possible actions.
    num_actions = 3
    policy_logits = [[[0., 0., 1.]], [[0., 1., 0.]]]
    actions = [[0], [1]]
    baseline_values = [[0.2], [0.3]]
    rewards = [[0.4], [0.5]]
    pcontinues = [[0.9], [0.8]]
    bootstrap_value = [0.1]
    baseline_cost = 0.15
    entropy_cost = 0.25

    if is_multi_actions:
      num_action_components = 3
      policy_logits_nest = [tf.constant(policy_logits, dtype=tf.float32)
                            for _ in xrange(num_action_components)]
      actions_nest = [tf.constant(actions, dtype=tf.int32)
                      for _ in xrange(num_action_components)]
    else:
      num_action_components = 1
      policy_logits_nest = tf.constant(policy_logits, dtype=tf.float32)
      actions_nest = tf.constant(actions, dtype=tf.int32)

    loss, extra = pg_ops.sequence_advantage_actor_critic_loss(
        policy_logits_nest,
        tf.constant(baseline_values, dtype=tf.float32),
        actions_nest,
        tf.constant(rewards, dtype=tf.float32),
        tf.constant(pcontinues, dtype=tf.float32),
        tf.constant(bootstrap_value, dtype=tf.float32),
        baseline_cost=baseline_cost,
        entropy_cost=entropy_cost,
        normalise_entropy=normalise_entropy)

    # Manually calculate the discounted returns.
    return1 = 0.5 + 0.8 * 0.1
    return0 = 0.4 + 0.9 * return1

    with self.test_session() as sess:
      # Discounted returns
      self.assertAllClose(sess.run(extra.discounted_returns),
                          [[return0], [return1]])

      # Advantages
      advantages = [return0 - baseline_values[0][0],
                    return1 - baseline_values[1][0]]
      self.assertAllClose(sess.run(extra.advantages),
                          [[adv] for adv in advantages])

      # Baseline
      expected_baseline_loss = baseline_cost*sum([0.5 * adv**2 for adv in
                                                  advantages])
      self.assertAllClose(
          sess.run(extra.baseline_loss), [expected_baseline_loss])

      # Policy Gradient loss
      #   loss = sum_t(action_value*(-logits[action] +
      #                              log(sum_a(exp(logits[a])))))
      #
      # The below takes advantage of there only being one minibatch dim.
      normalise = lambda logits: np.log(np.exp(logits).sum())
      batch = 0
      expected_policy_gradient_loss = num_action_components * sum([
          advantages[0]*(-(policy_logits[0][batch][actions[0][batch]]) +
                         normalise(policy_logits[0])),
          advantages[1]*(-(policy_logits[1][batch][actions[1][batch]]) +
                         normalise(policy_logits[1])),
      ])
      self.assertAllClose(sess.run(extra.policy_gradient_loss),
                          [expected_policy_gradient_loss])

      # Entropy, calculated as per discrete_policy_entropy tests.
      expected_entropy = num_action_components*0.97533*2
      expected_entropy_loss = -entropy_cost*expected_entropy
      if normalise_entropy:
        expected_entropy_loss /= (num_action_components * np.log(num_actions))
      self.assertAllClose(sess.run(extra.entropy),
                          [expected_entropy], atol=1e-4)
      self.assertAllClose(sess.run(extra.entropy_loss), [expected_entropy_loss],
                          atol=1e-4)

      # Total loss
      expected_loss = [expected_entropy_loss + expected_policy_gradient_loss +
                       expected_baseline_loss]
      self.assertAllClose(sess.run(loss), expected_loss, atol=1e-4)

  @parameterized.named_parameters(('Fixed', 5, 4, 3),
                                  ('DynamicLength', None, 4, 3),
                                  ('DynamicBatch', 5, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None))
  def testShapeInference(self, sequence_length, batch_size, num_actions):
    T, B, A = sequence_length, batch_size, num_actions  # pylint: disable=invalid-name

    loss, extra = pg_ops.sequence_advantage_actor_critic_loss(
        policy_logits=tf.placeholder(tf.float32, shape=[T, B, A]),
        baseline_values=tf.placeholder(tf.float32, shape=[T, B]),
        actions=tf.placeholder(tf.int32, shape=[T, B]),
        rewards=tf.placeholder(tf.float32, shape=[T, B]),
        pcontinues=tf.placeholder(tf.float32, shape=[T, B]),
        bootstrap_value=tf.placeholder(tf.float32, shape=[B]),
        entropy_cost=1)

    extra.discounted_returns.get_shape().assert_is_compatible_with([T, B])
    extra.advantages.get_shape().assert_is_compatible_with([T, B])
    extra.baseline_loss.get_shape().assert_is_compatible_with([B])
    extra.policy_gradient_loss.get_shape().assert_is_compatible_with([B])
    extra.entropy.get_shape().assert_is_compatible_with([B])
    extra.entropy_loss.get_shape().assert_is_compatible_with([B])
    loss.get_shape().assert_is_compatible_with([B])

  @parameterized.named_parameters(('Fixed', 5, 4, 3),
                                  ('DynamicLength', None, 4, 3),
                                  ('DynamicBatch', 5, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None))
  def testShapeInferenceGAE(self, sequence_length, batch_size, num_actions):
    T, B, A = sequence_length, batch_size, num_actions  # pylint: disable=invalid-name

    loss, extra = pg_ops.sequence_advantage_actor_critic_loss(
        policy_logits=tf.placeholder(tf.float32, shape=[T, B, A]),
        baseline_values=tf.placeholder(tf.float32, shape=[T, B]),
        actions=tf.placeholder(tf.int32, shape=[T, B]),
        rewards=tf.placeholder(tf.float32, shape=[T, B]),
        pcontinues=tf.placeholder(tf.float32, shape=[T, B]),
        bootstrap_value=tf.placeholder(tf.float32, shape=[B]),
        lambda_=0.9,
        entropy_cost=1)

    extra.discounted_returns.get_shape().assert_is_compatible_with([T, B])
    extra.advantages.get_shape().assert_is_compatible_with([T, B])
    extra.baseline_loss.get_shape().assert_is_compatible_with([B])
    extra.policy_gradient_loss.get_shape().assert_is_compatible_with([B])
    extra.entropy.get_shape().assert_is_compatible_with([B])
    extra.entropy_loss.get_shape().assert_is_compatible_with([B])
    loss.get_shape().assert_is_compatible_with([B])


class SequenceAdvantageActorCriticLossGradientTest(parameterized.TestCase,
                                                   tf.test.TestCase):

  def setUp(self):
    super(SequenceAdvantageActorCriticLossGradientTest, self).setUp()
    self.num_actions = 3
    self.num_action_components = 5
    policy_logits_np = np.array([[[0., 0., 1.]], [[0., 1., 0.]]])
    self.policy_logits = tf.constant(policy_logits_np, dtype=tf.float32)
    self.multi_policy_logits = [tf.constant(policy_logits_np, dtype=tf.float32)
                                for _ in xrange(self.num_action_components)]
    self.baseline_values = tf.constant([[0.2], [0.3]])
    actions_np = np.array([[0], [1]])
    actions = tf.constant(actions_np)
    multi_actions = [tf.constant(actions_np)
                     for _ in xrange(self.num_action_components)]
    rewards = tf.constant([[0.4], [0.5]])
    pcontinues = tf.constant([[0.9], [0.8]])
    bootstrap_value = tf.constant([0.1])
    baseline_cost = 0.15
    entropy_cost = 0.25

    self.op = pg_ops.sequence_advantage_actor_critic_loss(
        self.policy_logits, self.baseline_values, actions, rewards, pcontinues,
        bootstrap_value, baseline_cost=baseline_cost, entropy_cost=entropy_cost)

    self.multi_op = pg_ops.sequence_advantage_actor_critic_loss(
        self.multi_policy_logits, self.baseline_values, multi_actions, rewards,
        pcontinues, bootstrap_value, baseline_cost=baseline_cost,
        entropy_cost=entropy_cost)

    self.invalid_grad_inputs = [actions, rewards, pcontinues, bootstrap_value]
    self.invalid_grad_outputs = [None]*len(self.invalid_grad_inputs)

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testPolicyGradients(self, is_multi_actions):
    if is_multi_actions:
      loss = self.multi_op.extra.policy_gradient_loss
      policy_logits_nest = self.multi_policy_logits
    else:
      loss = self.op.extra.policy_gradient_loss
      policy_logits_nest = self.policy_logits

    grad_policy_list = [
        tf.gradients(loss, policy_logits)[0] * self.num_actions
        for policy_logits in nest.flatten(policy_logits_nest)]

    for grad_policy in grad_policy_list:
      self.assertEqual(grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))

    self.assertAllEqual(tf.gradients(loss, self.baseline_values), [None])
    self.assertAllEqual(tf.gradients(loss, self.invalid_grad_inputs),
                        self.invalid_grad_outputs)

  def testNonDifferentiableDiscountedReturns(self):
    self.assertAllEqual(tf.gradients(self.op.extra.discounted_returns,
                                     self.invalid_grad_inputs),
                        self.invalid_grad_outputs)

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testEntropyGradients(self, is_multi_actions):
    if is_multi_actions:
      loss = self.multi_op.extra.entropy_loss
      policy_logits_nest = self.multi_policy_logits
    else:
      loss = self.op.extra.entropy_loss
      policy_logits_nest = self.policy_logits

    grad_policy_list = [
        tf.gradients(loss, policy_logits)[0] * self.num_actions
        for policy_logits in nest.flatten(policy_logits_nest)]

    for grad_policy in grad_policy_list:
      self.assertEqual(grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))

    self.assertAllEqual(tf.gradients(loss, self.baseline_values), [None])
    self.assertAllEqual(tf.gradients(loss, self.invalid_grad_inputs),
                        self.invalid_grad_outputs)

  def testBaselineGradients(self):
    loss = self.op.extra.baseline_loss
    grad_baseline = tf.gradients(loss, self.baseline_values)[0]
    self.assertEqual(grad_baseline.get_shape(), tf.TensorShape([2, 1]))
    self.assertAllEqual(tf.gradients(loss, self.policy_logits), [None])
    self.assertAllEqual(tf.gradients(loss, self.invalid_grad_inputs),
                        self.invalid_grad_outputs)

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testTotalLossGradients(self, is_multi_actions):
    with self.test_session() as sess:
      if is_multi_actions:
        total_loss = tf.reduce_sum(self.multi_op.loss)
        policy_logits_nest = self.multi_policy_logits
      else:
        total_loss = tf.reduce_sum(self.op.loss)
        policy_logits_nest = self.policy_logits

      grad_policy_list = [
          tf.gradients(total_loss, policy_logits)[0]
          for policy_logits in nest.flatten(policy_logits_nest)]
      grad_baseline = tf.gradients(total_loss, self.baseline_values)[0]

      for grad_policy in grad_policy_list:
        self.assertEqual(grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))
        # These values were just generated once and hard-coded here to check for
        # regressions. Calculating by hand would be too time-consuming,
        # error-prone and unreadable.
        self.assertAllClose(sess.run(grad_policy),
                            [[[-0.5995, 0.1224, 0.4770]],
                             [[0.0288, -0.0576, 0.0288]]],
                            atol=1e-4)
      self.assertEqual(grad_baseline.get_shape(), tf.TensorShape([2, 1]))
      self.assertAllClose(sess.run(grad_baseline), [[-0.1083], [-0.0420]],
                          atol=1e-4)

      self.assertAllEqual(tf.gradients(total_loss, self.invalid_grad_inputs),
                          self.invalid_grad_outputs)

if __name__ == '__main__':
  tf.test.main()
