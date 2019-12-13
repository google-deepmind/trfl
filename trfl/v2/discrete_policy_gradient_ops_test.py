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
import tensorflow.compat.v2 as tf
import tree as nest
from trfl import discrete_policy_gradient_ops as pg_ops


class EntropyCostTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for discrete_policy_entropy op."""

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True))
  def testEntropy(self, is_multi_actions):
    # Large values check numerical stability through the logs
    policy_logits_np = np.array(
        [[0, 1], [1, 2], [0, 2], [1, 1], [0, -1000], [0, 1000]])
    if is_multi_actions:
      num_action_components = 3
      policy_logits_nest = [
          tf.constant(policy_logits_np, dtype=tf.float32)
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
    exp_entropy = num_action_components * np.array(
        [0.58220309, 0.58220309, 0.36533386, 0.69314718, 0, 0])
    self.assertAllClose(entropy, exp_entropy, atol=1e-4)

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True))
  def testGradient(self, is_multi_actions):
    policy_logits_np = np.array(
        [[0, 1], [1, 2], [0, 2], [1, 1], [0, -1000], [0, 1000]])
    if is_multi_actions:
      num_action_components = 3
      policy_logits_nest = [
          tf.constant(policy_logits_np, dtype=tf.float32)
          for _ in xrange(num_action_components)]
    else:
      num_action_components = 1
      policy_logits_nest = tf.constant(policy_logits_np, dtype=tf.float32)
    policy_logits_flat = nest.flatten(policy_logits_nest)

    # Counterintuitively, the gradient->0 as policy->deterministic, that's why
    # the gradients for the large logit cases are `[0, 0]`. They should
    # strictly be >0, but they get truncated when we run out of precision.
    expected_gradients = np.array(
        [[0.1966119, -0.1966119], [0.1966119, -0.1966119],
         [0.2099872, -0.2099872], [0, 0], [0, 0], [0, 0]])

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(policy_logits_flat)
      entropy = pg_ops.discrete_policy_entropy_loss(
          policy_logits_nest).extra.entropy
    for policy_logits in policy_logits_flat:
      grad = tape.gradient(entropy, policy_logits)
      self.assertAllClose(grad, expected_gradients, atol=1e-4)

  @parameterized.named_parameters(
      ('TwoActions', 2),
      ('FiveActions', 5),
      ('TenActions', 10),
      ('MixedMultiActions', [2, 5, 10]))
  def testNormalisation(self, num_actions):
    if isinstance(num_actions, list):
      policy_logits = [
          tf.constant([[1.0] * n], dtype=tf.float32)
          for n in num_actions]
    else:
      policy_logits = tf.constant(
          [[1.0] * num_actions], dtype=tf.float32)
    loss, _ = pg_ops.discrete_policy_entropy_loss(policy_logits, normalise=True)
    self.assertAllClose(loss, [-1.0])

  @parameterized.named_parameters(
      ('No-Norm', False),
      ('Norm', True))
  def testShapeInference3D(self, normalise):
    T, B, A = 5, 4, 3  # pylint: disable=invalid-name
    op = pg_ops.discrete_policy_entropy_loss(
        policy_logits=tf.ones(dtype=tf.float32, shape=[T, B, A]),
        normalise=normalise)
    op.extra.entropy.get_shape().assert_is_compatible_with([T, B])
    op.loss.get_shape().assert_is_compatible_with([T, B])

  @parameterized.named_parameters(
      ('No-Norm', False),
      ('Norm', True))
  def testShapeInference2D(self, normalise):
    batch_size, num_actions = 4, 3
    policy_logits = tf.ones(dtype=tf.float32, shape=[batch_size, num_actions])
    op = pg_ops.discrete_policy_entropy_loss(policy_logits, normalise=normalise)
    op.extra.entropy.get_shape().assert_is_compatible_with([batch_size])
    op.loss.get_shape().assert_is_compatible_with([batch_size])


@parameterized.named_parameters(
    ('SingleAction', False),
    ('MultiActions', True))
class DiscretePolicyGradientLossTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for discrete_policy_gradient_loss op."""

  def _setUp(self, is_multi_actions):
    policy_logits_np = np.array(
        [[[0, 1], [0, 1]],
         [[1, 1], [0, 100]]])
    actions_np = np.array(
        [[0, 0], [1, 1]], dtype=np.int32)

    if is_multi_actions:
      self._num_action_components = 3
      self._policy_logits_nest = [
          tf.constant(policy_logits_np, dtype=tf.float32)
          for _ in xrange(self._num_action_components)]
      self._actions_nest = [
          tf.constant(actions_np, dtype=tf.int32)
          for _ in xrange(self._num_action_components)]
    else:
      self._num_action_components = 1
      self._policy_logits_nest = tf.constant(policy_logits_np, dtype=tf.float32)
      self._actions_nest = tf.constant(actions_np, dtype=tf.int32)

    self._action_values = tf.constant([[0, 1], [2, 1]], dtype=tf.float32)

  def testLoss(self, is_multi_actions):
    self._setUp(is_multi_actions)

    loss = pg_ops.discrete_policy_gradient_loss(
        self._policy_logits_nest, self._actions_nest, self._action_values)

    # computed by summing expected losses from DiscretePolicyGradientTest
    # over the two sequences of length two which I've split the batch into:
    expected_loss = self._num_action_components * np.array([1.386294, 1.313262])

    self.assertEqual(loss.get_shape(), tf.TensorShape(2))  # [B]
    self.assertAllClose(loss, expected_loss)

  def testGradients(self, is_multi_actions):
    self._setUp(is_multi_actions)

    tensors = nest.flatten(self._policy_logits_nest)
    dead_tensors = nest.flatten(self._actions_nest) + [self._action_values]

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(tensors + dead_tensors)
      loss = pg_ops.discrete_policy_gradient_loss(
          self._policy_logits_nest, self._actions_nest, self._action_values)
      total_loss = tf.reduce_sum(loss)

    expected_grad = [
        [[0, 0], [-0.731, 0.731]],
        [[1, -1], [0, 0]]]

    grad_tensors = tape.gradient(total_loss, tensors)
    for grad in grad_tensors:
      self.assertAllClose(grad, expected_grad, atol=1e-4)

    grad_dead_tensors = tape.gradient(total_loss, dead_tensors)
    for grad in grad_dead_tensors:
      self.assertIsNone(grad)


class DiscretePolicyGradientTest(tf.test.TestCase):
  """Tests for discrete_policy_gradient op."""

  def setUp(self):
    super(DiscretePolicyGradientTest, self).setUp()
    self.policy_logits = tf.constant(
        [[0, 1], [0, 1], [1, 1], [0, 100]], dtype=tf.float32)
    self.action_values = tf.constant([0, 1, 2, 1], dtype=tf.float32)
    self.actions = tf.constant([0, 0, 1, 1], dtype=tf.int32)

  def testLoss(self):
    loss = pg_ops.discrete_policy_gradient(
        self.policy_logits, self.actions, self.action_values)

    # Calculate the targets with:
    #     loss = action_value*(-logits[action] + log(sum_a(exp(logits[a]))))
    #  The final case (with large logits), runs out of precision and gets
    #  truncated to 0, but isn't `nan`.
    expected_loss = [0, 1.313262, 1.386294, 0]

    self.assertEqual(loss.get_shape(), tf.TensorShape(4))
    self.assertAllClose(loss, expected_loss)

  def testGradients(self):
    tensor = self.policy_logits
    dead_tensors = [self.actions, self.action_values]

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([tensor] + dead_tensors)
      loss = pg_ops.discrete_policy_gradient(
          self.policy_logits, self.actions, self.action_values)
      total_loss = tf.reduce_sum(loss)

    #  The final case (with large logits), runs out of precision and gets
    #  truncated to 0, but isn't `nan`.
    expected_grad = [[0, 0], [-0.731, 0.731], [1, -1], [0, 0]]

    grad = tape.gradient(total_loss, tensor)
    self.assertAllClose(grad, expected_grad, atol=1e-4)

    dead_grads = tape.gradient(total_loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))


class SequenceAdvantageActorCriticLossTest(
    parameterized.TestCase, tf.test.TestCase):

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

    # Discounted returns
    self.assertAllClose(extra.discounted_returns, [[return0], [return1]])

    # Advantages
    exp_advantages = [
        [return0 - baseline_values[0][0]],
        [return1 - baseline_values[1][0]]]
    self.assertAllClose(extra.advantages, exp_advantages)

    # Baseline
    exp_baseline_loss = baseline_cost * sum(
        [0.5 * adv[0]**2 for adv in exp_advantages])
    self.assertAllClose(extra.baseline_loss, [exp_baseline_loss])

    # Policy Gradient loss
    #   loss = sum_t(action_value*(-logits[action] +
    #                              log(sum_a(exp(logits[a])))))
    #
    # The below takes advantage of there only being one minibatch dim.
    normalise = lambda logits: np.log(np.exp(logits).sum())
    batch = 0
    exp_policy_gradient_loss = num_action_components * sum([
        exp_advantages[0][0]*(-(policy_logits[0][batch][actions[0][batch]]) +
                              normalise(policy_logits[0])),
        exp_advantages[1][0]*(-(policy_logits[1][batch][actions[1][batch]]) +
                              normalise(policy_logits[1])),])
    self.assertAllClose(
        extra.policy_gradient_loss, [exp_policy_gradient_loss])

    # Entropy, calculated as per discrete_policy_entropy tests.
    exp_entropy = num_action_components*0.97533*2
    exp_entropy_loss = -entropy_cost*exp_entropy
    if normalise_entropy:
      exp_entropy_loss /= (num_action_components * np.log(num_actions))
    self.assertAllClose(
        extra.entropy, [exp_entropy], atol=1e-4)
    self.assertAllClose(
        extra.entropy_loss, [exp_entropy_loss], atol=1e-4)

    # Total loss
    expected_loss = [
        exp_entropy_loss + exp_policy_gradient_loss + exp_baseline_loss]
    self.assertAllClose(loss, expected_loss, atol=1e-4)

  def testShapeInference(self):
    T, B, A = 5, 4, 3  # pylint: disable=invalid-name

    loss, extra = pg_ops.sequence_advantage_actor_critic_loss(
        policy_logits=tf.ones([T, B, A], tf.float32),
        baseline_values=tf.ones([T, B], tf.float32),
        actions=tf.zeros([T, B], tf.int32),
        rewards=tf.zeros([T, B], tf.float32),
        pcontinues=.99 * tf.ones([T, B], tf.float32),
        bootstrap_value=tf.ones([B], tf.float32),
        entropy_cost=1)

    extra.discounted_returns.get_shape().assert_is_compatible_with([T, B])
    extra.advantages.get_shape().assert_is_compatible_with([T, B])
    extra.baseline_loss.get_shape().assert_is_compatible_with([B])
    extra.policy_gradient_loss.get_shape().assert_is_compatible_with([B])
    extra.entropy.get_shape().assert_is_compatible_with([B])
    extra.entropy_loss.get_shape().assert_is_compatible_with([B])
    loss.get_shape().assert_is_compatible_with([B])

  def testShapeInferenceGAE(self):
    T, B, A = 5, 4, 3  # pylint: disable=invalid-name

    loss, extra = pg_ops.sequence_advantage_actor_critic_loss(
        policy_logits=tf.ones([T, B, A], tf.float32),
        baseline_values=tf.ones([T, B], tf.float32),
        actions=tf.zeros([T, B], tf.int32),
        rewards=tf.zeros([T, B], tf.float32),
        pcontinues=.99 * tf.ones([T, B], tf.float32),
        bootstrap_value=tf.ones([B], tf.float32),
        lambda_=0.9,
        entropy_cost=1)

    extra.discounted_returns.get_shape().assert_is_compatible_with([T, B])
    extra.advantages.get_shape().assert_is_compatible_with([T, B])
    extra.baseline_loss.get_shape().assert_is_compatible_with([B])
    extra.policy_gradient_loss.get_shape().assert_is_compatible_with([B])
    extra.entropy.get_shape().assert_is_compatible_with([B])
    extra.entropy_loss.get_shape().assert_is_compatible_with([B])
    loss.get_shape().assert_is_compatible_with([B])


class SequenceAdvantageActorCriticLossGradientTest(
    parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super(SequenceAdvantageActorCriticLossGradientTest, self).setUp()
    self.num_actions = 3
    self.num_action_components = 5
    policy_logits_np = np.array([[[0., 0., 1.]], [[0., 1., 0.]]])
    self.policy_logits = tf.constant(policy_logits_np, dtype=tf.float32)
    self.multi_policy_logits = [
        tf.constant(policy_logits_np, dtype=tf.float32)
        for _ in xrange(self.num_action_components)]
    self.baseline_values = tf.constant([[0.2], [0.3]])
    actions_np = np.array([[0], [1]])
    self.actions = tf.constant(actions_np, dtype=tf.int32)
    self.multi_actions = [
        tf.constant(actions_np, dtype=tf.int32)
        for _ in xrange(self.num_action_components)]
    self.rewards = tf.constant([[0.4], [0.5]])
    self.pcontinues = tf.constant([[0.9], [0.8]])
    self.bootstrap_value = tf.constant([0.1])
    self.baseline_cost = 0.15
    self.entropy_cost = 0.25

  def testNonDifferentiableDiscountedReturns(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value]

    with tf.GradientTape() as tape:
      tape.watch(dead_tensors)
      discounted_returns = pg_ops.sequence_advantage_actor_critic_loss(
          self.policy_logits, self.baseline_values,
          self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.discounted_returns

    dead_grads = tape.gradient(discounted_returns, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  def testPolicyGradientsSingleAction(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
        self.baseline_values]

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([self.policy_logits] + dead_tensors)

      loss = pg_ops.sequence_advantage_actor_critic_loss(
          self.policy_logits, self.baseline_values,
          self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.policy_gradient_loss

    grad = tape.gradient(loss, self.policy_logits) * self.num_actions
    self.assertEqual(grad.get_shape(), tf.TensorShape([2, 1, 3]))

    dead_grads = tape.gradient(loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  def testPolicyGradientsMultiAction(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
        self.baseline_values]

    flat_multi_policy_logits = nest.flatten(self.multi_policy_logits)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(flat_multi_policy_logits + dead_tensors)

      loss = pg_ops.sequence_advantage_actor_critic_loss(
          self.multi_policy_logits, self.baseline_values,
          self.multi_actions, self.rewards, self.pcontinues,
          self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.policy_gradient_loss

    grad_policy_list = [
        tape.gradient(loss, t) * self.num_actions
        for t in flat_multi_policy_logits]
    for grad_policy in grad_policy_list:
      self.assertEqual(grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))

    dead_grads = tape.gradient(loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  def testEntropyGradientsSingleAction(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
        self.baseline_values]

    with tf.GradientTape(persistent=True) as tape:
      tape.watch([self.policy_logits] + dead_tensors)
      loss = pg_ops.sequence_advantage_actor_critic_loss(
          self.policy_logits, self.baseline_values,
          self.actions, self.rewards, self.pcontinues,
          self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.entropy_loss

    grad_policy = tape.gradient(loss, self.policy_logits) * self.num_actions
    self.assertEqual(grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))

    dead_grads = tape.gradient(loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  def testEntropyGradientsMultiAction(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
        self.baseline_values]

    flat_multi_policy_logits = nest.flatten(self.multi_policy_logits)

    with tf.GradientTape(persistent=True) as tape:
      tape.watch(flat_multi_policy_logits + dead_tensors)
      loss = pg_ops.sequence_advantage_actor_critic_loss(
          self.multi_policy_logits, self.baseline_values,
          self.multi_actions, self.rewards, self.pcontinues,
          self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.entropy_loss

    grad_policy_list = [
        tape.gradient(loss, t) * self.num_actions
        for t in flat_multi_policy_logits]
    for grad_policy in grad_policy_list:
      self.assertEqual(
          grad_policy.get_shape(), tf.TensorShape([2, 1, 3]))

    dead_grads = tape.gradient(loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  def testBaselineGradients(self):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
        self.policy_logits]
    with tf.GradientTape(persistent=True) as tape:
      tape.watch([self.baseline_values] + dead_tensors)
      loss = pg_ops.sequence_advantage_actor_critic_loss(
          self.policy_logits, self.baseline_values,
          self.actions, self.rewards, self.pcontinues, self.bootstrap_value,
          baseline_cost=self.baseline_cost,
          entropy_cost=self.entropy_cost).extra.baseline_loss

    grad = tape.gradient(loss, self.baseline_values)
    self.assertEqual(grad.get_shape(), tf.TensorShape([2, 1]))

    dead_grads = tape.gradient(loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True))
  def testTotalLossGradients(self, is_multi_actions):
    dead_tensors = [
        self.actions, self.rewards, self.pcontinues, self.bootstrap_value]

    with tf.GradientTape(persistent=True) as tape:
      if is_multi_actions:
        flat_logits = nest.flatten(self.multi_policy_logits)
        tape.watch(
            [self.baseline_values] + flat_logits + dead_tensors)
        loss = pg_ops.sequence_advantage_actor_critic_loss(
            self.multi_policy_logits, self.baseline_values,
            self.multi_actions, self.rewards, self.pcontinues,
            self.bootstrap_value,
            baseline_cost=self.baseline_cost,
            entropy_cost=self.entropy_cost).loss
        total_loss = tf.reduce_sum(loss)
      else:
        flat_logits = nest.flatten(self.policy_logits)
        tape.watch(
            [self.baseline_values] + flat_logits + dead_tensors)
        loss = pg_ops.sequence_advantage_actor_critic_loss(
            self.policy_logits, self.baseline_values,
            self.actions, self.rewards, self.pcontinues,
            self.bootstrap_value,
            baseline_cost=self.baseline_cost,
            entropy_cost=self.entropy_cost).loss
        total_loss = tf.reduce_sum(loss)

    # These values were just generated once and hard-coded here to check for
    # regressions. Calculating by hand would be too time-consuming,
    # error-prone and unreadable.
    expected_grad_policy = [
        [[-0.5995, 0.1224, 0.4770]],
        [[0.0288, -0.0576, 0.0288]]]
    expected_grad_baselines = [[-0.1083], [-0.0420]]

    grad_policy = [
        tape.gradient(total_loss, t) for t in flat_logits]
    for grad in grad_policy:
      self.assertEqual(grad.get_shape(), tf.TensorShape([2, 1, 3]))
      self.assertAllClose(grad, expected_grad_policy, atol=1e-4)

    grad_baseline = tape.gradient(total_loss, self.baseline_values)
    self.assertEqual(
        grad_baseline.get_shape(), tf.TensorShape([2, 1]))
    self.assertAllClose(
        grad_baseline, expected_grad_baselines, atol=1e-4)

    dead_grads = tape.gradient(total_loss, dead_tensors)
    self.assertAllEqual(dead_grads, [None] * len(dead_tensors))


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
