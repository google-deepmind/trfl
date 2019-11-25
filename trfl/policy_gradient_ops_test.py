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
"""Tests for policy_gradient_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_probability as tfp
import tree as nest
from trfl import policy_gradient_ops as pg_ops


class MockDistribution(object):
  """A mock univariate distribution with a given batch shape."""

  def __init__(self, batch_shape, parameter):
    self.batch_shape = tf.TensorShape(batch_shape)
    self.event_shape = tf.TensorShape([])
    self._parameter = parameter
    entropy = np.arange(np.prod(batch_shape)).reshape(batch_shape)
    entropy *= parameter * parameter
    self._entropy = tf.constant(entropy, dtype=tf.float32)

  def log_prob(self, actions):
    return tf.to_float(self._parameter * actions)

  def entropy(self):
    return self._entropy


def _setup_pgops_mock(sequence_length=3, batch_size=2, num_policies=3):
  """Setup ops using mock distribution for numerical tests."""
  t, b = sequence_length, batch_size
  policies = [MockDistribution((t, b), i + 1)  for i in xrange(num_policies)]
  actions = [tf.constant(np.arange(t * b).reshape((t, b)))
             for i in xrange(num_policies)]
  if num_policies == 1:
    policies, actions = policies[0], actions[0]
  entropy_scale_op = lambda policies: len(nest.flatten(policies))
  return policies, actions, entropy_scale_op


def _setup_pgops(multi_actions=False,
                 normalise_entropy=False,
                 sequence_length=4,
                 batch_size=2,
                 num_mvn_actions=3,
                 num_discrete_actions=5):
  """Setup polices, actions, policy_vars and (optionally) entropy_scale_op."""
  t = sequence_length
  b = batch_size
  a = num_mvn_actions
  c = num_discrete_actions

  # MVN actions
  mu = tf.placeholder(tf.float32, shape=(t, b, a))
  sigma = tf.placeholder(tf.float32, shape=(t, b, a))
  mvn_policies = tfp.distributions.MultivariateNormalDiag(
      loc=mu, scale_diag=sigma)
  mvn_actions = tf.placeholder(tf.float32, shape=(t, b, a))
  mvn_params = [mu, sigma]

  if multi_actions:
    # Create a list of n_cat Categorical distributions
    n_cat = 2
    cat_logits = [tf.placeholder(tf.float32, shape=(t, b, c))
                  for _ in xrange(n_cat)]
    cat_policies = [tfp.distributions.Categorical(logits=logits)
                    for logits in cat_logits]
    cat_actions = [tf.placeholder(tf.int32, shape=(t, b))
                   for _ in xrange(n_cat)]
    cat_params = [[logits] for logits in cat_logits]

    # Create an exponential distribution
    exp_rate = tf.placeholder(tf.float32, shape=(t, b))
    exp_policies = tfp.distributions.Exponential(rate=exp_rate)
    exp_actions = tf.placeholder(tf.float32, shape=(t, b))
    exp_params = [exp_rate]

    # Nest all policies and nest corresponding actions and parameters
    policies = [mvn_policies, cat_policies, exp_policies]
    actions = [mvn_actions, cat_actions, exp_actions]
    policy_vars = [mvn_params, cat_params, exp_params]
  else:
    # No nested policy structure
    policies = mvn_policies
    actions = mvn_actions
    policy_vars = mvn_params

  entropy_scale_op = None
  if normalise_entropy:
    # Scale op that divides by total action dims
    def scale_op(policies):
      policies = nest.flatten(policies)
      num_dims = [tf.to_float(tf.reduce_prod(policy.event_shape_tensor()))
                  for policy in policies]
      return 1. / tf.reduce_sum(tf.stack(num_dims))
    entropy_scale_op = scale_op

  return policies, actions, policy_vars, entropy_scale_op


class PolicyGradientTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_gradient op."""

  def _setUp_loss(self, sequence_length, batch_size, action_dim):
    # Use default single MVN action setup
    policies, self._actions, self._policy_vars, _ = _setup_pgops(
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_mvn_actions=action_dim)
    self._action_values = tf.placeholder(
        tf.float32, shape=(sequence_length, batch_size))
    self._loss = pg_ops.policy_gradient(
        policies, self._actions, self._action_values)

  @parameterized.named_parameters(('Fixed', 4, 2, 3),
                                  ('DynamicLength', None, 2, 3),
                                  ('DynamicBatch', 4, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None))
  def testLoss(self, sequence_length, batch_size, action_dim):
    self._setUp_loss(sequence_length, batch_size, action_dim)
    expected_loss_shape = [sequence_length, batch_size]
    self.assertEqual(self._loss.get_shape().as_list(), expected_loss_shape)

  @parameterized.named_parameters(('Fixed', 4, 2, 3),
                                  ('DynamicLength', None, 2, 3),
                                  ('DynamicBatch', 4, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None))
  def testGradients(self, sequence_length, batch_size, action_dim):
    self._setUp_loss(sequence_length, batch_size, action_dim)
    total_loss = tf.reduce_sum(self._loss)
    for policy_var in self._policy_vars:
      gradients = tf.gradients(total_loss, policy_var)
      self.assertEqual(gradients[0].get_shape().as_list(),
                       policy_var.get_shape().as_list())
    gradients = tf.gradients([total_loss], [self._actions, self._action_values])
    self.assertEqual(gradients, [None, None])

  def testRun(self):
    policies, actions, _ = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=1)
    action_values = tf.constant([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = pg_ops.policy_gradient(policies, actions, action_values)
    expected_loss = [[0., -0.5], [2., -1.5], [-6., 2.5]]
    with self.test_session() as sess:
      # Expected values are from manual calculation in a Colab.
      self.assertAllEqual(sess.run(loss), expected_loss)


class PolicyGradientLossTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_gradient op."""

  def _setUp_loss(self, multi_actions, batch_size=2):
    # Use fixed sizes
    sequence_length = 4
    policies, self._actions, self._policy_vars, _ = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_mvn_actions=3,
        num_discrete_actions=5)
    self._action_values = tf.placeholder(
        tf.float32, shape=(sequence_length, batch_size))
    self._loss = pg_ops.policy_gradient_loss(
        policies, self._actions, self._action_values, self._policy_vars)

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testLoss(self, multi_actions):
    batch_size = 2
    self._setUp_loss(multi_actions, batch_size=batch_size)
    self.assertEqual(self._loss.get_shape(), tf.TensorShape(batch_size))

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testGradients(self, multi_actions):
    self._setUp_loss(multi_actions)
    total_loss = tf.reduce_sum(self._loss)
    for policy_var in nest.flatten(self._policy_vars):
      gradients = tf.gradients(total_loss, policy_var)
      self.assertEqual(gradients[0].get_shape(), policy_var.get_shape())

  def testRun(self):
    policies, actions, _ = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=3)
    action_values = tf.constant([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = pg_ops.policy_gradient_loss(policies, actions, action_values)
    expected_loss = [-24., 3.]
    with self.test_session() as sess:
      # Expected values are from manual calculation in a Colab.
      self.assertAllEqual(sess.run(loss), expected_loss)


class EntropyCostTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_entropy op."""

  def _setUp_entropy(self, multi_actions, normalise,
                     sequence_length, batch_size, num_mvn_actions=3):
    policies, _, self._policy_vars, scale_op = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        normalise_entropy=normalise,
        batch_size=batch_size,
        num_mvn_actions=num_mvn_actions,
        num_discrete_actions=5)
    self._policy_entropy_loss = pg_ops.policy_entropy_loss(
        policies, self._policy_vars, scale_op=scale_op)

  @parameterized.named_parameters(('SingleAction', False, False),
                                  ('MultiActions', True, False),
                                  ('SingleActionNorm', False, True),
                                  ('MultiActionsNorm', True, True))
  def testEntropyLoss(self, multi_actions, normalise):
    sequence_length = 4
    batch_size = 2
    self._setUp_entropy(
        multi_actions, normalise, sequence_length, batch_size)
    entropy = self._policy_entropy_loss.extra.entropy
    loss = self._policy_entropy_loss.loss
    expected_shape = [sequence_length, batch_size]
    self.assertEqual(entropy.get_shape(), expected_shape)
    self.assertEqual(loss.get_shape(), expected_shape)

  @parameterized.named_parameters(('Length', None, 2, 3, False),
                                  ('Batch', 4, None, 3, False),
                                  ('BatchAndLength', None, None, 3, False),
                                  ('All', None, None, None, False),
                                  ('LengthNorm', None, 2, 3, True),
                                  ('BatchNorm', 4, None, 3, True),
                                  ('BatchAndLengthNorm', None, None, 3, True),
                                  ('AllNorm', None, None, None, True))
  def testEntropyLossMultiActionDynamic(self, sequence_length, batch_size,
                                        action_dim, normalise):
    self._setUp_entropy(
        multi_actions=True,
        normalise=normalise,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_mvn_actions=action_dim)
    entropy = self._policy_entropy_loss.extra.entropy
    loss = self._policy_entropy_loss.loss
    expected_shape = [sequence_length, batch_size]
    self.assertEqual(entropy.get_shape().as_list(), expected_shape)
    self.assertEqual(loss.get_shape().as_list(), expected_shape)

  @parameterized.named_parameters(('SingleAction', False, False),
                                  ('MultiActions', True, False),
                                  ('SingleActionNorm', False, True),
                                  ('MultiActionsNorm', True, True))
  def testGradient(self, multi_actions, normalise):
    sequence_length = 4
    batch_size = 2
    self._setUp_entropy(
        multi_actions, normalise, sequence_length, batch_size)
    loss = self._policy_entropy_loss.loss
    # MVN mu has None gradient
    self.assertIsNone(tf.gradients(loss, nest.flatten(self._policy_vars)[0])[0])
    for policy_var in nest.flatten(self._policy_vars)[1:]:
      gradient = tf.gradients(loss, policy_var)[0]
      self.assertEqual(gradient.get_shape(), policy_var.get_shape())

  def testRun(self):
    policies, _, scale_op = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=3)
    loss, extra = pg_ops.policy_entropy_loss(policies, scale_op=scale_op)
    expected_entropy = [[0., 14.], [28., 42.], [56., 70.]]
    expected_scaling = 3
    expected_loss = (-expected_scaling * np.array(expected_entropy))
    with self.test_session() as sess:
      # Expected values are from manual calculation in a Colab.
      self.assertAllEqual(sess.run(extra.entropy), expected_entropy)
      self.assertAllEqual(sess.run(loss), expected_loss)


class SequenceA2CLossTest(parameterized.TestCase, tf.test.TestCase):

  def _setUp_a2c_loss(self,
                      multi_actions=False,
                      normalise_entropy=False,
                      gae_lambda=1,
                      sequence_length=4,
                      batch_size=2,
                      num_mvn_actions=3,
                      num_discrete_actions=5):
    policies, self._actions, self._policy_vars, entropy_scale_op = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        normalise_entropy=normalise_entropy,
        batch_size=batch_size,
        num_mvn_actions=num_mvn_actions,
        num_discrete_actions=num_discrete_actions)

    t, b = sequence_length, batch_size
    entropy_cost, baseline_cost = 0.1, 0.2
    self._baseline_values = tf.placeholder(tf.float32, shape=(t, b))
    self._rewards = tf.placeholder(tf.float32, shape=(t, b))
    self._pcontinues = tf.placeholder(tf.float32, shape=(t, b))
    self._bootstrap_value = tf.placeholder(tf.float32, shape=(b,))
    self._loss, self._extra = pg_ops.sequence_a2c_loss(
        policies=policies,
        baseline_values=self._baseline_values,
        actions=self._actions,
        rewards=self._rewards,
        pcontinues=self._pcontinues,
        bootstrap_value=self._bootstrap_value,
        policy_vars=self._policy_vars,
        lambda_=gae_lambda,
        entropy_cost=entropy_cost,
        baseline_cost=baseline_cost,
        entropy_scale_op=entropy_scale_op)

  @parameterized.named_parameters(
      ('SingleActionEntropyNorm', False, True, 1),
      ('SingleActionNoEntropyNorm', False, False, 1),
      ('MultiActionsEntropyNorm', True, True, 1),
      ('MultiActionsNoEntropyNorm', True, False, 1),
      ('SingleActionEntropyNormGAE', False, True, 0.9),
      ('SingleActionNoEntropyNormGAE', False, False, 0.9),
      ('MultiActionsEntropyNormGAE', True, True, 0.9),
      ('MultiActionsNoEntropyNormGAE', True, False, 0.9),
  )
  def testShapeInference(self, multi_actions, normalise_entropy, gae_lambda):
    sequence_length = 4
    batch_size = 2
    self._setUp_a2c_loss(multi_actions, normalise_entropy, gae_lambda,
                         sequence_length=sequence_length, batch_size=batch_size)

    sequence_batch_shape = tf.TensorShape([sequence_length, batch_size])
    batch_shape = tf.TensorShape(batch_size)
    self.assertEqual(self._extra.discounted_returns.get_shape(),
                     sequence_batch_shape)
    self.assertEqual(self._extra.advantages.get_shape(), sequence_batch_shape)
    self.assertEqual(self._extra.policy_gradient_loss.get_shape(), batch_shape)
    self.assertEqual(self._extra.baseline_loss.get_shape(), batch_shape)
    self.assertEqual(self._extra.entropy.get_shape(), batch_shape)
    self.assertEqual(self._extra.entropy_loss.get_shape(), batch_shape)
    self.assertEqual(self._loss.get_shape(), batch_shape)

  @parameterized.named_parameters(('Length', None, 4, 3),
                                  ('Batch', 5, None, 3),
                                  ('BatchAndLength', None, None, 3),
                                  ('All', None, None, None))
  def testShapeInferenceSingleActionNoEntropyNormDynamic(
      self, sequence_length, batch_size, num_actions):
    self._setUp_a2c_loss(sequence_length=sequence_length,
                         batch_size=batch_size,
                         num_mvn_actions=num_actions,
                         num_discrete_actions=num_actions,
                         multi_actions=False,
                         normalise_entropy=False,
                         gae_lambda=1.)
    t, b = sequence_length, batch_size

    self.assertEqual(
        self._extra.discounted_returns.get_shape().as_list(), [t, b])
    self.assertEqual(self._extra.advantages.get_shape().as_list(), [t, b])
    self.assertEqual(
        self._extra.policy_gradient_loss.get_shape().as_list(), [b])
    self.assertEqual(self._extra.entropy.get_shape().as_list(), [b])
    self.assertEqual(self._extra.entropy_loss.get_shape().as_list(), [b])
    self.assertEqual(self._loss.get_shape().as_list(), [b])

  @parameterized.named_parameters(
      ('SingleAction', False, 1),
      ('MultiActions', True, 1),
      ('SingleActionGAE', False, 0.9),
      ('MultiActionsGAE', True, 0.9),
  )
  def testInvalidGradients(self, multi_actions, gae_lambda):
    self._setUp_a2c_loss(multi_actions=multi_actions, gae_lambda=gae_lambda)
    ins = nest.flatten(
        [self._actions, self._rewards, self._pcontinues, self._bootstrap_value])
    outs = [None] * len(ins)

    self.assertAllEqual(tf.gradients(
        self._extra.discounted_returns, ins), outs)
    self.assertAllEqual(tf.gradients(
        self._extra.policy_gradient_loss, ins), outs)
    self.assertAllEqual(tf.gradients(self._extra.entropy_loss, ins), outs)
    self.assertAllEqual(tf.gradients(self._extra.baseline_loss, ins), outs)
    self.assertAllEqual(tf.gradients(self._loss, ins), outs)

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True),
  )
  def testGradientsPolicyGradientLoss(self, multi_actions):
    self._setUp_a2c_loss(multi_actions=multi_actions)
    loss = self._extra.policy_gradient_loss

    for policy_var in nest.flatten(self._policy_vars):
      gradient = tf.gradients(loss, policy_var)[0]
      self.assertEqual(gradient.get_shape(), policy_var.get_shape())

    self.assertAllEqual(tf.gradients(loss, self._baseline_values), [None])

  @parameterized.named_parameters(
      ('SingleActionNoEntropyNorm', False, False),
      ('MultiActionsNoEntropyNorm', True, False),
      ('SingleActionEntropyNorm', False, True),
      ('MultiActionsEntropyNorm', True, True),
  )
  def testGradientsEntropy(self, multi_actions, normalise_entropy):
    self._setUp_a2c_loss(multi_actions=multi_actions,
                         normalise_entropy=normalise_entropy)
    loss = self._extra.entropy_loss

    # MVN mu has None gradient for entropy
    self.assertIsNone(tf.gradients(loss, nest.flatten(self._policy_vars)[0])[0])
    for policy_var in nest.flatten(self._policy_vars)[1:]:
      gradient = tf.gradients(loss, policy_var)[0]
      self.assertEqual(gradient.get_shape(), policy_var.get_shape())

    self.assertAllEqual(tf.gradients(loss, self._baseline_values), [None])

  def testGradientsBaselineLoss(self):
    self._setUp_a2c_loss()
    loss = self._extra.baseline_loss

    gradient = tf.gradients(loss, self._baseline_values)[0]
    self.assertEqual(gradient.get_shape(), self._baseline_values.get_shape())

    policy_vars = nest.flatten(self._policy_vars)
    self.assertAllEqual(tf.gradients(loss, policy_vars),
                        [None]*len(policy_vars))

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True),
  )
  def testGradientsTotalLoss(self, multi_actions):
    self._setUp_a2c_loss(multi_actions=multi_actions)
    loss = self._loss

    gradient = tf.gradients(loss, self._baseline_values)[0]
    self.assertEqual(gradient.get_shape(), self._baseline_values.get_shape())

    for policy_var in nest.flatten(self._policy_vars):
      gradient = tf.gradients(loss, policy_var)[0]
      self.assertEqual(gradient.get_shape(), policy_var.get_shape())

  def testRun(self):
    t, b = 3, 2
    policies, actions, entropy_scale_op = _setup_pgops_mock(
        sequence_length=t, batch_size=b, num_policies=3)
    baseline_values = tf.constant(np.arange(-3, 3).reshape((t, b)),
                                  dtype=tf.float32)
    rewards = tf.constant(np.arange(-2, 4).reshape((t, b)), dtype=tf.float32)
    pcontinues = tf.ones(shape=(t, b), dtype=tf.float32)
    bootstrap_value = tf.constant([-2., 4.], dtype=tf.float32)
    self._loss, self._extra = pg_ops.sequence_a2c_loss(
        policies=policies,
        baseline_values=baseline_values,
        actions=actions,
        rewards=rewards,
        pcontinues=pcontinues,
        bootstrap_value=bootstrap_value,
        entropy_cost=0.5,
        baseline_cost=2.,
        entropy_scale_op=entropy_scale_op)

    with self.test_session() as sess:
      # Expected values are from manual calculation in a Colab.
      self.assertAllEqual(
          sess.run(self._extra.baseline_loss), [3., 170.])
      self.assertAllEqual(
          sess.run(self._extra.policy_gradient_loss), [12., -348.])
      self.assertAllEqual(sess.run(self._extra.entropy_loss), [-126., -189.])
      self.assertAllEqual(sess.run(self._loss), [-111., -367.])


if __name__ == '__main__':
  tf.test.main()
