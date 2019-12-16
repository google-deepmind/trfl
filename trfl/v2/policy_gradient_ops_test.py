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
import tensorflow.compat.v2 as tf
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
    return tf.cast(self._parameter * actions, tf.float32)

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
  mu = tf.TensorSpec(dtype=tf.float32, shape=(t, b, a))
  sigma = tf.TensorSpec(dtype=tf.float32, shape=(t, b, a))
  mvn_policies = [tfp.distributions.MultivariateNormalDiag]
  mvn_actions = tf.TensorSpec(dtype=tf.float32, shape=(t, b, a))
  mvn_params = [[mu, sigma]]

  if multi_actions:
    # Create a list of n_cat Categorical distributions
    n_cat = 2
    cat_logits = [tf.TensorSpec(dtype=tf.float32, shape=(t, b, c))
                  for _ in xrange(n_cat)]
    cat_policies = [tfp.distributions.Categorical
                    for logits in cat_logits]
    cat_actions = [tf.TensorSpec(dtype=tf.int32, shape=(t, b))
                   for _ in xrange(n_cat)]
    cat_params = [[logits] for logits in cat_logits]

    # Create an exponential distribution
    exp_rate = tf.TensorSpec(dtype=tf.float32, shape=(t, b))
    exp_policies = tfp.distributions.Exponential
    exp_actions = tf.TensorSpec(dtype=tf.float32, shape=(t, b))
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
      num_dims = [tf.cast(tf.reduce_prod(policy.event_shape_tensor()),
                          tf.float32)
                  for policy in policies]
      return 1. / tf.reduce_sum(tf.stack(num_dims))
    entropy_scale_op = scale_op

  return policies, actions, policy_vars, entropy_scale_op


class PolicyGradientTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_gradient op."""

  def _setUp_loss(self, sequence_length, batch_size, action_dim):
    # Use default single MVN action setup
    policy_callables, self._actions, self._policy_vars, _ = _setup_pgops(
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_mvn_actions=action_dim)
    self._action_values = tf.TensorSpec(
        dtype=tf.float32, shape=(sequence_length, batch_size))

    @tf.function(input_signature=[self._policy_vars, self._actions,
                                  self._action_values],
                 autograph=False)
    def _loss_fn(policy_vars, actions, action_values):
      policies = nest.map_structure_up_to(policy_callables,
                                          lambda fn, args: fn(*args),
                                          policy_callables, policy_vars)
      # policy_gradient expects a single distribution, not a nest.
      policies = policies[0]
      policy_vars = policy_vars[0]
      with tf.GradientTape() as tape:
        tape.watch([policy_vars, actions, action_values])
        rval = pg_ops.policy_gradient(policies, actions, action_values)
      return rval, tape.gradient(rval, [policy_vars, actions, action_values])

    self._loss_fn = _loss_fn

  @parameterized.named_parameters(('Fixed', 4, 2, 3),
                                  ('DynamicLength', None, 2, 3),
                                  ('DynamicBatch', 4, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None)
                                 )
  def testLoss(self, sequence_length, batch_size, action_dim):
    self._setUp_loss(sequence_length, batch_size, action_dim)
    expected_loss_shape = [sequence_length, batch_size]
    self.assertEqual((self._loss_fn.get_concrete_function()
                      .output_shapes[0].as_list()), expected_loss_shape)

  @parameterized.named_parameters(('Fixed', 4, 2, 3),
                                  ('DynamicLength', None, 2, 3),
                                  ('DynamicBatch', 4, None, 3),
                                  ('DynamicBatchAndLength', None, None, 3),
                                  ('DynamicAll', None, None, None)
                                 )
  def testGradients(self, sequence_length, batch_size, action_dim):
    self._setUp_loss(sequence_length, batch_size, action_dim)
    self.assertEqual(self._loss_fn.get_concrete_function().output_shapes[1][1:],
                     [tf.TensorShape(None), tf.TensorShape(None)])

  def testRun(self):
    policies, actions, _ = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=1)
    action_values = tf.constant([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = pg_ops.policy_gradient(policies, actions, action_values)
    expected_loss = [[0., -0.5], [2., -1.5], [-6., 2.5]]
    # Expected values are from manual calculation in a Colab.
    self.assertAllEqual(loss, expected_loss)


class PolicyGradientLossTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_gradient op."""

  def _setUp_loss(self, multi_actions, batch_size=2):
    # Use fixed sizes
    sequence_length = 4
    policy_callables, self._actions, self._policy_vars, _ = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        batch_size=batch_size,
        num_mvn_actions=3,
        num_discrete_actions=5)
    self._action_values = tf.TensorSpec(
        dtype=tf.float32, shape=(sequence_length, batch_size))
    all_inputs = [self._actions, self._policy_vars, self._action_values]

    # Tracing fails when autograph is enabled.
    @tf.function(input_signature=all_inputs, autograph=False)
    def _loss_fn(actions, policy_vars, action_values):
      with tf.GradientTape() as tape:
        tape.watch(policy_vars)
        concrete_policies = nest.map_structure_up_to(policy_callables,
                                                     lambda fn, args: fn(*args),
                                                     policy_callables,
                                                     policy_vars)

        loss = pg_ops.policy_gradient_loss(concrete_policies, actions,
                                           action_values, policy_vars)
      grads = tape.gradient(loss, policy_vars)
      return loss, grads
    self._loss_fn = _loss_fn

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testLoss(self, multi_actions):
    batch_size = 2
    self._setUp_loss(multi_actions, batch_size=batch_size)
    self.assertEqual(self._loss_fn.get_concrete_function().output_shapes[0],
                     tf.TensorShape(batch_size))

  @parameterized.named_parameters(('SingleAction', False),
                                  ('MultiActions', True))
  def testGradients(self, multi_actions):
    self._setUp_loss(multi_actions)
    actions, policy_vars, action_values = nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype),
        [self._actions, self._policy_vars, self._action_values])
    _, grads = self._loss_fn(actions, policy_vars, action_values)
    for grad in nest.flatten(grads):
      self.assertIsNotNone(grad)

  def testRun(self):
    policies, actions, _ = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=3)
    action_values = tf.constant([[-0.5, 0.5], [-1.0, 0.5], [1.5, -0.5]])
    loss = pg_ops.policy_gradient_loss(policies, actions, action_values)
    expected_loss = [-24., 3.]
    # Expected values are from manual calculation in a Colab.
    self.assertAllEqual(loss, expected_loss)


class EntropyCostTest(parameterized.TestCase, tf.test.TestCase):
  """Tests for policy_entropy op."""

  def _setUp_entropy(self, multi_actions, normalise,
                     sequence_length, batch_size, num_mvn_actions=3):
    policy_callables, _, self._policy_vars, scale_op = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        normalise_entropy=normalise,
        batch_size=batch_size,
        num_mvn_actions=num_mvn_actions,
        num_discrete_actions=5)

    def _build_policies(policy_vars):
      return nest.map_structure_up_to(policy_callables,
                                      lambda fn, args: fn(*args),
                                      policy_callables, policy_vars)

    @tf.function(input_signature=[self._policy_vars], autograph=False)
    def _loss_fn(policy_vars):
      concrete_policies = _build_policies(policy_vars)
      loss = pg_ops.policy_entropy_loss(concrete_policies, policy_vars,
                                        scale_op=scale_op)
      return loss

    def _grad_fn(policy_vars):
      concrete_policies = _build_policies(policy_vars)
      with tf.GradientTape() as tape:
        tape.watch(policy_vars)
        loss = pg_ops.policy_entropy_loss(concrete_policies, policy_vars,
                                          scale_op=scale_op)
      return tape.gradient(loss, policy_vars)

    self._loss_fn = _loss_fn
    self._grad_fn = _grad_fn

  @parameterized.named_parameters(('SingleAction', False, False),
                                  ('MultiActions', True, False),
                                  ('SingleActionNorm', False, True),
                                  ('MultiActionsNorm', True, True))
  def testEntropyLoss(self, multi_actions, normalise):
    sequence_length = 4
    batch_size = 2
    self._setUp_entropy(
        multi_actions, normalise, sequence_length, batch_size)
    output_shapes = self._loss_fn.get_concrete_function().output_shapes
    expected_shape = [sequence_length, batch_size]
    self.assertEqual(output_shapes.loss, expected_shape)
    self.assertEqual(output_shapes.extra.entropy, expected_shape)

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
    output_shapes = self._loss_fn.get_concrete_function().output_shapes
    entropy_shape = output_shapes.extra.entropy
    loss_shape = output_shapes.loss
    expected_shape = [sequence_length, batch_size]
    self.assertEqual(entropy_shape.as_list(), expected_shape)
    self.assertEqual(loss_shape.as_list(), expected_shape)

  @parameterized.named_parameters(('SingleAction', False, False),
                                  ('MultiActions', True, False),
                                  ('SingleActionNorm', False, True),
                                  ('MultiActionsNorm', True, True))
  def testGradient(self, multi_actions, normalise):
    sequence_length = 4
    batch_size = 2
    self._setUp_entropy(
        multi_actions, normalise, sequence_length, batch_size)
    policy_vars = nest.map_structure(
        lambda spec: tf.ones(spec.shape, spec.dtype),
        self._policy_vars)
    # MVN mu has None gradient
    flat_grads = nest.flatten(self._grad_fn(policy_vars))
    self.assertIsNone(flat_grads[0])
    self.assertNotIn(None, flat_grads[1:])

  def testRun(self):
    policies, _, scale_op = _setup_pgops_mock(
        sequence_length=3, batch_size=2, num_policies=3)
    loss, extra = pg_ops.policy_entropy_loss(policies, scale_op=scale_op)
    expected_entropy = [[0., 14.], [28., 42.], [56., 70.]]
    expected_scaling = 3
    expected_loss = (-expected_scaling * np.array(expected_entropy))
    # Expected values are from manual calculation in a Colab.
    self.assertAllEqual(extra.entropy, expected_entropy)
    self.assertAllEqual(loss, expected_loss)


class SequenceA2CLossTest(parameterized.TestCase, tf.test.TestCase):

  def _setUp_a2c_loss(self,
                      multi_actions=False,
                      normalise_entropy=False,
                      gae_lambda=1,
                      sequence_length=4,
                      batch_size=2,
                      num_mvn_actions=3,
                      num_discrete_actions=5):
    callables, self._actions, self._policy_vars, scale_op = _setup_pgops(
        multi_actions=multi_actions,
        sequence_length=sequence_length,
        normalise_entropy=normalise_entropy,
        batch_size=batch_size,
        num_mvn_actions=num_mvn_actions,
        num_discrete_actions=num_discrete_actions)

    t, b = sequence_length, batch_size
    entropy_cost, baseline_cost = 0.1, 0.2
    self._baseline_values = tf.TensorSpec(dtype=tf.float32, shape=(t, b))
    self._rewards = tf.TensorSpec(dtype=tf.float32, shape=(t, b))
    self._pcontinues = tf.TensorSpec(dtype=tf.float32, shape=(t, b))
    self._bootstrap_value = tf.TensorSpec(dtype=tf.float32, shape=(b,))
    all_inputs = [self._policy_vars, self._baseline_values, self._actions,
                  self._rewards, self._pcontinues, self._bootstrap_value]

    @tf.function(input_signature=all_inputs, autograph=False)
    def _loss_fn(policy_vars, baseline_values, actions, rewards, pcontinues,
                 bootstrap_value):
      policies = nest.map_structure_up_to(callables,
                                          lambda fn, args: fn(*args),
                                          callables, policy_vars)
      return pg_ops.sequence_a2c_loss(
          policies=policies,
          baseline_values=baseline_values,
          actions=actions,
          rewards=rewards,
          pcontinues=pcontinues,
          bootstrap_value=bootstrap_value,
          policy_vars=policy_vars,
          lambda_=gae_lambda,
          entropy_cost=entropy_cost,
          baseline_cost=baseline_cost,
          entropy_scale_op=scale_op)

    # Don't make this a tf.function because it doesn't need to be and just slows
    # things down.
    def _grad_fn(grad_wrt, policy_vars, baseline_values, actions, rewards,
                 pcontinues, bootstrap_value):
      policies = nest.map_structure_up_to(callables,
                                          lambda fn, args: fn(*args),
                                          callables, policy_vars)
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(grad_wrt)
        entropy_cost, baseline_cost = 0.1, 0.2
        loss_outputs = pg_ops.sequence_a2c_loss(
            policies=policies,
            baseline_values=baseline_values,
            actions=actions,
            rewards=rewards,
            pcontinues=pcontinues,
            bootstrap_value=bootstrap_value,
            policy_vars=policy_vars,
            lambda_=gae_lambda,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            entropy_scale_op=scale_op)

      return nest.map_structure(lambda out: tape.gradient(out, grad_wrt),
                                loss_outputs)

    self._loss_fn = _loss_fn
    self._grad_fn = _grad_fn

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
    loss_shape, extra_shapes = (self._loss_fn.get_concrete_function()
                                .output_shapes)

    sequence_batch_shape = tf.TensorShape([sequence_length, batch_size])
    batch_shape = tf.TensorShape(batch_size)
    self.assertEqual(extra_shapes.discounted_returns, sequence_batch_shape)
    self.assertEqual(extra_shapes.advantages, sequence_batch_shape)
    self.assertEqual(extra_shapes.policy_gradient_loss, batch_shape)
    self.assertEqual(extra_shapes.baseline_loss, batch_shape)
    self.assertEqual(extra_shapes.entropy, batch_shape)
    self.assertEqual(extra_shapes.entropy_loss, batch_shape)
    self.assertEqual(loss_shape, batch_shape)

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
    loss_shape, extra_shapes = (self._loss_fn.get_concrete_function()
                                .output_shapes)
    self.assertEqual(
        extra_shapes.discounted_returns.as_list(), [t, b])
    self.assertEqual(extra_shapes.advantages.as_list(), [t, b])
    self.assertEqual(
        extra_shapes.policy_gradient_loss.as_list(), [b])
    self.assertEqual(extra_shapes.entropy.as_list(), [b])
    self.assertEqual(extra_shapes.entropy_loss.as_list(), [b])
    self.assertEqual(loss_shape.as_list(), [b])

  @parameterized.named_parameters(
      ('SingleAction', False, 1),
      ('MultiActions', True, 1),
      ('SingleActionGAE', False, 0.9),
      ('MultiActionsGAE', True, 0.9),
  )
  def testInvalidGradients(self, multi_actions, gae_lambda):
    self._setUp_a2c_loss(multi_actions=multi_actions, gae_lambda=gae_lambda)
    ins = nest.map_structure(lambda spec: tf.ones(spec.shape, spec.dtype),
                             [self._policy_vars, self._baseline_values,
                              self._actions, self._rewards, self._pcontinues,
                              self._bootstrap_value])
    # ins[2:] excludes policy_vars and baseline_values, which do have gradients
    # with respect to some of these quantities.
    outs = self._grad_fn(ins[2:], *ins)
    expected = [None for _ in nest.flatten(ins[2:])]
    self.assertAllEqual(nest.flatten(outs.extra.discounted_returns),
                        expected)
    self.assertAllEqual(nest.flatten(outs.extra.policy_gradient_loss),
                        expected)
    self.assertAllEqual(nest.flatten(outs.extra.entropy_loss), expected)
    self.assertAllEqual(nest.flatten(outs.extra.baseline_loss), expected)
    self.assertAllEqual(nest.flatten(outs.loss), expected)

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True),
  )
  def testGradientsPolicyGradientLoss(self, multi_actions):
    self._setUp_a2c_loss(multi_actions=multi_actions)
    ins = nest.map_structure(lambda spec: tf.ones(spec.shape, spec.dtype),
                             [self._policy_vars, self._baseline_values,
                              self._actions, self._rewards, self._pcontinues,
                              self._bootstrap_value])
    grads = self._grad_fn(ins[0], *ins)
    self.assertNotIn(None, nest.flatten(grads.extra.policy_gradient_loss))

  @parameterized.named_parameters(
      ('SingleActionNoEntropyNorm', False, False),
      ('MultiActionsNoEntropyNorm', True, False),
      ('SingleActionEntropyNorm', False, True),
      ('MultiActionsEntropyNorm', True, True),
  )
  def testGradientsEntropy(self, multi_actions, normalise_entropy):
    self._setUp_a2c_loss(multi_actions=multi_actions,
                         normalise_entropy=normalise_entropy)
    ins = nest.map_structure(lambda spec: tf.ones(spec.shape, spec.dtype),
                             [self._policy_vars, self._baseline_values,
                              self._actions, self._rewards, self._pcontinues,
                              self._bootstrap_value])
    grads = self._grad_fn(ins[:2], *ins)
    # MVN mu has None gradient for entropy
    e_pv_grads = nest.flatten(grads.extra.entropy[0])
    self.assertIsNone(e_pv_grads[0])
    self.assertNotIn(None, e_pv_grads[1:])
    e_bv_grads = nest.flatten(grads.extra.entropy[1])
    self.assertAllEqual(e_bv_grads, [None] * len(e_bv_grads))

  def testGradientsBaselineLoss(self):
    self._setUp_a2c_loss()
    ins = nest.map_structure(lambda spec: tf.ones(spec.shape, spec.dtype),
                             [self._policy_vars, self._baseline_values,
                              self._actions, self._rewards, self._pcontinues,
                              self._bootstrap_value])
    grads = self._grad_fn(ins[:2], *ins)
    self.assertIsNotNone(grads.extra.baseline_loss[1])
    policy_var_grads = nest.flatten(grads.extra.baseline_loss[0])
    self.assertAllEqual(policy_var_grads, [None] * len(policy_var_grads))

  @parameterized.named_parameters(
      ('SingleAction', False),
      ('MultiActions', True),
  )
  def testGradientsTotalLoss(self, multi_actions):
    self._setUp_a2c_loss(multi_actions=multi_actions)
    ins = nest.map_structure(lambda spec: tf.ones(spec.shape, spec.dtype),
                             [self._policy_vars, self._baseline_values,
                              self._actions, self._rewards, self._pcontinues,
                              self._bootstrap_value])
    self.assertNotIn(None, nest.flatten(self._grad_fn(ins[:2], *ins).loss))

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

    # Expected values are from manual calculation in a Colab.
    self.assertAllEqual(
        self._extra.baseline_loss, [3., 170.])
    self.assertAllEqual(
        self._extra.policy_gradient_loss, [12., -348.])
    self.assertAllEqual(self._extra.entropy_loss, [-126., -189.])
    self.assertAllEqual(self._loss, [-111., -367.])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
