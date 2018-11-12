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
"""TensorFlow ops for continuous-action Policy Gradient algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from six.moves import zip
import tensorflow as tf
from trfl import base_ops
from trfl import value_ops

nest = tf.contrib.framework.nest

PolicyEntropyExtra = collections.namedtuple("policy_entropy_extra", ["entropy"])
SequenceA2CExtra = collections.namedtuple(
    "sequence_a2c_extra", ["entropy", "entropy_loss", "baseline_loss",
                           "policy_gradient_loss", "advantages",
                           "discounted_returns"])


def policy_gradient(policies, actions, action_values, policy_vars=None,
                    name="policy_gradient"):
  """Computes policy gradient losses for a batch of trajectories.

  See `policy_gradient_loss` for more information on expected inputs and usage.

  Args:
    policies: A distribution over a batch supporting a `log_prob` method, e.g.
        an instance of `tfp.distributions.Distribution`. For example, for
        a diagonal gaussian policy:
        `policies = tfp.distributions.MultivariateNormalDiag(mus, sigmas)`
    actions: An action batch Tensor used as the argument for `log_prob`. Has
        shape equal to the batch shape of the policies concatenated with the
        event shape of the policies (which may be scalar, in which case
        concatenation leaves shape just equal to batch shape).
    action_values: A Tensor containing estimates of the values of the `actions`.
        Has shape equal to the batch shape of the policies.
    policy_vars: An optional iterable of Tensors used by `policies`. If provided
        is used in scope checks. For the multivariate normal example above this
        would be `[mus, sigmas]`.
    name: Customises the name_scope for this op.

  Returns:
    loss: Tensor with same shape as `actions` containing the total loss for each
        element in the batch. Differentiable w.r.t the variables in `policies`
        only.
  """
  policy_vars = list(policy_vars) if policy_vars else list()
  with tf.name_scope(values=policy_vars + [actions, action_values], name=name):
    actions = tf.stop_gradient(actions)
    action_values = tf.stop_gradient(action_values)
    log_prob_actions = policies.log_prob(actions)
    # Prevent accidental broadcasting if possible at construction time.
    action_values.get_shape().assert_is_compatible_with(
        log_prob_actions.get_shape())
    return -tf.multiply(log_prob_actions, action_values)


def policy_gradient_loss(policies, actions, action_values, policy_vars=None,
                         name="policy_gradient_loss"):
  """Computes policy gradient losses for a batch of trajectories.

  This wraps `policy_gradient` to accept a possibly nested array of `policies`
  and `actions` in order to allow for multiple action distribution types or
  independent multivariate distributions if not directly available. It also sums
  up losses along the time dimension, and is more restrictive about shapes,
  assuming a [T, B] layout for the `batch_shape` of the policies and a
  concatenate(`[T, B]`, `event_shape` of the policies) shape for the actions.

  Args:
    policies: A (possibly nested structure of) distribution(s) supporting
        `batch_shape` and `event_shape` properties along with a `log_prob`
        method (e.g. an instance of `tfp.distributions.Distribution`),
        with `batch_shape` equal to `[T, B]`.
    actions: A (possibly nested structure of) N-D Tensor(s) with shape
        `[T, B, ...]` where the final dimensions are the `event_shape` of the
        corresponding distribution in the nested structure (the shape can be
        just `[T, B]` if the `event_shape` is scalar).
    action_values: Tensor of shape `[T, B]` containing an estimate of the value
        of the selected `actions`.
    policy_vars: An optional (possibly nested structure of) iterable(s) of
        Tensors used by `policies`. If provided is used in scope checks.
    name: Customises the name_scope for this op.

  Returns:
    loss: Tensor of shape `[B]` containing the total loss for each sequence
    in the batch. Differentiable w.r.t `policy_logits` only.
  """
  actions = nest.flatten(actions)
  if policy_vars:
    policy_vars = nest.flatten_up_to(policies, policy_vars)
  else:
    policy_vars = [list()] * len(actions)
  policies = nest.flatten(policies)

  # Check happens after flatten so that we can be more flexible on nest
  # structures. This is equivalent to asserting that `len(policies) ==
  # len(actions)`, which is sufficient for what we're doing here.
  nest.assert_same_structure(policies, actions)

  for policies_, actions_ in zip(policies, actions):
    policies_.batch_shape.assert_has_rank(2)
    actions_.get_shape().assert_is_compatible_with(
        policies_.batch_shape.concatenate(policies_.event_shape))

  scoped_values = policy_vars + actions + [action_values]
  with tf.name_scope(name, values=scoped_values):
    # Loss for the policy gradient. Doesn't push additional gradients through
    # the action_values.
    policy_gradient_loss_sequence = tf.add_n([
        policy_gradient(policies_, actions_, action_values, pvars)
        for policies_, actions_, pvars in zip(policies, actions, policy_vars)])

    return tf.reduce_sum(
        policy_gradient_loss_sequence, axis=[0],
        name="policy_gradient_loss")


def policy_entropy_loss(policies,
                        policy_vars=None,
                        scale_op=None,
                        name="policy_entropy_loss"):
  """Calculates entropy 'loss' for policies represented by a distributions.

  Given a (possible nested structure of) batch(es) of policies, this
  calculates the total entropy and corrects the sign so that minimizing the
  resulting loss op is equivalent to increasing entropy in the batch.

  This function accepts a nested structure of `policies` in order to allow for
  multiple distribution types or for multiple action dimensions in the case
  where there is no corresponding mutivariate form for available for a given
  univariate distribution. In this case, the loss is `sum_i(H(p_i, p_i))`
  where `p_i` are members of the `policies` nest. It can be shown that this is
  equivalent to calculating the entropy loss on the Cartesian product space
  over all the action dimensions, if the sampled actions are independent.

  The entropy loss is optionally scaled by some function of the policies.
  E.g. for Categorical distributions there exists such a scaling which maps
  the entropy loss into the range `[-1, 0]` in order to make it invariant to
  the size of the action space - specifically one can divide the loss by
  `sum_i(log(A_i))` where `A_i` is the number of categories in the i'th
  Categorical distribution in the `policies` nest).

  Args:
    policies: A (possibly nested structure of) batch distribution(s)
        supporting an `entropy` method that returns an N-D Tensor with shape
        equal to the `batch_shape` of the distribution, e.g. an instance of
        `tfp.distributions.Distribution`.
    policy_vars: An optional (possibly nested structure of) iterable(s) of
        Tensors used by `policies`. If provided is used in scope checks.
    scale_op: An optional op that takes `policies` as its only argument and
        returns a scalar Tensor that is used to scale the entropy loss.
        E.g. for Diag(sigma) Gaussian policies dividing by the number of
        dimensions makes entropy loss invariant to the action space dimension.
    name: Optional, name of this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape `[B1, B2, ...]`.
    * `extra`: a namedtuple with fields:
        * `entropy`: entropy of the policy, shape `[B1, B2, ...]`.
    where [B1, B2, ... ] == policy.batch_shape
  """
  flat_policy_vars = nest.flatten(policy_vars) if policy_vars else list()
  with tf.name_scope(name, values=flat_policy_vars):
    # We want a value that we can minimize along with other losses, and where
    # minimizing means driving the policy towards a uniform distribution over
    # the actions. We thus scale it by negative one so that it can be simply
    # added to other losses.
    scale = tf.constant(-1.0, dtype=tf.float32)
    if scale_op:
      scale *= scale_op(policies)

    policies = nest.flatten(policies)
    entropy = tf.add_n(
        [policy.entropy() for policy in policies], name="entropy")
    loss = tf.multiply(scale, entropy, name="entropy_loss")
    return base_ops.LossOutput(loss, PolicyEntropyExtra(entropy))


def sequence_a2c_loss(policies,
                      baseline_values,
                      actions,
                      rewards,
                      pcontinues,
                      bootstrap_value,
                      policy_vars=None,
                      lambda_=1,
                      entropy_cost=None,
                      baseline_cost=1,
                      entropy_scale_op=None,
                      name="SequenceA2CLoss"):
  """Constructs a TensorFlow graph computing the A2C/GAE loss for sequences.

  This loss jointly learns the policy and the baseline. Therefore, gradients
  for this loss flow through each tensor in `policies` and through each tensor
  in `baseline_values`, but no other input tensors. The policy is learnt with
  the advantage actor-critic loss, plus an optional entropy term. The baseline
  is regressed towards the n-step bootstrapped returns given by the
  reward/pcontinue sequence. The `baseline_cost` parameter scales the
  gradients w.r.t the baseline relative to the policy gradient, i.e.
  d(loss) / d(baseline) = baseline_cost * (n_step_return - baseline)`.

  This function is designed for batches of sequences of data. Tensors are
  assumed to be time major (i.e. the outermost dimension is time, the second
  outermost dimension is the batch dimension). We denote the sequence length in
  the shapes of the arguments with the variable `T`, the batch size with the
  variable `B`, neither of which needs to be known at construction time. Index
  `0` of the time dimension is assumed to be the start of the sequence.

  `rewards` and `pcontinues` are the sequences of data taken directly from the
  environment, possibly modulated by a discount. `baseline_values` are the
  sequences of (typically learnt) estimates of the values of the states
  visited along a batch of trajectories as observed by the agent given the
  sequences of one or more actions sampled from `policies`.

  The sequences in the tensors should be aligned such that an agent in a state
  with value `V` that takes an action `a` transitions into another state
  with value `V'`, receiving reward `r` and pcontinue `p`. Then `V`, `a`, `r`
  and `p` are all at the same index `i` in the corresponding tensors. `V'` is
  at index `i+1`, or in the `bootstrap_value` tensor if `i == T`.

  For n-dimensional action vectors, a multivariate distribution must be used
  for `policies`. In case there is no multivariate version for the desired
  univariate distribution, or in case the `actions` object is a nested
  structure (e.g. for multiple action types), this function also accepts a
  nested structure  of `policies`. In this case, the loss is given by
  `sum_i(loss(p_i, a_i))` where `p_i` are members of the `policies` nest, and
  `a_i` are members of the `actions` nest. We assume that a single baseline is
  used across all action dimensions for each timestep.

  Args:
    policies: A (possibly nested structure of) distribution(s) supporting
        `batch_shape` and `event_shape` properties & `log_prob` and `entropy`
        methods (e.g. an instance of `tfp.distributions.Distribution`),
        with `batch_shape` equal to `[T, B]`. E.g. for a (non-nested) diagonal
        multivariate gaussian with dimension `A` this would be:
        `policies = tfp.distributions.MultivariateNormalDiag(mus, sigmas)`
        where `mus` and `sigmas` have shape `[T, B, A]`.
    baseline_values: 2-D Tensor containing an estimate of the state value with
        shape `[T, B]`.
    actions: A (possibly nested structure of) N-D Tensor(s) with shape
        `[T, B, ...]` where the final dimensions are the `event_shape` of the
        corresponding distribution in the nested structure (the shape can be
        just `[T, B]` if the `event_shape` is scalar).
    rewards: 2-D Tensor with shape `[T, B]`.
    pcontinues: 2-D Tensor with shape `[T, B]`.
    bootstrap_value: 1-D Tensor with shape `[B]`.
    policy_vars: An optional (possibly nested structure of) iterables of
        Tensors used by `policies`. If provided is used in scope checks. For
        the multivariate normal example above this would be `[mus, sigmas]`.
    lambda_: an optional scalar or 2-D Tensor with shape `[T, B]` for
        Generalised Advantage Estimation as per
        https://arxiv.org/abs/1506.02438.
    entropy_cost: optional scalar cost that pushes the policy to have high
        entropy, larger values cause higher entropies.
    baseline_cost: scalar cost that scales the derivatives of the baseline
        relative to the policy gradient.
    entropy_scale_op: An optional op that takes `policies` as its only
        argument and returns a scalar Tensor that is used to scale the entropy
        loss. E.g. for Diag(sigma) Gaussian policies dividing by the number of
        dimensions makes entropy loss invariant to the action space dimension.
        See `policy_entropy_loss` for more info.
    name: Customises the name_scope for this op.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the total loss, shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `entropy`: total loss per sequence, shape `[B]`.
        * `entropy_loss`: scaled entropy loss per sequence, shape `[B]`.
        * `baseline_loss`: scaled baseline loss per sequence, shape `[B]`.
        * `policy_gradient_loss`: policy gradient loss per sequence,
            shape `[B]`.
        * `advantages`: advantange estimates per timestep, shape `[T, B]`.
        * `discounted_returns`: discounted returns per timestep,
            shape `[T, B]`.
  """
  flat_policy_vars = nest.flatten(policy_vars) if policy_vars else list()
  scoped_values = (flat_policy_vars + nest.flatten(actions) +
                   [baseline_values, rewards, pcontinues, bootstrap_value])
  with tf.name_scope(name, values=scoped_values):
    # Loss for the baseline, summed over the time dimension.
    baseline_loss_td, td_lambda = value_ops.td_lambda(
        baseline_values, rewards, pcontinues, bootstrap_value, lambda_)

    # The TD error provides an estimate of the advantages of the actions.
    advantages = td_lambda.temporal_differences
    baseline_loss = tf.multiply(
        tf.convert_to_tensor(baseline_cost, dtype=tf.float32),
        baseline_loss_td,
        name="baseline_loss")

    # Loss for the policy. Doesn't push additional gradients through
    # the advantages.
    pg_loss = policy_gradient_loss(
        policies, actions, advantages, policy_vars,
        name="policy_gradient_loss")

    total_loss = tf.add(pg_loss, baseline_loss, name="total_loss")

    if entropy_cost is not None:
      loss, extra = policy_entropy_loss(policies, policy_vars, entropy_scale_op)
      entropy = tf.reduce_sum(extra.entropy, axis=0, name="entropy")  # [B].
      entropy_loss = tf.multiply(
          tf.convert_to_tensor(entropy_cost, dtype=tf.float32),
          tf.reduce_sum(loss, axis=0),
          name="scaled_entropy_loss")  # [B].
      total_loss = tf.add(total_loss, entropy_loss,
                          name="total_loss_with_entropy")
    else:
      entropy = None
      entropy_loss = None

    extra = SequenceA2CExtra(
        entropy=entropy,
        entropy_loss=entropy_loss,
        baseline_loss=baseline_loss,
        policy_gradient_loss=pg_loss,
        advantages=advantages,
        discounted_returns=td_lambda.discounted_returns)
    return base_ops.LossOutput(total_loss, extra)
