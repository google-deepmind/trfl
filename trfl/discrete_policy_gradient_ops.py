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
"""TensorFlow ops for discrete-action Policy Gradient functions."""

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

DiscretePolicyEntropyExtra = collections.namedtuple(
    "discrete_policy_entropy_extra", ["entropy"])
SequenceAdvantageActorCriticExtra = collections.namedtuple(
    "sequence_advantage_actor_critic_extra",
    ["entropy", "entropy_loss", "baseline_loss", "policy_gradient_loss",
     "advantages", "discounted_returns"])


def discrete_policy_entropy_loss(policy_logits,
                                 normalise=False,
                                 name="discrete_policy_entropy_loss"):
  """Computes the entropy 'loss' for a batch of policy logits.

  Given a batch of policy logits, calculates the entropy and corrects the sign
  so that minimizing the resulting loss op is equivalent to increasing entropy
  in the batch. This loss is optionally normalised to the range `[-1, 0]` by
  dividing by the log number of actions. This makes it more invariant to the
  size of the action space.

  This function accepts a nested array of `policy_logits` in order
  to allow for multiple discrete actions. In this case, the loss is given by
  `-sum_i(H(p_i))` where `p_i` are members of the `policy_logits` nest and
  H is the Shannon entropy.

  Args:
    policy_logits: A (possibly nested structure of) (N+1)-D Tensor(s) with
        shape `[..., A]`,  representing the log-probabilities of a set of
        Categorical distributions, where `...` represents at least one
        dimension (e.g., batch, sequence), and `A` is the number of discrete
        actions (which need not be identical across all tensors).
        Does not need to be centered.
    normalise: If True, divide the loss by the `sum_i(log(A_i))` where `A_i`
        is the number of actions for the i'th tensor in the `policy_logits`
        nest. Default is False.
    name: Optional, name of this op.

  Returns:
    A namedtuple with fields:

    * `loss`: Entropy 'loss', shape `[B]`.
    * `extra`: a namedtuple with fields:
        * `entropy`: Entropy of the policy, shape `[B]`.
  """
  policy_logits = nest.flatten(policy_logits)

  with tf.name_scope(name, values=policy_logits):
    entropy = tf.add_n([
        tf.reduce_sum(
            -tf.nn.softmax(scalar_policy_logits)
            * tf.nn.log_softmax(scalar_policy_logits), axis=-1)
        for scalar_policy_logits in policy_logits], name="entropy")
    # We want a value that we can minimize along with other losses, and where
    # minimizing means driving the policy towards a uniform distribution over
    # the actions. We thus scale it by negative one so that it can be simply
    # added to other losses.
    scale = tf.constant(-1.0, dtype=tf.float32)
    if normalise:
      num_actions = [tf.to_float(tf.shape(scalar_policy_logits)[-1])
                     for scalar_policy_logits in policy_logits]
      scale /= tf.reduce_sum(tf.log(tf.stack(num_actions)))
    loss = tf.multiply(scale, entropy, name="entropy_loss")

  return base_ops.LossOutput(loss, DiscretePolicyEntropyExtra(entropy))


def sequence_advantage_actor_critic_loss(
    policy_logits, baseline_values, actions, rewards,
    pcontinues, bootstrap_value, lambda_=1, entropy_cost=None,
    baseline_cost=1, normalise_entropy=False,
    name="SequenceAdvantageActorCriticLoss"):
  """Calculates the loss for an A2C update along a batch of trajectories.

  Technically A2C is the special case where lambda=1; for general lambda
  this is the loss for Generalized Advantage Estimation (GAE), modulo chunking
  behaviour if passing chunks of episodes (see `generalized_lambda_returns` for
  more detail).

  Note: This function takes policy _logits_ as input, not the log-policy like
  Reinforce does.

  This loss jointly learns the policy and the baseline. Therefore, gradients
  for this loss flow through each tensor in `policy_logits` and
  `baseline_values`, but no other input tensors. The policy is learnt with the
  advantage actor-critic loss, plus an optional entropy term. The baseline is
  regressed towards the n-step bootstrapped returns given by the
  reward/pcontinue sequence.  The `baseline_cost` parameter scales the
  gradients w.r.t the baseline relative to the policy gradient. i.e:
  `d(loss) / d(baseline) = baseline_cost * (n_step_return - baseline)`.

  `rewards` and `pcontinues` are the sequences of data taken directly from the
  environment, possibly modulated by a discount. `baseline_values` are the
  sequences of (typically learnt) estimates of the values of the states
  visited along a batch of trajectories as observed by the agent given the
  sequences of one or more actions sampled from the `policy_logits`.

  The sequences in the tensors should be aligned such that an agent in a state
  with value `V` that takes an action `a` transitions into another state
  with value `V'`, receiving reward `r` and pcontinue `p`. Then `V`, `a`, `r`
  and `p` are all at the same index `i` in the corresponding tensors. `V'` is
  at index `i+1`, or in the `bootstrap_value` tensor if `i == T`.

  This function accepts a nested array of `policy_logits` and `actions` in order
  to allow for multidimensional discrete action spaces. In this case, the loss
  is given by `sum_i(loss(p_i, a_i))` where `p_i` are members of the
  `policy_logits` nest, and `a_i` are members of the `actions` nest.
  We assume that a single baseline is used across all action dimensions for
  each timestep.

  Args:
    policy_logits: A (possibly nested structure of) 3-D Tensor(s) with shape
        `[T, B, num_actions]` and possibly different dimension `num_actions`.
    baseline_values: 2-D Tensor containing an estimate of state values `[T, B]`.
    actions: A (possibly nested structure of) 2-D Tensor(s) with shape
        `[T, B]` and integer type.
    rewards: 2-D Tensor with shape `[T, B]`.
    pcontinues: 2-D Tensor with shape `[T, B]`.
    bootstrap_value: 1-D Tensor with shape `[B]`.
    lambda_: an optional scalar or 2-D Tensor with shape `[T, B]` for
        Generalised Advantage Estimation as per
        https://arxiv.org/abs/1506.02438.
    entropy_cost: optional scalar cost that pushes the policy to have high
        entropy, larger values cause higher entropies.
    baseline_cost: scalar cost that scales the derivatives of the baseline
        relative to the policy gradient.
    normalise_entropy: if True, the entropy loss is normalised to the range
        `[-1, 0]` by dividing by the log number of actions. This makes it more
        invariant to the size of the action space. Default is False.
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
  scoped_values = (nest.flatten(policy_logits) + nest.flatten(actions) +
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
    policy_gradient_loss = discrete_policy_gradient_loss(
        policy_logits, actions, advantages, name="policy_gradient_loss")

    total_loss = tf.add(policy_gradient_loss, baseline_loss, name="total_loss")

    if entropy_cost is not None:
      entropy_loss_op, policy_entropy = discrete_policy_entropy_loss(
          policy_logits, normalise=normalise_entropy)  # [T,B].
      entropy = tf.reduce_sum(
          policy_entropy.entropy, axis=0, name="entropy")  # [B].
      entropy_loss = tf.multiply(
          tf.convert_to_tensor(entropy_cost, dtype=tf.float32),
          tf.reduce_sum(entropy_loss_op, axis=0),
          name="scaled_entropy_loss")  # [B].
      total_loss = tf.add(total_loss, entropy_loss,
                          name="total_loss_with_entropy")
    else:
      entropy = None
      entropy_loss = None

    extra = SequenceAdvantageActorCriticExtra(
        entropy=entropy, entropy_loss=entropy_loss,
        baseline_loss=baseline_loss,
        policy_gradient_loss=policy_gradient_loss,
        advantages=advantages,
        discounted_returns=td_lambda.discounted_returns)

    return base_ops.LossOutput(total_loss, extra)


def discrete_policy_gradient(policy_logits, actions, action_values,
                             name="discrete_policy_gradient"):
  """Computes a batch of discrete-action policy gradient losses.

  See notes by Silver et al here:
  http://www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching_files/pg.pdf

  From slide 41, denoting by `policy` the probability distribution with
  log-probabilities `policy_logit`:
  ```
  *   `action` should have been sampled according to `policy`.
  *   `action_value` can be any estimate of `Q^{policy}(s, a)`, potentially
      minus a baseline that doesn't depend on the action. This admits
      many possible algorithms:
      * `v_t` (Monte-Carlo return for time t) : REINFORCE
      * `Q^w(s, a)` : Q Actor-Critic
      * `v_t - V(s)` : Monte-Carlo Advantage Actor-Critic
      * `A^{GAE(gamma, lambda)}` : Generalized Avantage Actor Critic
      * + many more.
  ```

  Gradients for this op are only defined with respect to the `policy_logits`,
  not `actions` or `action_values`.

  This op supports multiple batch dimensions. The first N >= 1 dimensions of
  each input/output tensor index into independent values. All tensors must
  having matching sizes for each batch dimension.

  Args:
    policy_logits: (N+1)-D Tensor of shape
        `[batch_size_1, ..., batch_size_N, num_actions]` containing uncentered
        log-probabilities.
    actions: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]` and integer
        type, containing indices for the selected actions.
    action_values: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]`
        containing an estimate of the value of the selected `actions`.
    name: Customises the name_scope for this op.

  Returns:
    loss: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]` containing the
        loss. Differentiable w.r.t `policy_logits` only.

  Raises:
    ValueError: If the batch dimensions of `policy_logits` and `action_values`
        do not match.
  """
  with tf.name_scope(name, values=[policy_logits, actions, action_values]):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=actions, logits=policy_logits)
    action_values = tf.stop_gradient(action_values)
    # Prevent accidental broadcasting if possible at construction time.
    action_values.get_shape().assert_is_compatible_with(
        cross_entropy.get_shape())
    return tf.multiply(cross_entropy, action_values)


def discrete_policy_gradient_loss(policy_logits, actions, action_values,
                                  name="discrete_policy_gradient_loss"):
  """Computes discrete policy gradient losses for a batch of trajectories.

  This wraps `discrete_policy_gradient` to accept a possibly nested array of
  `policy_logits` and `actions` in order to allow for multiple discrete actions.
  It also sums up losses along the time dimension, and is more restrictive about
  shapes, assuming a [T, B] layout.

  Args:
    policy_logits: A (possibly nested structure of) Tensor(s) of shape
        `[T, B, num_actions]` containing uncentered log-probabilities.
    actions: A (possibly nested structure of) Tensor(s) of shape
        `[T, B]` and integer type, containing indices for the selected actions.
    action_values: Tensor of shape `[T, B]`
        containing an estimate of the value of the selected `actions`, see
        `discrete_policy_gradient`.
    name: Customises the name_scope for this op.

  Returns:
    loss: Tensor of shape `[B]` containing the total loss for each sequence
    in the batch. Differentiable w.r.t `policy_logits` only.
  """
  policy_logits = nest.flatten(policy_logits)
  actions = nest.flatten(actions)

  # Check happens after flatten so that we can be more flexible on
  # nest structures. This is equivalent to asserting that
  # `len(policy_logits) == len(actions)`, which is sufficient for what we're
  # doing here. In particular, it means that we can allow one argument to be
  # a tensor, while the other one to be a single-element tensor iterable.
  nest.assert_same_structure(policy_logits, actions)
  for scalar_policy_logits in policy_logits:
    scalar_policy_logits.get_shape().assert_has_rank(3)
  for scalar_actions in actions:
    scalar_actions.get_shape().assert_has_rank(2)

  scoped_values = policy_logits + actions + [action_values]
  with tf.name_scope(name, values=scoped_values):
    # Loss for the policy gradient. Doesn't push additional gradients through
    # the action_values.
    policy_gradient_loss_sequence = tf.add_n([
        discrete_policy_gradient(
            scalar_policy_logits, scalar_actions, action_values)
        for scalar_policy_logits, scalar_actions
        in zip(policy_logits, actions)])

    return tf.reduce_sum(
        policy_gradient_loss_sequence, axis=[0],
        name="policy_gradient_loss")
