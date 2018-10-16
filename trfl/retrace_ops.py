# coding=utf8
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
"""TensorFlow ops for the Retrace algorithm.

These Retrace ops implement an action-value learning rule for minibatches of
multi-step sequences. Retrace supports off-policy learning and has
well-analysed theoretical guarantees.

The ops in this file support only discrete action spaces.  Actions must be
indices in the range `[0, K)`, where `K` is the number of distinct actions
available to the agent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
from six.moves import zip
import tensorflow as tf
from trfl import base_ops
from trfl import indexing_ops
from trfl import sequence_ops

RetraceCoreExtra = collections.namedtuple(
    'retrace_core_extra', ['retrace_weights', 'target'])


def retrace(lambda_,
            qs,
            targnet_qs,
            actions,
            rewards,
            pcontinues,
            target_policy_probs,
            behaviour_policy_probs,
            stop_targnet_gradients=True,
            name=None):
  """Retrace algorithm loss calculation op.

  Given a minibatch of temporally-contiguous sequences of Q values, policy
  probabilities, and various other typical RL algorithm inputs, this
  Op creates a subgraph that computes a loss according to the
  Retrace multi-step off-policy value learning algorithm. This Op supports the
  use of target networks, but does not require them.

  For more details of Retrace, refer to
  [the arXiv paper](http://arxiv.org/abs/1606.02647).

  In argument descriptions, `T` counts the number of transitions over which
  the Retrace loss is computed, and `B` is the minibatch size. Note that all
  tensor arguments list a first-dimension (time dimension) size of T+1;
  this is because in order to compute the loss over T timesteps, the
  algorithm must be aware of the values of many of its inputs at timesteps
  before and after each transition.

  All tensor arguments are indexed first by transition, with specific
  details of this indexing in the argument descriptions.

  Args:
    lambda_: Positive scalar value or 0-D `Tensor` controlling the degree to
      which future timesteps contribute to the loss computed at each
      transition.
    qs: 3-D tensor holding per-action Q-values for the states encountered
      just before taking the transitions that correspond to each major index.
      Since these values are the predicted values we wish to update (in other
      words, the values we intend to change as we learn), in a target network
      setting, these nearly always come from the "non-target" network, which
      we usually call the "learning network".
      Shape is `[(T+1), B, num_actions]`.
    targnet_qs: Like `qs`, but in the target network setting, these values
      should be computed by the target network. We use these values to
      compute multi-step error values for timesteps that follow the first
      timesteps in each sequence and sequence fragment we consider.
      Shape is `[(T+1), B, num_actions]`.
    actions: 2-D tensor holding the indices of actions executed during the
      transition that corresponds to each major index.
      Shape is `[(T+1), B]`.
    rewards: 2-D tensor holding rewards received during the transition
      that corresponds to each major index.
      Shape is `[(T+1), B]`.
    pcontinues: 2-D tensor holding pcontinue values received during the
      transition that corresponds to each major index.
      Shape is `[(T+1), B]`.
    target_policy_probs: 3-D tensor holding per-action policy probabilities
      for the states encountered just before taking the transitions that
      correspond to each major index, according to the target policy (i.e.
      the policy we wish to learn). These probabilities usually derive from
      the learning net.
      Shape is `[(T+1), B, num_actions]`.
    behaviour_policy_probs: 2-D tensor holding the *behaviour* policy's
      probabilities of having taken actions `action` during the transitions
      that correspond to each major index. These probabilities derive from
      whatever policy you used to generate the data.
      Shape is `[(T+1), B]`.
    stop_targnet_gradients: `bool` that enables a sensible default way of
      handling gradients through the Retrace op (essentially, gradients
      are not permitted to involve the `targnet_qs` inputs). Can be disabled
      if you require a different arrangement, but you'll probably want to
      block some gradients somewhere.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: Tensor containing the batch of losses, shape `[B]`.
    * `extra`: None
  """
  all_args = [
      lambda_, qs, targnet_qs, actions, rewards, pcontinues,
      target_policy_probs, behaviour_policy_probs
  ]
  with tf.name_scope(name, 'Retrace', values=all_args):
    # Mainly to simplify testing:
    (lambda_, qs, targnet_qs, actions, rewards, pcontinues, target_policy_probs,
     behaviour_policy_probs) = (
         tf.convert_to_tensor(arg) for arg in all_args)

    # Require correct tensor ranks---as long as we have shape information
    # available to check. If there isn't any, we print a warning.
    def check_rank(tensors, ranks):
      for i, (tensor, rank) in enumerate(zip(tensors, ranks)):
        if tensor.get_shape():
          base_ops.assert_rank_and_shape_compatibility([tensor], rank)
        else:
          tf.logging.error(
              'Tensor "%s", which was offered as Retrace parameter %d, has '
              'no rank at construction time, so Retrace can\'t verify that '
              'it has the necessary rank of %d', tensor.name, i + 1, rank)

    check_rank([
        lambda_, qs, targnet_qs, actions, rewards, pcontinues,
        target_policy_probs, behaviour_policy_probs
    ], [0, 3, 3, 2, 2, 2, 3, 2])

    # Deduce the shapes of the arguments we'll create for retrace_core.
    qs_shape = tf.shape(qs)
    timesteps = qs_shape[0]  # Batch size is qs_shape[1].

    # Deduce the time indices for the arguments we'll create for retrace_core.
    timestep_indices_tm1 = tf.range(0, timesteps - 1)
    timestep_indices_t = tf.range(1, timesteps)

    # Construct arguments for retrace_core and call.
    q_tm1 = tf.gather(qs, timestep_indices_tm1)
    a_tm1 = tf.gather(actions, timestep_indices_tm1)

    r_t = tf.gather(rewards, timestep_indices_tm1)
    pcont_t = tf.gather(pcontinues, timestep_indices_tm1)

    target_policy_t = tf.gather(target_policy_probs, timestep_indices_t)
    behaviour_policy_t = tf.gather(behaviour_policy_probs, timestep_indices_t)
    targnet_q_t = tf.gather(targnet_qs, timestep_indices_t)
    a_t = tf.gather(actions, timestep_indices_t)

    core = retrace_core(lambda_, q_tm1, a_tm1, r_t, pcont_t, target_policy_t,
                        behaviour_policy_t, targnet_q_t, a_t,
                        stop_targnet_gradients)

    return base_ops.LossOutput(core.loss, None)


def _general_off_policy_corrected_multistep_target(r_t,
                                                   pcont_t,
                                                   target_policy_t,
                                                   c_t,
                                                   q_t,
                                                   a_t,
                                                   back_prop=False,
                                                   name=None):
  """Evaluates targets for various off-policy value correction based algorithms.

  `target_policy_t` is the policy that this function aims to evaluate. New
  action-value estimates (target values `T`) must be expressible in this
  recurrent form:
  ```none
  T(x_{t-1}, a_{t-1}) = r_t + γ[ 𝔼_π Q(x_t, .) - c_t Q(x_t, a_t) +
                                                 c_t T(x_t, a_t) ]
  ```
  `T(x_t, a_t)` is an estimate of expected discounted future returns based
  on the current Q value estimates `Q(x_t, a_t)` and rewards `r_t`. The
  evaluated target values can be used as supervised targets for learning the Q
  function itself or as returns for various policy gradient algorithms.
  `Q==T` if convergence is reached. As the formula is recurrent, it will
  evaluate multistep returns for non-zero importance weights `c_t`.

  In the usual moving and target network setup `q_t` should be calculated by
  the target network while the `target_policy_t` may be evaluated by either of
  the networks. If `target_policy_t` is evaluated by the current moving network
  the algorithm implemented will have a similar flavour as double DQN.

  Depending on the choice of c_t, the algorithm can implement:
  ```none
  Importance Sampling             c_t = π(x_t, a_t) / μ(x_t, a_t),
  Harutyunyan's et al. Q(lambda)  c_t = λ,
  Precup's et al. Tree-Backup     c_t = π(x_t, a_t),
  Munos' et al. Retrace           c_t = λ min(1, π(x_t, a_t) / μ(x_t, a_t)).
  ```
  Please refer to page 3 for more details:
  https://arxiv.org/pdf/1606.02647v1.pdf

  Args:
    r_t: 2-D tensor holding rewards received during the transition
      that corresponds to each major index.
      Shape is `[T, B]`.
    pcont_t: 2-D tensor holding pcontinue values received during the
      transition that corresponds to each major index.
      Shape is `[T, B]`.
    target_policy_t:  3-D tensor holding per-action policy probabilities for
      the states encountered just AFTER the transitions that correspond to
      each major index, according to the target policy (i.e. the policy we
      wish to learn). These usually derive from the learning net.
      Shape is `[T, B, num_actions]`.
    c_t: 2-D tensor holding importance weights; see discussion above.
      Shape is `[T, B]`.
    q_t: 3-D tensor holding per-action Q-values for the states
      encountered just AFTER taking the transitions that correspond to each
      major index. Shape is `[T, B, num_actions]`.
    a_t: 2-D tensor holding the indices of actions executed during the
      transition AFTER the transition that corresponds to each major index.
      Shape is `[T, B]`.
    back_prop: whether to backpropagate gradients through time.
    name: name of the op.

  Returns:
    Tensor of shape `[T, B, num_actions]` containing Q values.
  """
  # Formula (4) in https://arxiv.org/pdf/1606.02647v1.pdf can be expressed
  # in a recursive form where T is a new target value:
  # T(x_{t-1}, a_{t-1}) = r_t + γ[ 𝔼_π Q(x_t, .) - c_t Q(x_t, a_t) +
  #                                                c_t T(x_t, a_t) ]
  # This recurrent form allows us to express Retrace by using
  # `scan_discounted_sum`.
  # Define:
  #   T_tm1   = T(x_{t-1}, a_{t-1})
  #   T_t     = T(x_t, a_t)
  #   exp_q_t = 𝔼_π Q(x_{t+1},.)
  #   qa_t    = Q(x_t, a_t)
  # Hence:
  #   T_tm1   = (r_t + γ * exp_q_t - c_t * qa_t) + γ * c_t * T_t
  # Define:
  #   current = r_t + γ * (exp_q_t - c_t * qa_t)
  # Thus:
  #   T_tm1 = scan_discounted_sum(current, γ * c_t, reverse=True)
  args = [r_t, pcont_t, target_policy_t, c_t, q_t, a_t]
  with tf.name_scope(
      name, 'general_returns_based_off_policy_target', values=args):
    exp_q_t = tf.reduce_sum(target_policy_t * q_t, axis=2)
    qa_t = indexing_ops.batched_index(q_t, a_t)
    current = r_t + pcont_t * (exp_q_t - c_t * qa_t)
    initial_value = qa_t[-1]
    return sequence_ops.scan_discounted_sum(
        current,
        pcont_t * c_t,
        initial_value,
        reverse=True,
        back_prop=back_prop)


def _retrace_weights(pi_probs, mu_probs):
  """Evaluates importance weights for the Retrace algorithm.

  Args:
    pi_probs: taken action probabilities according to target policy.
      Shape is `[T, B]`.
    mu_probs: taken action probabilities according to behaviour policy.
      Shape is `[T, B]`.

  Returns:
    Tensor of shape `[T, B]` containing importance weights.
  """
  # tf.minimum seems to handle potential NaNs when pi_probs[i]=mu_probs[i]=0
  return tf.minimum(1.0, pi_probs / mu_probs)


def retrace_core(lambda_,
                 q_tm1,
                 a_tm1,
                 r_t,
                 pcont_t,
                 target_policy_t,
                 behaviour_policy_t,
                 targnet_q_t,
                 a_t,
                 stop_targnet_gradients=True,
                 name=None):
  """Retrace algorithm core loss calculation op.

  Given a minibatch of temporally-contiguous sequences of Q values, policy
  probabilities, and various other typical RL algorithm inputs, this
  Op creates a subgraph that computes a loss according to the
  Retrace multi-step off-policy value learning algorithm. This Op supports the
  use of target networks, but does not require them.

  This function is the "core" Retrace op only because its arguments are less
  user-friendly and more implementation-convenient. For a more user-friendly
  operator, consider using `retrace`. For more details of Retrace, refer to
  [the arXiv paper](http://arxiv.org/abs/1606.02647).

  Construct the "core" retrace loss subgraph for a batch of sequences.

  Note that two pairs of arguments (one holding target network values; the
  other, actions) are temporally-offset versions of each other and will share
  many values in common (nb: a good setting for using `IndexedSlices`). *This
  op does not include any checks that these pairs of arguments are
  consistent*---that is, it does not ensure that temporally-offset
  arguments really do share the values they are supposed to share.

  In argument descriptions, `T` counts the number of transitions over which
  the Retrace loss is computed, and `B` is the minibatch size. All tensor
  arguments are indexed first by transition, with specific details of this
  indexing in the argument descriptions (pay close attention to "subscripts"
  in variable names).

  Args:
    lambda_: Positive scalar value or 0-D `Tensor` controlling the degree to
      which future timesteps contribute to the loss computed at each
      transition.
    q_tm1: 3-D tensor holding per-action Q-values for the states encountered
      just before taking the transitions that correspond to each major index.
      Since these values are the predicted values we wish to update (in other
      words, the values we intend to change as we learn), in a target network
      setting, these nearly always come from the "non-target" network, which
      we usually call the "learning network".
      Shape is `[T, B, num_actions]`.
    a_tm1: 2-D tensor holding the indices of actions executed during the
      transition that corresponds to each major index.
      Shape is `[T, B]`.
    r_t: 2-D tensor holding rewards received during the transition
      that corresponds to each major index.
      Shape is `[T, B]`.
    pcont_t: 2-D tensor holding pcontinue values received during the
      transition that corresponds to each major index.
      Shape is `[T, B]`.
    target_policy_t: 3-D tensor holding per-action policy probabilities for
      the states encountered just AFTER the transitions that correspond to
      each major index, according to the target policy (i.e. the policy we
      wish to learn). These usually derive from the learning net.
      Shape is `[T, B, num_actions]`.
    behaviour_policy_t: 2-D tensor holding the *behaviour* policy's
      probabilities of having taken action `a_t` at the states encountered
      just AFTER the transitions that correspond to each major index. Derived
      from whatever policy you used to generate the data. All values MUST be
      greater that 0. Shape is `[T, B]`.
    targnet_q_t: 3-D tensor holding per-action Q-values for the states
      encountered just AFTER taking the transitions that correspond to each
      major index. Since these values are used to calculate target values for
      the network, in a target in a target network setting, these should
      probably come from the target network.
      Shape is `[T, B, num_actions]`.
    a_t: 2-D tensor holding the indices of actions executed during the
      transition AFTER the transition that corresponds to each major index.
      Shape is `[T, B]`.
    stop_targnet_gradients: `bool` that enables a sensible default way of
      handling gradients through the Retrace op (essentially, gradients
      are not permitted to involve the `targnet_q_t` input).
      Can be disabled if you require a different arragement, but
      you'll probably want to block some gradients somewhere.
    name: name to prefix ops created by this function.

  Returns:
    A namedtuple with fields:

    * `loss`: Tensor containing the batch of losses, shape `[B]`.
    * `extra`: A namedtuple with fields:
        * `retrace_weights`: Tensor containing batch of retrace weights,
        shape `[T, B]`.
        * `target`: Tensor containing target action values, shape `[T, B]`.
  """
  all_args = [
      lambda_, q_tm1, a_tm1, r_t, pcont_t, target_policy_t, behaviour_policy_t,
      targnet_q_t, a_t
  ]

  with tf.name_scope(name, 'RetraceCore', all_args):
    (lambda_, q_tm1, a_tm1, r_t, pcont_t, target_policy_t, behaviour_policy_t,
     targnet_q_t, a_t) = (
         tf.convert_to_tensor(arg) for arg in all_args)

    # Evaluate importance weights.
    c_t = _retrace_weights(
        indexing_ops.batched_index(target_policy_t, a_t),
        behaviour_policy_t) * lambda_
    # Targets are evaluated by using only Q values from the target network.
    # This provides fixed regression targets until the next target network
    # update.
    target = _general_off_policy_corrected_multistep_target(
        r_t, pcont_t, target_policy_t, c_t, targnet_q_t, a_t,
        not stop_targnet_gradients)

    if stop_targnet_gradients:
      target = tf.stop_gradient(target)
    # Regress Q values of the learning network towards the targets evaluated
    # by using the target network.
    qa_tm1 = indexing_ops.batched_index(q_tm1, a_tm1)
    delta = target - qa_tm1
    loss = 0.5 * tf.square(delta)

    return base_ops.LossOutput(
        loss, RetraceCoreExtra(retrace_weights=c_t, target=target))
