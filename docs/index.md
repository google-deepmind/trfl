# TRFL: Reinforcement Learning Building Blocks

TRFL (pronounced "truffle") is a library built on top of TensorFlow that exposes
several useful building blocks for implementing Reinforcement Learning agents.

## Background

Common RL algorithms describe a particular update to either a Policy, a Value
function, or an Action-Value (Q) function. In Deep-RL, a policy, value- or Q-
function is typically represented by a neural network (the _model_, not to be
confused with an _environment model_, which is used in _model-based RL_). We
formulate common RL update rules for these neural networks as differentiable
_loss_ functions, as is common in (un-)supervised learning. Under automatic
differentiation, the original update rule is recovered. We find that loss
functions are more modular and composable than traditional RL updates, and more
natural when combining with supervised or unsupervised objectives.

The loss functions and other operations provided here are implemented in pure
TensorFlow. They are not complete algorithms, but implementations of RL-specific
mathematical operations needed when building fully-functional RL agents. In
particular, the updates are only valid if the input data are sampled in the
correct manner. For example, the [sequence-advantage-actor-critic
loss](trfl.md#sequence_advantage_actor_critic_loss) (i.e. A2C) is only valid if
the input trajectory is an unbiased sample from the current policy; i.e. the
data are _on-policy_. This library cannot check or enforce such constraints.

## Installation

TRFL can be installed from pip directly from github, with the following command:
`pip install git+git://github.com/deepmind/trfl.git`

TRFL will work with both the CPU and GPU version of tensorflow, but to allow
for that it does not list Tensorflow as a requirement, so you need to install
Tensorflow and Tensorflow-probability separately if you haven't already done so.

## Example usage

Import TensorFlow and TRFL.

```python
import tensorflow as tf
import trfl
```

Define the relevant data associated to a `transition` in the environment from
state `s_tm1` to state `s_t`. This typically includes action values (or other
characterization of the agent's policy) in both the `source` and `destination`
states. The action `a_tm1` is the one selected after observing `s_tm1`, and
resulted in observing the immediate reward `r_t` and the subsequent state `s_t`.
`pcont_t` represents a time dependent discount factor, or (equivalently) the
continuation probability from state `s_t`. In most applications, its value will
be equal to a constant factor (e.g., 0.99), except if `s_t` is the final state
in an episode, in which case it is set to zero.

```python
# Q-values for the previous and next timesteps, shape [batch_size, num_actions].
q_tm1 = tf.get_variable(
    "q_tm1", initializer=[[1., 1., 0.], [1., 2., 0.]], dtype=tf.float32)
q_t = tf.get_variable(
    "q_t", initializer=[[0., 1., 0.], [1., 2., 0.]], dtype=tf.float32)

# Action indices, discounts and rewards, shape [batch_size].
a_tm1 = tf.constant([0, 1], dtype=tf.int32)
r_t = tf.constant([1, 1], dtype=tf.float32)
pcont_t = tf.constant([0, 1], dtype=tf.float32)  # the discount factor

# Q-learning loss, and auxiliary data.
loss, q_learning = trfl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)
```

`loss` is the tensor representing the loss. For Q-learning, it is half the
squared difference between the predicted Q-values and the TD targets, shape
`[batch_size]`. Extra information is in the `q_learning` namedtuple, including
`q_learning.td_error` and `q_learning.target`.

Most of the time, you may only be interested in the loss, in which case you can
use any of the following expressions:

```python
loss, _ = trfl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)
loss = trfl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t).loss
```

The `loss` tensor can be differentiated to derive the corresponding RL update.
Note that in Q-learning, as in other bootstrapped losses, the TD targets
are wrapped in a `tf.stop_gradient`. Differentiating `loss` therefore
results in gradients with respect to `q_tm1` but not with respect to `q_t`.

```python
reduced_loss = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(reduced_loss)
```

All loss functions in the package return both a loss tensor and a namedtuple
with extra information, using the above convention, but different functions
may have different `extra` fields. Check the documentation of each function
below for more information.

## Naming Conventions and Developer Guidelines

Throughout the package, we use the following conventions:

*   Time indices and variable names:

    *   `q_tm1`: the action value in the `source` state of a transition.
    *   `a_tm1`: the action that was selected in the `source` state.
    *   `r_t`: the resulting rewards collected in the `destination` state.
    *   `pcont_t`: the continuation probability / `discount` for a transition.
    *   `q_t`: the action values in the `destination` state.

*   Tensor shapes:

    *   All ops support minibatches only. We use `B` to denote the batch size.
    *   Batches of rewards, continuation probabilities / discounts have shape [B].
    *   Batches of state-values have shape `[B]`.
    *   Batches of action-values / q-values have shape `[B, num_actions]`.
    *   All losses have shape [B], i.e. the loss is not reduced over the batch
        dimension. This allows the user to easily weight the loss for different
        elements of the batch (e.g., by their importance sampling weights).
    *   For ops that take batches of sequences of data, `T` denotes the sequence
        length. Tensors are time-major, and have shape `[T, B, ...]`. Index `0`
        of the time dimension is assumed to be the start of the sequence.

## Learning updates

*   State Value learning:

    *   [td_learning](trfl.md#td_learningv_tm1-r_t-pcont_t-v_t-nametdlearning)
    *   [generalized_lambda_returns](trfl.md#generalized_lambda_returnsrewards-pcontinues-values-bootstrap_value-lambda_1-namegeneralized_lambda_returns)
    *   [td_lambda](trfl.md#td_lambdastate_values-rewards-pcontinues-bootstrap_value-lambda_1-namebaselineloss)
    *   [qv_max](trfl.md#qv_maxv_tm1-r_t-pcont_t-q_t-nameqvmax)

*   Discrete-action Value learning:

    *   [qlearning](trfl.md#qlearningq_tm1-a_tm1-r_t-pcont_t-q_t-nameqlearning)
    *   [double_qlearning](trfl.md#double_qlearningq_tm1-a_tm1-r_t-pcont_t-q_t_value-q_t_selector-namedoubleqlearning)
    *   [persistent_qlearning](trfl.md#persistent_qlearningq_tm1-a_tm1-r_t-pcont_t-q_t-action_gap_scale05-namepersistentqlearning)
    *   [sarsa](trfl.md#sarsaq_tm1-a_tm1-r_t-pcont_t-q_t-a_t-namesarsa)
    *   [sarse](trfl.md#sarseq_tm1-a_tm1-r_t-pcont_t-q_t-probs_a_t-debugfalse-namesarse)
    *   [qlambda](trfl.md#qlambdaq_tm1-a_tm1-r_t-pcont_t-q_t-lambda_-namegeneralizedqlambda)
    *   [qv_learning](trfl.md#qv_learningq_tm1-a_tm1-r_t-pcont_t-v_t-nameqvlearning)

*   Distributional Value learning:

    *   [categorical_dist_qlearning](trfl.md#categorical_dist_qlearningatoms_tm1-logits_q_tm1-a_tm1-r_t-pcont_t-atoms_t-logits_q_t-namecategoricaldistqlearning)
    *   [categorical_dist_double_qlearning](trfl.md#categorical_dist_double_qlearningatoms_tm1-logits_q_tm1-a_tm1-r_t-pcont_t-atoms_t-logits_q_t-q_t_selector-namecategoricaldistdoubleqlearning)
    *   [categorical_dist_td_learning](trfl.md#categorical_dist_td_learningatoms_tm1-logits_v_tm1-r_t-pcont_t-atoms_t-logits_v_t-namecategoricaldisttdlearning)

*   Continuous-action Policy Gradient:

    *   [policy_gradient](trfl.md#policy_gradientpolicies-actions-action_values-policy_varsnone-namepolicy_gradient)
    *   [policy_gradient_loss](trfl.md#policy_gradient_losspolicies-actions-action_values-policy_varsnone-namepolicy_gradient_loss)
    *   [policy_entropy_loss](trfl.md#policy_entropy_losspolicies-policy_varsnone-scale_opnone-namepolicy_entropy_loss)
    *   [sequence_a2c_loss](trfl.md#sequence_a2c_losspolicies-baseline_values-actions-rewards-pcontinues-bootstrap_value-policy_varsnone-lambda_1-entropy_costnone-baseline_cost1-entropy_scale_opnone-namesequencea2closs)

*   Deterministic Policy Gradient:

    *   [dpg](trfl.md#dpg)

*   Discrete-action Policy Gradient:

    *   [discrete_policy_entropy_loss](trfl.md#discrete_policy_entropy_losspolicy_logits-normalisefalse-namediscrete_policy_entropy_loss)
    *   [sequence_advantage_actor_critic_loss](trfl.md#sequence_advantage_actor_critic_losspolicy_logits-baseline_values-actions-rewards-pcontinues-bootstrap_value-lambda_1-entropy_costnone-baseline_cost1-normalise_entropyfalse-namesequenceadvantageactorcriticloss):
        this is the commonly-used A2C/A3C loss function.
    *   [discrete_policy_gradient](trfl.md#discrete_policy_gradientpolicy_logits-actions-action_values-namediscrete_policy_gradient)
    *   [discrete_policy_gradient_loss](trfl.md#discrete_policy_gradient_losspolicy_logits-actions-action_values-namediscrete_policy_gradient_loss)

*   Pixel control:

    *   [pixel_control_rewards](trfl.md#pixel_control_rewardsobservations-cell_size)
    *   [pixel_control_loss](trfl.md#pixel_control_lossobservations-actions-action_values-cell_size-discount_factor-scale-crop_height_dimnone-none-crop_width_dimnone-none)

*   Retrace:

    *   [retrace](trfl.md#retracelambda_-qs-targnet_qs-actions-rewards-pcontinues-target_policy_probs-behaviour_policy_probs-stop_targnet_gradientstrue-namenone)
    *   [retrace_core](trfl.md#retrace_corelambda_-q_tm1-a_tm1-r_t-pcont_t-target_policy_t-behaviour_policy_t-targnet_q_t-a_t-stop_targnet_gradientstrue-namenone)

*   Target Network Updating:

    *   [update_target_variables](trfl.md#update_target_variablestarget_variables-source_variables-tau10-use_lockingfalse-nameupdate_target_variables)
    *   [periodic_target_update](trfl.md#periodic_target_updatetarget_variables-source_variables-update_period-tau10,use_lockingfalse-nameperiodic_target_update)

*   V-trace:

    *   [vtrace_from_logits](trfl.md#vtrace_from_logitsbehaviour_policy_logits-target_policy_logits-actions-discounts-rewards-values-bootstrap_value-clip_rho_threshold10-clip_pg_rho_threshold10-namevtrace_from_logits)
    *   [vtrace_from_importance_weights](trfl.md#vtrace_from_importance_weightslog_rhos-discounts-rewards-values-bootstrap_value-clip_rho_threshold10-clip_pg_rho_threshold10-namevtrace_from_importance_weights)

## Other

*   Clipping ops

    *   [huber_loss](trfl.md#huber_lossinput_tensor-quadratic_linear_boundary-namenone)

*   Distributions

    *   [l2_project](trfl.md#l2_projectsupport-weights-new_support)
    *   [hard_cumulative_project](trfl.md#hard_cumulative_projectsupport-weights-new_support-reverse)
    *   [factorised_kl_gaussian](trfl.md#factorised_kl_gaussiandist1_mean-dist1_covariance_or_scale-dist2_mean-dist2_covariance_or_scale-both_diagonalfalse)

*   Indexing ops

    *   [batched_index](trfl.md#batched_indexvalues-indices)

*   Periodic execution ops

    *   [periodically](trfl.md#periodicallybody-period-nameperiodically)

*   Sequence ops

    *   [scan_discounted_sum](trfl.md#scan_discounted_sumsequence-decay-initial_value-reversefalse-sequence_lengthsnone-back_proptrue-namescan_discounted_sum)
    *   [multistep_forward_view](trfl.md#multistep_forward_viewrewards-pcontinues-state_values-lambda_-back_proptrue-sequence_lengthsnone-namemultistep_forward_view_op)

## More information

*   [Multistep Forward View](multistep_forward_view.md)
