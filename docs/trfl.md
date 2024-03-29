<!-- common_typos_disable -->
<!--* freshness: { exempt: true } *-->

# trfl - module reference
Flattened namespace for trfl.

## Other Functions and Classes
### [`assert_rank_and_shape_compatibility(tensors, rank)`](https://github.com/deepmind/trfl/blob/master/trfl/base_ops.py?l=64)<!-- RULE: assert_rank_and_shape_compatibility .code-reference -->

Asserts that the tensors have the correct rank and compatible shapes.

Shapes (of equal rank) are compatible if corresponding dimensions are all
equal or unspecified. E.g. `[2, 3]` is compatible with all of `[2, 3]`,
`[None, 3]`, `[2, None]` and `[None, None]`.

##### Args:


* `tensors`: List of tensors.
* `rank`: A scalar specifying the rank that the tensors passed need to have.

##### Raises:


* `ValueError`: If the list of tensors is empty or fail the rank and mutual
    compatibility asserts.


### [`batched_index(values, indices, keepdims=None)`](https://github.com/deepmind/trfl/blob/master/trfl/indexing_ops.py?l=64)<!-- RULE: batched_index .code-reference -->

Equivalent to `values[:, indices]`.

Performs indexing on batches and sequence-batches by reducing over
zero-masked values. Compared to indexing with `tf.gather` this approach is
more general and TPU-friendly, but may be less efficient if `num_values`
is large. It works with tensors whose shapes are unspecified or
partially-specified, but this op will only do shape checking on shape
information available at graph construction time. When complete shape
information is absent, certain shape incompatibilities may not be detected at
runtime! See `indexing_ops_test` for detailed examples.

##### Args:


* `values`: tensor of shape `[B, num_values]` or `[T, B, num_values]`
* `indices`: tensor of shape `[B]` or `[T, B]` containing indices.
* `keepdims`: If `True`, the returned tensor will have an added 1 dimension at
    the end (e.g. `[B, 1]` or `[T, B, 1]`).

##### Returns:

  Tensor of shape `[B]` or `[T, B]` containing values for the given indices.


* `Raises`: ValueError if values and indices have sizes that are known
  statically (i.e. during graph construction), and those sizes are not
  compatible (see shape descriptions in Args list above).


### [`best_effort_shape(tensor, with_rank=None)`](https://github.com/deepmind/trfl/blob/master/trfl/base_ops.py?l=31)<!-- RULE: best_effort_shape .code-reference -->

Extract as much static shape information from a tensor as possible.

##### Args:


* `tensor`: A `Tensor`. If `with_rank` is None, must have statically-known
      number of dimensions.
* `with_rank`: Optional, an integer number of dimensions to force the shape to
      be. Useful for tensors with no static shape information that must be
      of a particular rank. Default is None (number of dimensions must be
      statically known).

##### Returns:

  An iterable with length equal to the number of dimensions in `tensor`,
  containing integers for the dimensions with statically-known size, and
  scalar `Tensor`s for dimensions with size only known at run-time.

##### Raises:


* `ValueError`: If `with_rank` is None and `tensor` does not have
    statically-known number of dimensions.


### [`categorical_dist_double_qlearning(atoms_tm1, logits_q_tm1, a_tm1, r_t, pcont_t, atoms_t, logits_q_t, q_t_selector, name='CategoricalDistDoubleQLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py?l=148)<!-- RULE: categorical_dist_double_qlearning .code-reference -->

Implements Distributional Double Q-learning as TensorFlow ops.

The function assumes categorical value distributions parameterized by logits,
and combines distributional RL with double Q-learning.

See "Rainbow: Combining Improvements in Deep Reinforcement Learning" by
Hessel, Modayil, van Hasselt, Schaul et al.
(https://arxiv.org/abs/1710.02298).

##### Args:


* `atoms_tm1`: 1-D tensor containing atom values for first timestep,
    shape `[num_atoms]`.
* `logits_q_tm1`: Tensor holding logits for first timestep in a batch of
    transitions, shape `[B, num_actions, num_atoms]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `atoms_t`: 1-D tensor containing atom values for second timestep,
    shape `[num_atoms]`.
* `logits_q_t`: Tensor holding logits for second timestep in a batch of
    transitions, shape `[B, num_actions, num_atoms]`.
* `q_t_selector`: Tensor holding another set of Q-values for second timestep
    in a batch of transitions, shape `[B, num_actions]`.
    These values are used for estimating the best action. In Double DQN they
    come from the online network.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: Tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`:  Tensor containing the values that `q_tm1` at actions
      `a_tm1` are regressed towards, shape `[B, num_atoms]` .

##### Raises:


* `ValueError`: If the tensors do not have the correct rank or compatibility.


### [`categorical_dist_qlearning(atoms_tm1, logits_q_tm1, a_tm1, r_t, pcont_t, atoms_t, logits_q_t, name='CategoricalDistQLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py?l=73)<!-- RULE: categorical_dist_qlearning .code-reference -->

Implements Distributional Q-learning as TensorFlow ops.

The function assumes categorical value distributions parameterized by logits.

See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
Dabney and Munos. (https://arxiv.org/abs/1707.06887).

##### Args:


* `atoms_tm1`: 1-D tensor containing atom values for first timestep,
    shape `[num_atoms]`.
* `logits_q_tm1`: Tensor holding logits for first timestep in a batch of
    transitions, shape `[B, num_actions, num_atoms]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `atoms_t`: 1-D tensor containing atom values for second timestep,
    shape `[num_atoms]`.
* `logits_q_t`: Tensor holding logits for second timestep in a batch of
    transitions, shape `[B, num_actions, num_atoms]`.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: a tensor containing the values that `q_tm1` at actions
      `a_tm1` are regressed towards, shape `[B, num_atoms]`.

##### Raises:


* `ValueError`: If the tensors do not have the correct rank or compatibility.


### [`categorical_dist_td_learning(atoms_tm1, logits_v_tm1, r_t, pcont_t, atoms_t, logits_v_t, name='CategoricalDistTDLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py?l=230)<!-- RULE: categorical_dist_td_learning .code-reference -->

Implements Distributional TD-learning as TensorFlow ops.

The function assumes categorical value distributions parameterized by logits.

See "A Distributional Perspective on Reinforcement Learning" by Bellemare,
Dabney and Munos. (https://arxiv.org/abs/1707.06887).

##### Args:


* `atoms_tm1`: 1-D tensor containing atom values for first timestep,
    shape `[num_atoms]`.
* `logits_v_tm1`: Tensor holding logits for first timestep in a batch of
    transitions, shape `[B, num_atoms]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `atoms_t`: 1-D tensor containing atom values for second timestep,
    shape `[num_atoms]`.
* `logits_v_t`: Tensor holding logits for second timestep in a batch of
    transitions, shape `[B, num_atoms]`.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: Tensor containing the batch of losses, shape `[B]`.
  * `extra`: A namedtuple with fields:
      * `target`: Tensor containing the values that `v_tm1` are
      regressed towards, shape `[B, num_atoms]`.

##### Raises:


* `ValueError`: If the tensors do not have the correct rank or compatibility.


### [`discrete_policy_entropy_loss(policy_logits, normalise=False, name='discrete_policy_entropy_loss')`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?l=42)<!-- RULE: discrete_policy_entropy_loss .code-reference -->

Computes the entropy 'loss' for a batch of policy logits.

Given a batch of policy logits, calculates the entropy and corrects the sign
so that minimizing the resulting loss op is equivalent to increasing entropy
in the batch. This loss is optionally normalised to the range `[-1, 0]` by
dividing by the log number of actions. This makes it more invariant to the
size of the action space.

This function accepts a nested array of `policy_logits` in order
to allow for multiple discrete actions. In this case, the loss is given by
`-sum_i(H(p_i))` where `p_i` are members of the `policy_logits` nest and
H is the Shannon entropy.

##### Args:


* `policy_logits`: A (possibly nested structure of) (N+1)-D Tensor(s) with
      shape `[..., A]`,  representing the log-probabilities of a set of
      Categorical distributions, where `...` represents at least one
      dimension (e.g., batch, sequence), and `A` is the number of discrete
      actions (which need not be identical across all tensors).
      Does not need to be centered.
* `normalise`: If True, divide the loss by the `sum_i(log(A_i))` where `A_i`
      is the number of actions for the i'th tensor in the `policy_logits`
      nest. Default is False.
* `name`: Optional, name of this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: Entropy 'loss', shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `entropy`: Entropy of the policy, shape `[B]`.


### [`discrete_policy_gradient(policy_logits, actions, action_values, name='discrete_policy_gradient')`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?l=223)<!-- RULE: discrete_policy_gradient .code-reference -->

Computes a batch of discrete-action policy gradient losses.

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

##### Args:


* `policy_logits`: (N+1)-D Tensor of shape
      `[batch_size_1, ..., batch_size_N, num_actions]` containing uncentered
      log-probabilities.
* `actions`: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]` and integer
      type, containing indices for the selected actions.
* `action_values`: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]`
      containing an estimate of the value of the selected `actions`.
* `name`: Customises the name_scope for this op.

##### Returns:


* `loss`: N-D Tensor of shape `[batch_size_1, ..., batch_size_N]` containing the
      loss. Differentiable w.r.t `policy_logits` only.

##### Raises:


* `ValueError`: If the batch dimensions of `policy_logits` and `action_values`
      do not match.


### [`discrete_policy_gradient_loss(policy_logits, actions, action_values, name='discrete_policy_gradient_loss')`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?l=279)<!-- RULE: discrete_policy_gradient_loss .code-reference -->

Computes discrete policy gradient losses for a batch of trajectories.

This wraps `discrete_policy_gradient` to accept a possibly nested array of
`policy_logits` and `actions` in order to allow for multiple discrete actions.
It also sums up losses along the time dimension, and is more restrictive about
shapes, assuming a [T, B] layout.

##### Args:


* `policy_logits`: A (possibly nested structure of) Tensor(s) of shape
      `[T, B, num_actions]` containing uncentered log-probabilities.
* `actions`: A (possibly nested structure of) Tensor(s) of shape
      `[T, B]` and integer type, containing indices for the selected actions.
* `action_values`: Tensor of shape `[T, B]`
      containing an estimate of the value of the selected `actions`, see
      `discrete_policy_gradient`.
* `name`: Customises the name_scope for this op.

##### Returns:


* `loss`: Tensor of shape `[B]` containing the total loss for each sequence
  in the batch. Differentiable w.r.t `policy_logits` only.


### [`double_qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t_value, q_t_selector, name='DoubleQLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=88)<!-- RULE: double_qlearning .code-reference -->

Implements the double Q-learning loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
the target `r_t + pcont_t * q_t_value[argmax q_t_selector]`.

See "Double Q-learning" by van Hasselt.
(https://papers.nips.cc/paper/3964-double-q-learning.pdf).

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t_value`: Tensor of Q-values for second timestep in a batch of transitions,
    used to estimate the value of the best action, shape `[B x num_actions]`.
* `q_t_selector`: Tensor of Q-values for second timestep in a batch of
    transitions used to estimate the best action, shape `[B x num_actions]`.
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`
      * `td_error`: batch of temporal difference errors, shape `[B]`
      * `best_action`: batch of greedy actions wrt `q_t_selector`, shape `[B]`


### [`dpg(q_max, a_max, dqda_clipping=None, clip_norm=False, name='DpgLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/dpg_ops.py?l=36)<!-- RULE: dpg .code-reference -->

Implements the Deterministic Policy Gradient (DPG) loss as a TensorFlow Op.

This op implements the loss for the `actor`, the `critic` can instead be
updated by minimizing the `value_ops.td_learning` loss.

See "Deterministic Policy Gradient Algorithms" by Silver, Lever, Heess,
Degris, Wierstra, Riedmiller (http://proceedings.mlr.press/v32/silver14.pdf).

##### Args:


* `q_max`: Tensor holding Q-values generated by Q network with the input of
    (state, a_max) pair, shape `[B]`.
* `a_max`: Tensor holding the optimal action, shape `[B, action_dimension]`.
* `dqda_clipping`: `int` or `float`, clips the gradient dqda element-wise
    between `[-dqda_clipping, dqda_clipping]`.
* `clip_norm`: Whether to perform dqda clipping on the vector norm of the last
    dimension, or component wise (default).
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `q_max`: Tensor holding the optimal Q values, `[B]`.
      * `a_max`: Tensor holding the optimal action, `[B, action_dimension]`.
      * `dqda`: Tensor holding the derivative dq/da, `[B, action_dimension]`.

##### Raises:


* `ValueError`: If `q_max` doesn't depend on `a_max` or if `dqda_clipping <= 0`.


### [`epsilon_greedy(action_values, epsilon, legal_actions_mask=None)`](https://github.com/deepmind/trfl/blob/master/trfl/policy_ops.py?l=28)<!-- RULE: epsilon_greedy .code-reference -->

Computes an epsilon-greedy distribution over actions.

This returns a categorical distribution over a discrete action space. It is
assumed that the trailing dimension of `action_values` is of length A, i.e.
the number of actions. It is also assumed that actions are 0-indexed.

This policy does the following:

- With probability 1 - epsilon, take the action corresponding to the highest
action value, breaking ties uniformly at random.
- With probability epsilon, take an action uniformly at random.

##### Args:


* `action_values`: A Tensor of action values with any rank >= 1 and dtype float.
    Shape can be flat ([A]), batched ([B, A]), a batch of sequences
    ([T, B, A]), and so on.
* `epsilon`: A scalar Tensor (or Python float) with value between 0 and 1.
* `legal_actions_mask`: An optional one-hot tensor having the shame shape and
    dtypes as `action_values`, defining the legal actions:
    legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
    If not provided, all actions will be considered legal and
    `tf.ones_like(action_values)`.

##### Returns:


* `policy`: tfp.distributions.Categorical distribution representing the policy.


### [`generalized_lambda_returns(rewards, pcontinues, values, bootstrap_value, lambda_=1, name='generalized_lambda_returns')`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?l=75)<!-- RULE: generalized_lambda_returns .code-reference -->

Computes lambda-returns along a batch of (chunks of) trajectories.

For lambda=1 these will be multistep returns looking ahead from each
state to the end of the chunk, where bootstrap_value is used. If you pass an
entire trajectory and zeros for bootstrap_value, this is just the Monte-Carlo
return / TD(1) target.

For lambda=0 these are one-step TD(0) targets.

For inbetween values of lambda these are lambda-returns / TD(lambda) targets,
except that traces are always cut off at the end of the chunk, since we can't
see returns beyond then. If you pass an entire trajectory with zeros for
bootstrap_value though, then they're plain TD(lambda) targets.

lambda can also be a tensor of values in [0, 1], determining the mix of
bootstrapping vs further accumulation of multistep returns at each timestep.
This can be used to implement Retrace and other algorithms. See
`sequence_ops.multistep_forward_view` for more info on this. Another way to
think about the end-of-chunk cutoff is that lambda is always effectively zero
on the timestep after the end of the chunk, since at the end of the chunk we
rely entirely on bootstrapping and can't accumulate returns looking further
into the future.

The sequences in the tensors should be aligned such that an agent in a state
with value `V` transitions into another state with value `V'`, receiving
reward `r` and pcontinue `p`. Then `V`, `r` and `p` are all at the same index
`i` in the corresponding tensors. `V'` is at index `i+1`, or in the
`bootstrap_value` tensor if `i == T`.

Subtracting `values` from these lambda-returns will yield estimates of the
advantage function which can be used for both the policy gradient loss and
the baseline value function loss in A3C / GAE.

##### Args:


* `rewards`: 2-D Tensor with shape `[T, B]`.
* `pcontinues`: 2-D Tensor with shape `[T, B]`.
* `values`: 2-D Tensor containing estimates of the state values for timesteps
    0 to `T-1`. Shape `[T, B]`.
* `bootstrap_value`: 1-D Tensor containing an estimate of the value of the
    final state at time `T`, used for bootstrapping the target n-step
    returns. Shape `[B]`.
* `lambda_`: an optional scalar or 2-D Tensor with shape `[T, B]`.
* `name`: Customises the name_scope for this op.

##### Returns:

  2-D Tensor with shape `[T, B]`


### [`huber_loss(input_tensor, quadratic_linear_boundary, name=None)`](https://github.com/deepmind/trfl/blob/master/trfl/clipping_ops.py?l=25)<!-- RULE: huber_loss .code-reference -->

Calculates huber loss of `input_tensor`.

For each value x in `input_tensor`, the following is calculated:

```
  0.5 * x^2                  if |x| <= d
  0.5 * d^2 + d * (|x| - d)  if |x| > d
```

where d is `quadratic_linear_boundary`.

When `input_tensor` is a loss this results in a form of gradient clipping.
This is, for instance, how gradients are clipped in DQN and its variants.

##### Args:


* `input_tensor`: `Tensor`, input values to calculate the huber loss on.
* `quadratic_linear_boundary`: `float`, the point where the huber loss function
    changes from a quadratic to linear.
* `name`: `string`, name for the operation (optional).

##### Returns:

  `Tensor` of the same shape as `input_tensor`, containing values calculated
  in the manner described above.

##### Raises:


* `ValueError`: if quadratic_linear_boundary <= 0.


### [`multistep_forward_view(rewards, pcontinues, state_values, lambda_, back_prop=True, sequence_lengths=None, name='multistep_forward_view_op')`](https://github.com/deepmind/trfl/blob/master/trfl/sequence_ops.py?l=127)<!-- RULE: multistep_forward_view .code-reference -->

Evaluates complex backups (forward view of eligibility traces).

  ```python
  result[t] = rewards[t] +
      pcontinues[t]*(lambda_[t]*result[t+1] + (1-lambda_[t])*state_values[t])
  result[last] = rewards[last] + pcontinues[last]*state_values[last]
  ```

  This operation evaluates multistep returns where lambda_ parameter controls
  mixing between full returns and boostrapping. It is users responsibility
  to provide state_values. Depending on how state_values are evaluated this
  function can evaluate targets for Q(lambda), Sarsa(lambda) or some other
  multistep boostrapping algorithm.

  More information about a forward view is given here:
    http://incompleteideas.net/sutton/book/ebook/node74.html

  Please note that instead of evaluating traces and then explicitly summing
  them we instead evaluate mixed returns in the reverse temporal order
  by using the recurrent relationship given above.

  The parameter lambda_ can either be a constant value (e.g for Peng's
  Q(lambda) and Sarsa(_lambda)) or alternatively it can be a tensor containing
  arbitrary values (Watkins' Q(lambda), Munos' Retrace, etc).

  The result of evaluating this recurrence relation is a weighted sum of
  n-step returns, as depicted in the diagram below. One strategy to prove this
  equivalence notes that many of the terms in adjacent n-step returns
  "telescope", or cancel out, when the returns are summed.

  Below L3 is lambda at time step 3 (important: this diagram is 1-indexed, not
  0-indexed like Python). If lambda is scalar then L1=L2=...=Ln.
  g1,...,gn are discounts.

  ```
  Weights:  (1-L1)        (1-L2)*l1      (1-L3)*l1*l2  ...  L1*L2*...*L{n-1}
  Returns:    |r1*(g1)+     |r1*(g1)+      |r1*(g1)+          |r1*(g1)+
            v1*(g1)         |r2*(g1*g2)+   |r2*(g1*g2)+       |r2*(g1*g2)+
                          v2*(g1*g2)       |r3*(g1*g2*g3)+    |r3*(g1*g2*g3)+
                                         v3*(g1*g2*g3)               ...
                                                              |rn*(g1*...*gn)+
                                                            vn*(g1*...*gn)
  ```

##### Args:


* `rewards`: Tensor of shape `[T, B]` containing rewards.
* `pcontinues`: Tensor of shape `[T, B]` containing discounts.
* `state_values`: Tensor of shape `[T, B]` containing state values.
* `lambda_`: Mixing parameter lambda.
      The parameter can either be a scalar or a Tensor of shape `[T, B]`
      if mixing is a function of state.
* `back_prop`: Whether to backpropagate.
* `sequence_lengths`: Tensor of shape `[B]` containing sequence lengths to be
    (reversed and then) summed, same as in `scan_discounted_sum`.
* `name`: Sets the name_scope for this op.

##### Returns:

    Tensor of shape `[T, B]` containing multistep returns.


### [`periodic_target_update(target_variables, source_variables, update_period, tau=1.0, use_locking=False, counter=None, name='periodic_target_update')`](https://github.com/deepmind/trfl/blob/master/trfl/target_update_ops.py?l=89)<!-- RULE: periodic_target_update .code-reference -->

Returns an op to periodically update a list of target variables.

The `update_target_variables` op is executed every `update_period`
executions of the `periodic_target_update` op.

The update rule is:
`target_variable = (1 - tau) * target_variable + tau * source_variable`.

##### Args:


* `target_variables`: a list of the variables to be updated.
* `source_variables`: a list of the variables used for the update.
* `update_period`: inverse frequency with which to apply the update.
* `tau`: weight used to gate the update. The permitted range is 0 < tau <= 1,
    with small tau representing an incremental update, and tau == 1
    representing a full update (that is, a straight copy).
* `use_locking`: use `tf.variable.Assign`'s locking option when assigning
    source variable values to target variables.
* `counter`: an optional tensorflow variable to use as a counter relative to
    `update_period`, which be passed to `periodic_ops.periodically`. See
    description in `periodic_ops.periodically` for details.
* `name`: sets the `name_scope` for this op.

##### Returns:

  An op that periodically updates `target_variables` with `source_variables`.


### [`periodically(body, period, counter=None, name='periodically')`](https://github.com/deepmind/trfl/blob/master/trfl/periodic_ops.py?l=34)<!-- RULE: periodically .code-reference -->

Periodically performs a tensorflow op.

The body tensorflow op will be executed every `period` times the periodically
op is executed. More specifically, with `n` the number of times the op has
been executed, the body will be executed when `n` is a non zero positive
multiple of `period` (i.e. there exist an integer `k > 0` such that
`k * period == n`).

If `period` is 0 or `None`, it would not perform any op and would return a
`tf.no_op()`.

##### Args:


* `body`: callable that returns the tensorflow op to be performed every time
    an internal counter is divisible by the period. The op must have no
    output (for example, a tf.group()).
* `period`: inverse frequency with which to perform the op.
* `counter`: an optional tensorflow variable to use as a counter relative to the
    period. It will be incremented per call and reset to 1 in every update. In
    order to ensure that `body` is run in the first count, initialize the
    counter at a value bigger than `period`. If not given, an internal counter
    will be created in the graph. (not that this is incompatible with
    Tensorflow 2 behavior)
* `name`: name of the variable_scope.

##### Raises:


* `TypeError`: if body is not a callable.
* `ValueError`: if period is negative.

##### Returns:

  An op that periodically performs the specified op.


### [`persistent_qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t, action_gap_scale=0.5, name='PersistentQLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=142)<!-- RULE: persistent_qlearning .code-reference -->

Implements the persistent Q-learning loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
`r_t + pcont_t * [(1-action_gap_scale) max q_t + action_gap_scale qa_t]`

See "Increasing the Action Gap: New Operators for Reinforcement Learning"
by Bellemare, Ostrovski, Guez et al. (https://arxiv.org/abs/1512.04860).

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t`: Tensor holding Q-values for second timestep in a batch of
    transitions, shape `[B x num_actions]`.
    These values are used for estimating the value of the best action. In
    DQN they come from the target network.
* `action_gap_scale`: coefficient in [0, 1] for scaling the action gap term.
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`pixel_control_loss(observations, actions, action_values, cell_size, discount_factor, scale, crop_height_dim=(None, None), crop_width_dim=(None, None))`](https://github.com/deepmind/trfl/blob/master/trfl/pixel_control_ops.py?l=95)<!-- RULE: pixel_control_loss .code-reference -->

Calculate n-step Q-learning loss for pixel control auxiliary task.

For each pixel-based pseudo reward signal, the corresponding action-value
function is trained off-policy, using Q(lambda). A discount of 0.9 is
commonly used for learning the value functions.

Note that, since pseudo rewards have a spatial structure, with neighbouring
cells exhibiting strong correlations, it is convenient to predict the action
values for all the cells through a deconvolutional head.

See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

##### Args:


* `observations`: A tensor of shape `[T+1,B, ...]`; `...` is the observation
    shape, `T` the sequence length, and `B` the batch size. `T` and `B` can
    be statically unknown for `observations`, `actions` and `action_values`.
* `actions`: A tensor, shape `[T,B]`, of the actions across each sequence.
* `action_values`: A tensor, shape `[T+1,B,H,W,N]` of pixel control action
    values, where `H`, `W` are the number of pixel control cells/tasks, and
    `N` is the number of actions.
* `cell_size`: size of the cells used to derive the pixel based pseudo-rewards.
* `discount_factor`: discount used for learning the value function associated
    to the pseudo rewards; must be a scalar or a Tensor of shape [T,B].
* `scale`: scale factor for pixels in `observations`.
* `crop_height_dim`: tuple (min_height, max_height) specifying how
    to crop the input observations before computing the pseudo-rewards.
* `crop_width_dim`: tuple (min_width, max_width) specifying how
    to crop the input observations before computing the pseudo-rewards.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape [B].
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
      * `td_error`: batch of temporal difference errors, shape [B].

##### Raises:


* `ValueError`: if the shape of `action_values` is not compatible with that of
    the pseudo-rewards derived from the observations.


### [`pixel_control_rewards(observations, cell_size)`](https://github.com/deepmind/trfl/blob/master/trfl/pixel_control_ops.py?l=41)<!-- RULE: pixel_control_rewards .code-reference -->

Calculates pixel control task rewards from observation sequence.

The observations are first split in a grid of KxK cells. For each cell a
distinct pseudo reward is computed as the average absolute change in pixel
intensity for all pixels in the cell. The change in intensity is averaged
across both pixels and channels (e.g. RGB).

The `observations` provided to this function should be cropped suitably, to
ensure that the observations' height and width are a multiple of `cell_size`.
The values of the `observations` tensor should be rescaled to [0, 1]. In the
UNREAL agent observations are cropped to 80x80, and each cell is 4x4 in size.

See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

##### Args:


* `observations`: A tensor of shape `[T+1,B,H,W,C...]`, where
    * `T` is the sequence length, `B` is the batch size.
    * `H` is height, `W` is width.
    * `C...` is at least one channel dimension (e.g., colour, stack).
    * `T` and `B` can be statically unknown.
* `cell_size`: The size of each cell.

##### Returns:

  A tensor of pixel control rewards calculated from the observation. The
  shape is `[T,B,H',W']`, where `H'` and `W'` are determined by the
  `cell_size`. If evenly-divisible, `H' = H/cell_size`, and similar for `W`.


### [`policy_entropy_loss(policies, policy_vars=None, scale_op=None, name='policy_entropy_loss')`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?l=137)<!-- RULE: policy_entropy_loss .code-reference -->

Calculates entropy 'loss' for policies represented by a distributions.

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

##### Args:


* `policies`: A (possibly nested structure of) batch distribution(s)
      supporting an `entropy` method that returns an N-D Tensor with shape
      equal to the `batch_shape` of the distribution, e.g. an instance of
      `tfp.distributions.Distribution`.
* `policy_vars`: An optional (possibly nested structure of) iterable(s) of
      Tensors used by `policies`. If provided is used in scope checks.
* `scale_op`: An optional op that takes `policies` as its only argument and
      returns a scalar Tensor that is used to scale the entropy loss.
      E.g. for Diag(sigma) Gaussian policies dividing by the number of
      dimensions makes entropy loss invariant to the action space dimension.
* `name`: Optional, name of this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B1, B2, ...]`.
  * `extra`: a namedtuple with fields:
      * `entropy`: entropy of the policy, shape `[B1, B2, ...]`.
  where [B1, B2, ... ] == policy.batch_shape


### [`policy_gradient(policies, actions, action_values, policy_vars=None, name='policy_gradient')`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?l=39)<!-- RULE: policy_gradient .code-reference -->

Computes policy gradient losses for a batch of trajectories.

See `policy_gradient_loss` for more information on expected inputs and usage.

##### Args:


* `policies`: A distribution over a batch supporting a `log_prob` method, e.g.
      an instance of `tfp.distributions.Distribution`. For example, for
      a diagonal gaussian policy:
      `policies = tfp.distributions.MultivariateNormalDiag(mus, sigmas)`
* `actions`: An action batch Tensor used as the argument for `log_prob`. Has
      shape equal to the batch shape of the policies concatenated with the
      event shape of the policies (which may be scalar, in which case
      concatenation leaves shape just equal to batch shape).
* `action_values`: A Tensor containing estimates of the values of the `actions`.
      Has shape equal to the batch shape of the policies.
* `policy_vars`: An optional iterable of Tensors used by `policies`. If provided
      is used in scope checks. For the multivariate normal example above this
      would be `[mus, sigmas]`.
* `name`: Customises the name_scope for this op.

##### Returns:


* `loss`: Tensor with same shape as `actions` containing the total loss for each
      element in the batch. Differentiable w.r.t the variables in `policies`
      only.


### [`policy_gradient_loss(policies, actions, action_values, policy_vars=None, name='policy_gradient_loss')`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?l=77)<!-- RULE: policy_gradient_loss .code-reference -->

Computes policy gradient losses for a batch of trajectories.

This wraps `policy_gradient` to accept a possibly nested array of `policies`
and `actions` in order to allow for multiple action distribution types or
independent multivariate distributions if not directly available. It also sums
up losses along the time dimension, and is more restrictive about shapes,
assuming a [T, B] layout for the `batch_shape` of the policies and a
concatenate(`[T, B]`, `event_shape` of the policies) shape for the actions.

##### Args:


* `policies`: A (possibly nested structure of) distribution(s) supporting
      `batch_shape` and `event_shape` properties along with a `log_prob`
      method (e.g. an instance of `tfp.distributions.Distribution`),
      with `batch_shape` equal to `[T, B]`.
* `actions`: A (possibly nested structure of) N-D Tensor(s) with shape
      `[T, B, ...]` where the final dimensions are the `event_shape` of the
      corresponding distribution in the nested structure (the shape can be
      just `[T, B]` if the `event_shape` is scalar).
* `action_values`: Tensor of shape `[T, B]` containing an estimate of the value
      of the selected `actions`.
* `policy_vars`: An optional (possibly nested structure of) iterable(s) of
      Tensors used by `policies`. If provided is used in scope checks.
* `name`: Customises the name_scope for this op.

##### Returns:


* `loss`: Tensor of shape `[B]` containing the total loss for each sequence
  in the batch. Differentiable w.r.t `policy_logits` only.


### [`qlambda(q_tm1, a_tm1, r_t, pcont_t, q_t, lambda_, name='GeneralizedQLambda')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=308)<!-- RULE: qlambda .code-reference -->

Implements Peng's and Watkins' Q(lambda) loss as a TensorFlow op.

This function is general enough to implement both Peng's and Watkins'
Q-lambda algorithms.

See "Reinforcement Learning: An Introduction" by Sutton and Barto.
(http://incompleteideas.net/book/ebook/node78.html).

##### Args:


* `q_tm1`: `Tensor` holding a sequence of Q-values starting at the first
    timestep; shape `[T, B, num_actions]`
* `a_tm1`: `Tensor` holding a sequence of action indices, shape `[T, B]`
* `r_t`: Tensor holding a sequence of rewards, shape `[T, B]`
* `pcont_t`: `Tensor` holding a sequence of pcontinue values, shape `[T, B]`
* `q_t`: `Tensor` holding a sequence of Q-values for second timestep;
    shape `[T, B, num_actions]`. In a target network setting,
    this quantity is often supplied by the target network.
* `lambda_`: a scalar or `Tensor` of shape `[T, B]`
    specifying the ratio of mixing between bootstrapped and MC returns;
    if lambda_ is the same for all time steps then the function implements
    Peng's Q-learning algorithm; if lambda_ = 0 at every sub-optimal action
    and a constant otherwise, then the function implements Watkins'
    Q-learning algorithm. Generally lambda_ can be a Tensor of any values
    in the range [0, 1] supplied by the user.
* `name`: a name of the op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[T, B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[T, B]`.
      * `td_error`: batch of temporal difference errors, shape `[T, B]`.


### [`qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t, name='QLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=41)<!-- RULE: qlearning .code-reference -->

Implements the Q-learning loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
the target `r_t + pcont_t * max q_t`.

See "Reinforcement Learning: An Introduction" by Sutton and Barto.
(http://incompleteideas.net/book/ebook/node65.html).

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t`: Tensor holding Q-values for second timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`qv_learning(q_tm1, a_tm1, r_t, pcont_t, v_t, name='QVLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=427)<!-- RULE: qv_learning .code-reference -->

Implements the QV loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
the target `r_t + pcont_t * v_t`, where `v_t` is separately learned through
temporal difference learning (c.f. `value_ops.td_learning`).

See "Two Novel On-policy Reinforcement Learning Algorithms based on
TD(lambda)-methods" by Wiering and van Hasselt
(https://ieeexplore.ieee.org/abstract/document/4220845.)

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `v_t`: Tensor holding state-values for second timestep in a batch of
    transitions, shape `[B]`.
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`qv_max(v_tm1, r_t, pcont_t, q_t, name='QVMAX')`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?l=225)<!-- RULE: qv_max .code-reference -->

Implements the QVMAX learning loss as a TensorFlow op.

The QVMAX loss is `0.5` times the squared difference between `v_tm1` and
the target `r_t + pcont_t * max q_t`, where `q_t` is separately learned
through QV learning (c.f. `action_value_ops.qv_learning`).

See "The QV Family Compared to Other Reinforcement Learning Algorithms" by
Wiering and van Hasselt (2009).
(http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.713.1931)

##### Args:


* `v_tm1`: Tensor holding values at previous timestep, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t`: Tensor of action values at current timestep, shape `[B, num_actions]`.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `v_tm1`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`retrace(lambda_, qs, targnet_qs, actions, rewards, pcontinues, target_policy_probs, behaviour_policy_probs, stop_targnet_gradients=True, name=None)`](https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py?l=45)<!-- RULE: retrace .code-reference -->

Retrace algorithm loss calculation op.

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

##### Args:


* `lambda_`: Positive scalar value or 0-D `Tensor` controlling the degree to
    which future timesteps contribute to the loss computed at each
    transition.
* `qs`: 3-D tensor holding per-action Q-values for the states encountered
    just before taking the transitions that correspond to each major index.
    Since these values are the predicted values we wish to update (in other
    words, the values we intend to change as we learn), in a target network
    setting, these nearly always come from the "non-target" network, which
    we usually call the "learning network".
    Shape is `[(T+1), B, num_actions]`.
* `targnet_qs`: Like `qs`, but in the target network setting, these values
    should be computed by the target network. We use these values to
    compute multi-step error values for timesteps that follow the first
    timesteps in each sequence and sequence fragment we consider.
    Shape is `[(T+1), B, num_actions]`.
* `actions`: 2-D tensor holding the indices of actions executed during the
    transition that corresponds to each major index.
    Shape is `[(T+1), B]`.
* `rewards`: 2-D tensor holding rewards received during the transition
    that corresponds to each major index.
    Shape is `[(T+1), B]`.
* `pcontinues`: 2-D tensor holding pcontinue values received during the
    transition that corresponds to each major index.
    Shape is `[(T+1), B]`.
* `target_policy_probs`: 3-D tensor holding per-action policy probabilities
    for the states encountered just before taking the transitions that
    correspond to each major index, according to the target policy (i.e.
    the policy we wish to learn). These probabilities usually derive from
    the learning net.
    Shape is `[(T+1), B, num_actions]`.
* `behaviour_policy_probs`: 2-D tensor holding the *behaviour* policy's
    probabilities of having taken actions `action` during the transitions
    that correspond to each major index. These probabilities derive from
    whatever policy you used to generate the data.
    Shape is `[(T+1), B]`.
* `stop_targnet_gradients`: `bool` that enables a sensible default way of
    handling gradients through the Retrace op (essentially, gradients
    are not permitted to involve the `targnet_qs` inputs). Can be disabled
    if you require a different arrangement, but you'll probably want to
    block some gradients somewhere.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: Tensor containing the batch of losses, shape `[B]`.
  * `extra`: None


### [`retrace_core(lambda_, q_tm1, a_tm1, r_t, pcont_t, target_policy_t, behaviour_policy_t, targnet_q_t, a_t, stop_targnet_gradients=True, name=None)`](https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py?l=292)<!-- RULE: retrace_core .code-reference -->

Retrace algorithm core loss calculation op.

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

##### Args:


* `lambda_`: Positive scalar value or 0-D `Tensor` controlling the degree to
    which future timesteps contribute to the loss computed at each
    transition.
* `q_tm1`: 3-D tensor holding per-action Q-values for the states encountered
    just before taking the transitions that correspond to each major index.
    Since these values are the predicted values we wish to update (in other
    words, the values we intend to change as we learn), in a target network
    setting, these nearly always come from the "non-target" network, which
    we usually call the "learning network".
    Shape is `[T, B, num_actions]`.
* `a_tm1`: 2-D tensor holding the indices of actions executed during the
    transition that corresponds to each major index.
    Shape is `[T, B]`.
* `r_t`: 2-D tensor holding rewards received during the transition
    that corresponds to each major index.
    Shape is `[T, B]`.
* `pcont_t`: 2-D tensor holding pcontinue values received during the
    transition that corresponds to each major index.
    Shape is `[T, B]`.
* `target_policy_t`: 3-D tensor holding per-action policy probabilities for
    the states encountered just AFTER the transitions that correspond to
    each major index, according to the target policy (i.e. the policy we
    wish to learn). These usually derive from the learning net.
    Shape is `[T, B, num_actions]`.
* `behaviour_policy_t`: 2-D tensor holding the *behaviour* policy's
    probabilities of having taken action `a_t` at the states encountered
    just AFTER the transitions that correspond to each major index. Derived
    from whatever policy you used to generate the data. All values MUST be
    greater that 0. Shape is `[T, B]`.
* `targnet_q_t`: 3-D tensor holding per-action Q-values for the states
    encountered just AFTER taking the transitions that correspond to each
    major index. Since these values are used to calculate target values for
    the network, in a target in a target network setting, these should
    probably come from the target network.
    Shape is `[T, B, num_actions]`.
* `a_t`: 2-D tensor holding the indices of actions executed during the
    transition AFTER the transition that corresponds to each major index.
    Shape is `[T, B]`.
* `stop_targnet_gradients`: `bool` that enables a sensible default way of
    handling gradients through the Retrace op (essentially, gradients
    are not permitted to involve the `targnet_q_t` input).
    Can be disabled if you require a different arragement, but
    you'll probably want to block some gradients somewhere.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: Tensor containing the batch of losses, shape `[B]`.
  * `extra`: A namedtuple with fields:
      * `retrace_weights`: Tensor containing batch of retrace weights,
      shape `[T, B]`.
      * `target`: Tensor containing target action values, shape `[T, B]`.


### [`sarsa(q_tm1, a_tm1, r_t, pcont_t, q_t, a_t, name='Sarsa')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=197)<!-- RULE: sarsa .code-reference -->

Implements the SARSA loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
the target `r_t + pcont_t * q_t[a_t]`.

See "Reinforcement Learning: An Introduction" by Sutton and Barto.
(http://incompleteideas.net/book/ebook/node64.html.)

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t`: Tensor holding Q-values for second timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_t`: Tensor holding action indices for second timestep, shape `[B]`.
* `name`: name to prefix ops created within this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`sarsa_lambda(q_tm1, a_tm1, r_t, pcont_t, q_t, a_t, lambda_, name='SarsaLambda')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=371)<!-- RULE: sarsa_lambda .code-reference -->

Implements SARSA(lambda) loss as a TensorFlow op.

See "Reinforcement Learning: An Introduction" by Sutton and Barto.
(http://incompleteideas.net/book/ebook/node77.html).

##### Args:


* `q_tm1`: `Tensor` holding a sequence of Q-values starting at the first
    timestep; shape `[T, B, num_actions]`
* `a_tm1`: `Tensor` holding a sequence of action indices, shape `[T, B]`
* `r_t`: Tensor holding a sequence of rewards, shape `[T, B]`
* `pcont_t`: `Tensor` holding a sequence of pcontinue values, shape `[T, B]`
* `q_t`: `Tensor` holding a sequence of Q-values for second timestep;
    shape `[T, B, num_actions]`.
* `a_t`: `Tensor` holding a sequence of action indices for second timestep;
    shape `[T, B]`
* `lambda_`: a scalar specifying the ratio of mixing between bootstrapped and
    MC returns.
* `name`: a name of the op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[T, B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[T, B]`.
      * `td_error`: batch of temporal difference errors, shape `[T, B]`.


### [`sarse(q_tm1, a_tm1, r_t, pcont_t, q_t, probs_a_t, debug=False, name='Sarse')`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?l=244)<!-- RULE: sarse .code-reference -->

Implements the SARSE (Expected SARSA) loss as a TensorFlow op.

The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
the target `r_t + pcont_t * (sum_a probs_a_t[a] * q_t[a])`.

See "A Theoretical and Empirical Analysis of Expected Sarsa" by Seijen,
van Hasselt, Whiteson et al.
(http://www.cs.ox.ac.uk/people/shimon.whiteson/pubs/vanseijenadprl09.pdf).

##### Args:


* `q_tm1`: Tensor holding Q-values for first timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `a_tm1`: Tensor holding action indices, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `q_t`: Tensor holding Q-values for second timestep in a batch of
    transitions, shape `[B x num_actions]`.
* `probs_a_t`: Tensor holding action probabilities for second timestep,
    shape `[B x num_actions]`.
* `debug`: Boolean flag, when set to True adds ops to check whether probs_a_t
    is a batch of (approximately) valid probability distributions.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`scan_discounted_sum(sequence, decay, initial_value, reverse=False, sequence_lengths=None, back_prop=True, name='scan_discounted_sum')`](https://github.com/deepmind/trfl/blob/master/trfl/sequence_ops.py?l=47)<!-- RULE: scan_discounted_sum .code-reference -->

Evaluates a cumulative discounted sum along dimension 0.

  ```python
  if reverse = False:
    result[1] = sequence[1] + decay[1] * initial_value
    result[k] = sequence[k] + decay[k] * result[k - 1]
  if reverse = True:
    result[last] = sequence[last] + decay[last] * initial_value
    result[k] = sequence[k] + decay[k] * result[k + 1]
  ```

Respective dimensions T, B and ... have to be the same for all input tensors.
T: temporal dimension of the sequence; B: batch dimension of the sequence.

  if sequence_lengths is set then x1 and x2 below are equivalent:
  ```python
  x1 = zero_pad_to_length(
    scan_discounted_sum(
        sequence[:length], decays[:length], **kwargs), length=T)
  x2 = scan_discounted_sum(sequence, decays,
                           sequence_lengths=[length], **kwargs)
  ```

##### Args:


* `sequence`: Tensor of shape `[T, B, ...]` containing values to be summed.
* `decay`: Tensor of shape `[T, B, ...]` containing decays/discounts.
* `initial_value`: Tensor of shape `[B, ...]` containing initial value.
* `reverse`: Whether to process the sum in a reverse order.
* `sequence_lengths`: Tensor of shape `[B]` containing sequence lengths to be
    (reversed and then) summed.
* `back_prop`: Whether to backpropagate.
* `name`: Sets the name_scope for this op.

##### Returns:

  Cumulative sum with discount. Same shape and type as `sequence`.


### [`sequence_a2c_loss(policies, baseline_values, actions, rewards, pcontinues, bootstrap_value, policy_vars=None, lambda_=1, entropy_cost=None, baseline_cost=1, entropy_scale_op=None, name='SequenceA2CLoss')`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?l=200)<!-- RULE: sequence_a2c_loss .code-reference -->

Constructs a TensorFlow graph computing the A2C/GAE loss for sequences.

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

##### Args:


* `policies`: A (possibly nested structure of) distribution(s) supporting
      `batch_shape` and `event_shape` properties & `log_prob` and `entropy`
      methods (e.g. an instance of `tfp.distributions.Distribution`),
      with `batch_shape` equal to `[T, B]`. E.g. for a (non-nested) diagonal
      multivariate gaussian with dimension `A` this would be:
      `policies = tfp.distributions.MultivariateNormalDiag(mus, sigmas)`
      where `mus` and `sigmas` have shape `[T, B, A]`.
* `baseline_values`: 2-D Tensor containing an estimate of the state value with
      shape `[T, B]`.
* `actions`: A (possibly nested structure of) N-D Tensor(s) with shape
      `[T, B, ...]` where the final dimensions are the `event_shape` of the
      corresponding distribution in the nested structure (the shape can be
      just `[T, B]` if the `event_shape` is scalar).
* `rewards`: 2-D Tensor with shape `[T, B]`.
* `pcontinues`: 2-D Tensor with shape `[T, B]`.
* `bootstrap_value`: 1-D Tensor with shape `[B]`.
* `policy_vars`: An optional (possibly nested structure of) iterables of
      Tensors used by `policies`. If provided is used in scope checks. For
      the multivariate normal example above this would be `[mus, sigmas]`.
* `lambda_`: an optional scalar or 2-D Tensor with shape `[T, B]` for
      Generalised Advantage Estimation as per
      https://arxiv.org/abs/1506.02438.
* `entropy_cost`: optional scalar cost that pushes the policy to have high
      entropy, larger values cause higher entropies.
* `baseline_cost`: scalar cost that scales the derivatives of the baseline
      relative to the policy gradient.
* `entropy_scale_op`: An optional op that takes `policies` as its only
      argument and returns a scalar Tensor that is used to scale the entropy
      loss. E.g. for Diag(sigma) Gaussian policies dividing by the number of
      dimensions makes entropy loss invariant to the action space dimension.
      See `policy_entropy_loss` for more info.
* `name`: Customises the name_scope for this op.

##### Returns:

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


### [`sequence_advantage_actor_critic_loss(policy_logits, baseline_values, actions, rewards, pcontinues, bootstrap_value, lambda_=1, entropy_cost=None, baseline_cost=1, normalise_entropy=False, name='SequenceAdvantageActorCriticLoss')`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?l=99)<!-- RULE: sequence_advantage_actor_critic_loss .code-reference -->

Calculates the loss for an A2C update along a batch of trajectories.

Technically A2C is the special case where lambda=1; for general lambda
this is the loss for Generalized Advantage Estimation (GAE), modulo chunking
behaviour if passing chunks of episodes (see `generalized_lambda_returns` for
more detail).

Note: This function takes policy _logits_ as input, not the log-policy like
`learning.deepmind.lua.rl.learners.Reinforce` does.

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

##### Args:


* `policy_logits`: A (possibly nested structure of) 3-D Tensor(s) with shape
      `[T, B, num_actions]` and possibly different dimension `num_actions`.
* `baseline_values`: 2-D Tensor containing an estimate of state values `[T, B]`.
* `actions`: A (possibly nested structure of) 2-D Tensor(s) with shape
      `[T, B]` and integer type.
* `rewards`: 2-D Tensor with shape `[T, B]`.
* `pcontinues`: 2-D Tensor with shape `[T, B]`.
* `bootstrap_value`: 1-D Tensor with shape `[B]`.
* `lambda_`: an optional scalar or 2-D Tensor with shape `[T, B]` for
      Generalised Advantage Estimation as per
      https://arxiv.org/abs/1506.02438.
* `entropy_cost`: optional scalar cost that pushes the policy to have high
      entropy, larger values cause higher entropies.
* `baseline_cost`: scalar cost that scales the derivatives of the baseline
      relative to the policy gradient.
* `normalise_entropy`: if True, the entropy loss is normalised to the range
      `[-1, 0]` by dividing by the log number of actions. This makes it more
      invariant to the size of the action space. Default is False.
* `name`: Customises the name_scope for this op.

##### Returns:

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


### [`td_lambda(state_values, rewards, pcontinues, bootstrap_value, lambda_=1, name='BaselineLoss')`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?l=160)<!-- RULE: td_lambda .code-reference -->

Constructs a TensorFlow graph computing the L2 loss for sequences.

This loss learns the baseline for advantage actor-critic models. Gradients
for this loss flow through each tensor in `state_values`, but no other
input tensors. The baseline is regressed towards the n-step bootstrapped
returns given by the reward/pcontinue sequence.

This function is designed for batches of sequences of data. Tensors are
assumed to be time major (i.e. the outermost dimension is time, the second
outermost dimension is the batch dimension). We denote the sequence length
in the shapes of the arguments with the variable `T`, the batch size with
the variable `B`, neither of which needs to be known at construction time.
Index `0` of the time dimension is assumed to be the start of the sequence.

`rewards` and `pcontinues` are the sequences of data taken directly from the
environment, possibly modulated by a discount. `state_values` are the
sequences of (typically learnt) estimates of the values of the states
visited along a batch of trajectories.

The sequences in the tensors should be aligned such that an agent in a state
with value `V` that takes an action transitions into another state
with value `V'`, receiving reward `r` and pcontinue `p`. Then `V`, `r`
and `p` are all at the same index `i` in the corresponding tensors. `V'` is
at index `i+1`, or in the `bootstrap_value` tensor if `i == T`.

See "High-dimensional continuous control using generalized advantage
estimation" by Schulman, Moritz, Levine et al.
(https://arxiv.org/abs/1506.02438).

##### Args:


* `state_values`: 2-D Tensor of state-value estimates with shape `[T, B]`.
* `rewards`: 2-D Tensor with shape `[T, B]`.
* `pcontinues`: 2-D Tensor with shape `[T, B]`.
* `bootstrap_value`: 1-D Tensor with shape `[B]`.
* `lambda_`: an optional scalar or 2-D Tensor with shape `[T, B]`.
* `name`: Customises the name_scope for this op.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * temporal_differences, Tensor of shape `[T, B]`
      * discounted_returns, Tensor of shape `[T, B]`


### [`td_learning(v_tm1, r_t, pcont_t, v_t, name='TDLearning')`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?l=35)<!-- RULE: td_learning .code-reference -->

Implements the TD(0)-learning loss as a TensorFlow op.

The TD loss is `0.5` times the squared difference between `v_tm1` and
the target `r_t + pcont_t * v_t`.

See "Learning to Predict by the Methods of Temporal Differences" by Sutton.
(https://link.springer.com/article/10.1023/A:1022633531479).

##### Args:


* `v_tm1`: Tensor holding values at previous timestep, shape `[B]`.
* `r_t`: Tensor holding rewards, shape `[B]`.
* `pcont_t`: Tensor holding pcontinue values, shape `[B]`.
* `v_t`: Tensor holding values at current timestep, shape `[B]`.
* `name`: name to prefix ops created by this function.

##### Returns:

  A namedtuple with fields:

  * `loss`: a tensor containing the batch of losses, shape `[B]`.
  * `extra`: a namedtuple with fields:
      * `target`: batch of target values for `v_tm1`, shape `[B]`.
      * `td_error`: batch of temporal difference errors, shape `[B]`.


### [`update_target_variables(target_variables, source_variables, tau=1.0, use_locking=False, name='update_target_variables')`](https://github.com/deepmind/trfl/blob/master/trfl/target_update_ops.py?l=32)<!-- RULE: update_target_variables .code-reference -->

Returns an op to update a list of target variables from source variables.

The update rule is:
`target_variable = (1 - tau) * target_variable + tau * source_variable`.

##### Args:


* `target_variables`: a list of the variables to be updated.
* `source_variables`: a list of the variables used for the update.
* `tau`: weight used to gate the update. The permitted range is 0 < tau <= 1,
    with small tau representing an incremental update, and tau == 1
    representing a full update (that is, a straight copy).
* `use_locking`: use `tf.Variable.assign`'s locking option when assigning
    source variable values to target variables.
* `name`: sets the `name_scope` for this op.

##### Raises:


* `TypeError`: when tau is not a Python float
* `ValueError`: when tau is out of range, or the source and target variables
    have different numbers or shapes.

##### Returns:

  An op that executes all the variable updates.


### [`vtrace_from_importance_weights(log_rhos, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, name='vtrace_from_importance_weights')`](https://github.com/deepmind/trfl/blob/master/trfl/vtrace_ops.py?l=154)<!-- RULE: vtrace_from_importance_weights .code-reference -->

V-trace from log importance weights.

Calculates V-trace actor critic targets as described in

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

In the notation used throughout documentation and comments, T refers to the
time dimension ranging from 0 to T-1. B refers to the batch size. This code
also supports the case where all tensors have the same number of additional
dimensions, e.g., `rewards` is `[T, B, C]`, `values` is `[T, B, C]`,
`bootstrap_value` is `[B, C]`.

##### Args:


* `log_rhos`: A float32 tensor of shape `[T, B]` representing the
    log importance sampling weights, i.e.
    log(target_policy(a) / behaviour_policy(a)). V-trace performs operations
    on rhos in log-space for numerical stability.
* `discounts`: A float32 tensor of shape `[T, B]` with discounts encountered
    when following the behaviour policy.
* `rewards`: A float32 tensor of shape `[T, B]` containing rewards generated by
    following the behaviour policy.
* `values`: A float32 tensor of shape `[T, B]` with the value function estimates
    wrt. the target policy.
* `bootstrap_value`: A float32 of shape `[B]` with the value function estimate
    at time T.
* `clip_rho_threshold`: A scalar float32 tensor with the clipping threshold for
    importance weights (rho) when calculating the baseline targets (vs).
    rho^bar in the paper. If None, no clipping is applied.
* `clip_pg_rho_threshold`: A scalar float32 tensor with the clipping threshold
    on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)). If
    None, no clipping is applied.
* `name`: The name scope that all V-trace operations will be created in.

##### Returns:

  A VTraceReturns namedtuple (vs, pg_advantages) where:

* `vs`: A float32 tensor of shape `[T, B]`. Can be used as target to
      train a baseline (V(x_t) - vs_t)^2.
* `pg_advantages`: A float32 tensor of shape `[T, B]`. Can be used as the
      advantage in the calculation of policy gradients.


### [`vtrace_from_logits(behaviour_policy_logits, target_policy_logits, actions, discounts, rewards, values, bootstrap_value, clip_rho_threshold=1.0, clip_pg_rho_threshold=1.0, name='vtrace_from_logits')`](https://github.com/deepmind/trfl/blob/master/trfl/vtrace_ops.py?l=61)<!-- RULE: vtrace_from_logits .code-reference -->

V-trace for softmax policies.

Calculates V-trace actor critic targets for softmax polices as described in

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

Target policy refers to the policy we are interested in improving and
behaviour policy refers to the policy that generated the given
rewards and actions.

In the notation used throughout documentation and comments, `T` refers to the
time dimension ranging from `0` to `T-1`. `B` refers to the batch size and
`NUM_ACTIONS` refers to the number of actions.

##### Args:


* `behaviour_policy_logits`: A float32 tensor of shape `[T, B, NUM_ACTIONS]`
    with un-normalized log-probabilities parametrizing the softmax behaviour
    policy.
* `target_policy_logits`: A float32 tensor of shape `[T, B, NUM_ACTIONS]` with
    un-normalized log-probabilities parametrizing the softmax target policy.
* `actions`: An int32 tensor of shape `[T, B]` of actions sampled from the
    behaviour policy.
* `discounts`: A float32 tensor of shape `[T, B]` with the discount encountered
    when following the behaviour policy.
* `rewards`: A float32 tensor of shape `[T, B]` with the rewards generated by
    following the behaviour policy.
* `values`: A float32 tensor of shape `[T, B]` with the value function estimates
    wrt. the target policy.
* `bootstrap_value`: A float32 of shape `[B]` with the value function estimate
    at time T.
* `clip_rho_threshold`: A scalar float32 tensor with the clipping threshold for
    importance weights (rho) when calculating the baseline targets (vs).
    rho^bar in the paper.
* `clip_pg_rho_threshold`: A scalar float32 tensor with the clipping threshold
    on rho_s in \rho_s \delta log \pi(a|x) (r + \gamma v_{s+1} - V(x_s)).
* `name`: The name scope that all V-trace operations will be created in.

##### Returns:

  A `VTraceFromLogitsReturns` namedtuple with the following fields:

* `vs`: A float32 tensor of shape `[T, B]`. Can be used as target to train a
        baseline (V(x_t) - vs_t)^2.
* `pg_advantages`: A float 32 tensor of shape `[T, B]`. Can be used as an
      estimate of the advantage in the calculation of policy gradients.
* `log_rhos`: A float32 tensor of shape `[T, B]` containing the log importance
      sampling weights (log rhos).
* `behaviour_action_log_probs`: A float32 tensor of shape `[T, B]` containing
      behaviour policy action log probabilities (log \mu(a_t)).
* `target_action_log_probs`: A float32 tensor of shape `[T, B]` containing
      target policy action probabilities (log \pi(a_t)).


### [`class action_value_ops.DoubleQExtra`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?q=class:double_qlearning_extra)<!-- RULE: action_value_ops.DoubleQExtra .code-reference -->

double_qlearning_extra(target, td_error, best_action)

#### `action_value_ops.DoubleQExtra.best_action`<!-- RULE: action_value_ops.DoubleQExtra.best_action .code-reference -->

Alias for field number 2


#### `action_value_ops.DoubleQExtra.target`<!-- RULE: action_value_ops.DoubleQExtra.target .code-reference -->

Alias for field number 0


#### `action_value_ops.DoubleQExtra.td_error`<!-- RULE: action_value_ops.DoubleQExtra.td_error .code-reference -->

Alias for field number 1



### [`class action_value_ops.QExtra`](https://github.com/deepmind/trfl/blob/master/trfl/action_value_ops.py?q=class:qlearning_extra)<!-- RULE: action_value_ops.QExtra .code-reference -->

qlearning_extra(target, td_error)

#### `action_value_ops.QExtra.target`<!-- RULE: action_value_ops.QExtra.target .code-reference -->

Alias for field number 0


#### `action_value_ops.QExtra.td_error`<!-- RULE: action_value_ops.QExtra.td_error .code-reference -->

Alias for field number 1



### [`class base_ops.LossOutput`](https://github.com/deepmind/trfl/blob/master/trfl/base_ops.py?q=class:loss_output)<!-- RULE: base_ops.LossOutput .code-reference -->

loss_output(loss, extra)

#### `base_ops.LossOutput.extra`<!-- RULE: base_ops.LossOutput.extra .code-reference -->

Alias for field number 1


#### `base_ops.LossOutput.loss`<!-- RULE: base_ops.LossOutput.loss .code-reference -->

Alias for field number 0



### [`base_ops.assert_arg_bounded(value, min_value, max_value, op_name, arg_name)`](https://github.com/deepmind/trfl/blob/master/trfl/base_ops.py?l=100)<!-- RULE: base_ops.assert_arg_bounded .code-reference -->




### [`base_ops.wrap_rank_shape_assert(tensors_list, expected_ranks, op_name)`](https://github.com/deepmind/trfl/blob/master/trfl/base_ops.py?l=89)<!-- RULE: base_ops.wrap_rank_shape_assert .code-reference -->




### [`class discrete_policy_gradient_ops.DiscretePolicyEntropyExtra`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?q=class:discrete_policy_entropy_extra)<!-- RULE: discrete_policy_gradient_ops.DiscretePolicyEntropyExtra .code-reference -->

discrete_policy_entropy_extra(entropy,)

#### `discrete_policy_gradient_ops.DiscretePolicyEntropyExtra.entropy`<!-- RULE: discrete_policy_gradient_ops.DiscretePolicyEntropyExtra.entropy .code-reference -->

Alias for field number 0



### [`class discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra`](https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py?q=class:sequence_advantage_actor_critic_extra)<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra .code-reference -->

sequence_advantage_actor_critic_extra(entropy, entropy_loss, baseline_loss, policy_gradient_loss, advantages, discounted_returns)

#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.advantages`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.advantages .code-reference -->

Alias for field number 4


#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.baseline_loss`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.baseline_loss .code-reference -->

Alias for field number 2


#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.discounted_returns`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.discounted_returns .code-reference -->

Alias for field number 5


#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.entropy`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.entropy .code-reference -->

Alias for field number 0


#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.entropy_loss`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.entropy_loss .code-reference -->

Alias for field number 1


#### `discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.policy_gradient_loss`<!-- RULE: discrete_policy_gradient_ops.SequenceAdvantageActorCriticExtra.policy_gradient_loss .code-reference -->

Alias for field number 3



### [`class dist_value_ops.Extra`](https://github.com/deepmind/trfl/blob/master/trfl/dist_value_ops.py?q=class:dist_value_extra)<!-- RULE: dist_value_ops.Extra .code-reference -->

dist_value_extra(target,)

#### `dist_value_ops.Extra.target`<!-- RULE: dist_value_ops.Extra.target .code-reference -->

Alias for field number 0



### [`distribution_ops.factorised_kl_gaussian(dist1_mean, dist1_covariance_or_scale, dist2_mean, dist2_covariance_or_scale, both_diagonal=False)`](https://github.com/deepmind/trfl/blob/master/trfl/distribution_ops.py?l=71)<!-- RULE: distribution_ops.factorised_kl_gaussian .code-reference -->

Compute the KL divergence KL(dist1, dist2) between two Gaussians.

The KL is factorised into two terms - `kl_mean` and `kl_cov`. This
factorisation is specific to multivariate gaussian distributions and arises
from its analytic form.
Specifically, if we assume two multivariate Gaussian distributions with rank
k and means, M1 and M2 and variance S1 and S2, the analytic KL can be written
out as:

D_KL(N0 || N1) = 0.5 * (tr(inv(S1) * S0) + ln(det(S1)/det(S0)) - k +
                       (M1 - M0).T * inv(S1) * (M1 - M0))

The terms on the first row correspond to the covariance factor and the terms
on the second row correspond to the mean factor in the factorized KL.
These terms can thus be used to independently control how much the mean and
covariance between the two gaussians can vary.

This implementation ensures that gradient flow is equivalent to calling
`tfp.distributions.kl_divergence` once.

More details on the equation can be found here:
https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians


##### Args:


* `dist1_mean`: The mean of the first Multivariate Gaussian distribution.
* `dist1_covariance_or_scale`: The covariance or scale of the first Multivariate
    Gaussian distribution. In cases where *both* distributions are Gaussians
    with diagonal covariance matrices (for instance, if both are instances of
    `tfp.distributions.MultivariateNormalDiag`), then the `scale` can be
    passed in instead and the `both_diagonal` flag must be set to `True`.
    A more efficient sparse computation path is used in this case. For all
    other cases, the full covariance matrix must be passed in.
* `dist2_mean`: The mean of the second Multivariate Gaussian distribution.
* `dist2_covariance_or_scale`: The covariance or scale tensor of the second
    Multivariate Gaussian distribution, as for `dist1_covariance_or_scale`.
* `both_diagonal`: A `bool` indicating that both dist1 and dist2 are diagonal
    matrices. A more efficient sparse computation is used in this case.

##### Returns:

  A tuple consisting of (`kl_mean`, `kl_cov`) which correspond to the mean and
  the covariance factorisation of the KL.


### [`distribution_ops.hard_cumulative_project(support, weights, new_support, reverse)`](https://github.com/deepmind/trfl/blob/master/trfl/distribution_ops.py?l=50)<!-- RULE: distribution_ops.hard_cumulative_project .code-reference -->

Produces a cumulative categorical distribution on a new support.

##### Args:


* `support`: Tensor defining support of a categorical distribution(s). Must be
    of rank 1 or of the same rank as `weights`. The size of the last dimension
    has to match that of `weights`.
* `weights`: Tensor defining weights on the support points.
* `new_support`: Tensor holding positions of a new support.
* `reverse`: Whether to evalute cumulative from the left or right.

##### Returns:

  Cumulative distribution on the supplied support.
  The foolowing invariant is maintained across the last dimension:
  result[i] = (sum_j weights[j] for all j where support[j] < new_support[i])
              if reverse == False else
              (sum_j weights[j] for all j where support[j] > new_support[i])


### [`distribution_ops.l2_project(support, weights, new_support)`](https://github.com/deepmind/trfl/blob/master/trfl/distribution_ops.py?l=34)<!-- RULE: distribution_ops.l2_project .code-reference -->

Projects distribution (support, weights) onto new_support.

##### Args:


* `support`: Tensor defining support of a categorical distribution(s). Must be
    of rank 1 or of the same rank as `weights`. The size of the last dimension
    has to match that of `weights`.
* `weights`: Tensor defining weights on the support points.
* `new_support`: Tensor holding positions of a new support.

##### Returns:

  Projection of (support, weights) onto the new_support.


### [`class dpg_ops.DPGExtra`](https://github.com/deepmind/trfl/blob/master/trfl/dpg_ops.py?q=class:dpg_extra)<!-- RULE: dpg_ops.DPGExtra .code-reference -->

dpg_extra(q_max, a_max, dqda)

#### `dpg_ops.DPGExtra.a_max`<!-- RULE: dpg_ops.DPGExtra.a_max .code-reference -->

Alias for field number 1


#### `dpg_ops.DPGExtra.dqda`<!-- RULE: dpg_ops.DPGExtra.dqda .code-reference -->

Alias for field number 2


#### `dpg_ops.DPGExtra.q_max`<!-- RULE: dpg_ops.DPGExtra.q_max .code-reference -->

Alias for field number 0



### [`gen_distribution_ops.ProjectDistribution(support, weights, new_support, method, name=None)`](https://github.com/deepmind/trfl/blob/master/trfl/gen_distribution_ops.py?l=398)<!-- RULE: gen_distribution_ops.ProjectDistribution .code-reference -->

Projects one categorical distribution onto another.

##### Args:


* `support`: A `Tensor` of type `float32`.
* `weights`: A `Tensor` of type `float32`.
* `new_support`: A `Tensor` of type `float32`.
* `method`: A `Tensor` of type `int32`.
* `name`: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


### [`gen_distribution_ops.project_distribution(support, weights, new_support, method, name=None)`](https://github.com/deepmind/trfl/blob/master/trfl/gen_distribution_ops.py?l=24)<!-- RULE: gen_distribution_ops.project_distribution .code-reference -->

Projects one categorical distribution onto another.

##### Args:


* `support`: A `Tensor` of type `float32`.
* `weights`: A `Tensor` of type `float32`.
* `new_support`: A `Tensor` of type `float32`.
* `method`: A `Tensor` of type `int32`.
* `name`: A name for the operation (optional).

##### Returns:

  A `Tensor` of type `float32`.


### [`gen_distribution_ops.project_distribution_eager_fallback(support, weights, new_support, method, name, ctx)`](https://github.com/deepmind/trfl/blob/master/trfl/gen_distribution_ops.py?l=92)<!-- RULE: gen_distribution_ops.project_distribution_eager_fallback .code-reference -->




### [`indexing_ops.assert_compatible_shapes(value_shape, index_shape)`](https://github.com/deepmind/trfl/blob/master/trfl/indexing_ops.py?l=32)<!-- RULE: indexing_ops.assert_compatible_shapes .code-reference -->

Check shapes of the indices and the tensor to be indexed.

If all input shapes are known statically, obtain shapes of arguments and
perform compatibility checks. Otherwise, print a warning. The only check
we cannot perform statically (and do not attempt elsewhere) is making
sure that each action index in actions is in [0, num_actions).

##### Args:


* `value_shape`: static shape of the values.
* `index_shape`: static shape of the indices.


### [`class pixel_control_ops.PixelControlExtra`](https://github.com/deepmind/trfl/blob/master/trfl/pixel_control_ops.py?q=class:pixel_control_extra)<!-- RULE: pixel_control_ops.PixelControlExtra .code-reference -->

pixel_control_extra(spatial_loss, pseudo_rewards)

#### `pixel_control_ops.PixelControlExtra.pseudo_rewards`<!-- RULE: pixel_control_ops.PixelControlExtra.pseudo_rewards .code-reference -->

Alias for field number 1


#### `pixel_control_ops.PixelControlExtra.spatial_loss`<!-- RULE: pixel_control_ops.PixelControlExtra.spatial_loss .code-reference -->

Alias for field number 0



### [`class policy_gradient_ops.PolicyEntropyExtra`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?q=class:policy_entropy_extra)<!-- RULE: policy_gradient_ops.PolicyEntropyExtra .code-reference -->

policy_entropy_extra(entropy,)

#### `policy_gradient_ops.PolicyEntropyExtra.entropy`<!-- RULE: policy_gradient_ops.PolicyEntropyExtra.entropy .code-reference -->

Alias for field number 0



### [`class policy_gradient_ops.SequenceA2CExtra`](https://github.com/deepmind/trfl/blob/master/trfl/policy_gradient_ops.py?q=class:sequence_a2c_extra)<!-- RULE: policy_gradient_ops.SequenceA2CExtra .code-reference -->

sequence_a2c_extra(entropy, entropy_loss, baseline_loss, policy_gradient_loss, advantages, discounted_returns)

#### `policy_gradient_ops.SequenceA2CExtra.advantages`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.advantages .code-reference -->

Alias for field number 4


#### `policy_gradient_ops.SequenceA2CExtra.baseline_loss`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.baseline_loss .code-reference -->

Alias for field number 2


#### `policy_gradient_ops.SequenceA2CExtra.discounted_returns`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.discounted_returns .code-reference -->

Alias for field number 5


#### `policy_gradient_ops.SequenceA2CExtra.entropy`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.entropy .code-reference -->

Alias for field number 0


#### `policy_gradient_ops.SequenceA2CExtra.entropy_loss`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.entropy_loss .code-reference -->

Alias for field number 1


#### `policy_gradient_ops.SequenceA2CExtra.policy_gradient_loss`<!-- RULE: policy_gradient_ops.SequenceA2CExtra.policy_gradient_loss .code-reference -->

Alias for field number 3



### [`class retrace_ops.RetraceCoreExtra`](https://github.com/deepmind/trfl/blob/master/trfl/retrace_ops.py?q=class:retrace_core_extra)<!-- RULE: retrace_ops.RetraceCoreExtra .code-reference -->

retrace_core_extra(retrace_weights, target)

#### `retrace_ops.RetraceCoreExtra.retrace_weights`<!-- RULE: retrace_ops.RetraceCoreExtra.retrace_weights .code-reference -->

Alias for field number 0


#### `retrace_ops.RetraceCoreExtra.target`<!-- RULE: retrace_ops.RetraceCoreExtra.target .code-reference -->

Alias for field number 1



### [`class value_ops.TDExtra`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?q=class:td_extra)<!-- RULE: value_ops.TDExtra .code-reference -->

td_extra(target, td_error)

#### `value_ops.TDExtra.target`<!-- RULE: value_ops.TDExtra.target .code-reference -->

Alias for field number 0


#### `value_ops.TDExtra.td_error`<!-- RULE: value_ops.TDExtra.td_error .code-reference -->

Alias for field number 1



### [`class value_ops.TDLambdaExtra`](https://github.com/deepmind/trfl/blob/master/trfl/value_ops.py?q=class:td_lambda_extra)<!-- RULE: value_ops.TDLambdaExtra .code-reference -->

td_lambda_extra(temporal_differences, discounted_returns)

#### `value_ops.TDLambdaExtra.discounted_returns`<!-- RULE: value_ops.TDLambdaExtra.discounted_returns .code-reference -->

Alias for field number 1


#### `value_ops.TDLambdaExtra.temporal_differences`<!-- RULE: value_ops.TDLambdaExtra.temporal_differences .code-reference -->

Alias for field number 0



### [`class vtrace_ops.VTraceFromLogitsReturns`](https://github.com/deepmind/trfl/blob/master/trfl/vtrace_ops.py?q=class:VTraceFromLogitsReturns)<!-- RULE: vtrace_ops.VTraceFromLogitsReturns .code-reference -->

VTraceFromLogitsReturns(vs, pg_advantages, log_rhos, behaviour_action_log_probs, target_action_log_probs)

#### `vtrace_ops.VTraceFromLogitsReturns.behaviour_action_log_probs`<!-- RULE: vtrace_ops.VTraceFromLogitsReturns.behaviour_action_log_probs .code-reference -->

Alias for field number 3


#### `vtrace_ops.VTraceFromLogitsReturns.log_rhos`<!-- RULE: vtrace_ops.VTraceFromLogitsReturns.log_rhos .code-reference -->

Alias for field number 2


#### `vtrace_ops.VTraceFromLogitsReturns.pg_advantages`<!-- RULE: vtrace_ops.VTraceFromLogitsReturns.pg_advantages .code-reference -->

Alias for field number 1


#### `vtrace_ops.VTraceFromLogitsReturns.target_action_log_probs`<!-- RULE: vtrace_ops.VTraceFromLogitsReturns.target_action_log_probs .code-reference -->

Alias for field number 4


#### `vtrace_ops.VTraceFromLogitsReturns.vs`<!-- RULE: vtrace_ops.VTraceFromLogitsReturns.vs .code-reference -->

Alias for field number 0



### [`class vtrace_ops.VTraceReturns`](https://github.com/deepmind/trfl/blob/master/trfl/vtrace_ops.py?q=class:VTraceReturns)<!-- RULE: vtrace_ops.VTraceReturns .code-reference -->

VTraceReturns(vs, pg_advantages)

#### `vtrace_ops.VTraceReturns.pg_advantages`<!-- RULE: vtrace_ops.VTraceReturns.pg_advantages .code-reference -->

Alias for field number 1


#### `vtrace_ops.VTraceReturns.vs`<!-- RULE: vtrace_ops.VTraceReturns.vs .code-reference -->

Alias for field number 0



### [`vtrace_ops.log_probs_from_logits_and_actions(policy_logits, actions)`](https://github.com/deepmind/trfl/blob/master/trfl/vtrace_ops.py?l=35)<!-- RULE: vtrace_ops.log_probs_from_logits_and_actions .code-reference -->

Computes action log-probs from policy logits and actions.

In the notation used throughout documentation and comments, T refers to the
time dimension ranging from 0 to T-1. B refers to the batch size and
NUM_ACTIONS refers to the number of actions.

##### Args:


* `policy_logits`: A float32 tensor of shape `[T, B, NUM_ACTIONS]` with
    un-normalized log-probabilities parameterizing a softmax policy.
* `actions`: An int32 tensor of shape `[T, B]` with actions.

##### Returns:

  A float32 tensor of shape `[T, B]` corresponding to the sampling log
  probability of the chosen action w.r.t. the policy.


