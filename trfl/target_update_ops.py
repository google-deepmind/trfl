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
"""Tensorflow ops for updating target networks.

Tensorflow ops that are used to update a target network from a source network.
This is used in agents such as DQN or DPG, which use a target network that
changes more slowly than the online network, in order to improve stability.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf
from trfl import periodic_ops


def update_target_variables(target_variables,
                            source_variables,
                            tau=1.0,
                            use_locking=False,
                            name="update_target_variables"):
  """Returns an op to update a list of target variables from source variables.

  The update rule is:
  `target_variable = (1 - tau) * target_variable + tau * source_variable`.

  Args:
    target_variables: a list of the variables to be updated.
    source_variables: a list of the variables used for the update.
    tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
      with small tau representing an incremental update, and tau == 1
      representing a full update (that is, a straight copy).
    use_locking: use `tf.Variable.assign`'s locking option when assigning
      source variable values to target variables.
    name: sets the `name_scope` for this op.

  Raises:
    TypeError: when tau is not a Python float
    ValueError: when tau is out of range, or the source and target variables
      have different numbers or shapes.

  Returns:
    An op that executes all the variable updates.
  """
  if not isinstance(tau, float):
    raise TypeError("Tau has wrong type (should be float) {}".format(tau))
  if not 0.0 < tau <= 1.0:
    raise ValueError("Invalid parameter tau {}".format(tau))
  if len(target_variables) != len(source_variables):
    raise ValueError("Number of target variables {} is not the same as "
                     "number of source variables {}".format(
                         len(target_variables), len(source_variables)))

  same_shape = all(trg.get_shape() == src.get_shape()
                   for trg, src in zip(target_variables, source_variables))
  if not same_shape:
    raise ValueError("Target variables don't have the same shape as source "
                     "variables.")

  def update_op(target_variable, source_variable, tau):
    if tau == 1.0:
      return target_variable.assign(source_variable, use_locking)
    else:
      return target_variable.assign(
          tau * source_variable + (1.0 - tau) * target_variable, use_locking)

  with tf.name_scope(name, values=target_variables + source_variables):
    update_ops = [update_op(target_var, source_var, tau)
                  for target_var, source_var
                  in zip(target_variables, source_variables)]
    return tf.group(name="update_all_variables", *update_ops)


def periodic_target_update(target_variables,
                           source_variables,
                           update_period,
                           tau=1.0,
                           use_locking=False,
                           name="periodic_target_update"):
  """Returns an op to periodically update a list of target variables.

  The `update_target_variables` op is executed every `update_period`
  executions of the `periodic_target_update` op.

  The update rule is:
  `target_variable = (1 - tau) * target_variable + tau * source_variable`.

  Args:
    target_variables: a list of the variables to be updated.
    source_variables: a list of the variables used for the update.
    update_period: inverse frequency with which to apply the update.
    tau: weight used to gate the update. The permitted range is 0 < tau <= 1,
      with small tau representing an incremental update, and tau == 1
      representing a full update (that is, a straight copy).
    use_locking: use `tf.variable.Assign`'s locking option when assigning
      source variable values to target variables.
    name: sets the `name_scope` for this op.

  Returns:
    An op that periodically updates `target_variables` with `source_variables`.
  """

  def update_op():
    return update_target_variables(
        target_variables, source_variables, tau, use_locking)

  with tf.name_scope(name, values=target_variables + source_variables):
    return periodic_ops.periodically(update_op, update_period)
