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
"""Periodic execution ops.

It is very common in Reinforcement Learning for certain ops to only need to be
executed periodically, for example: once every N agent steps. The ops below
support this common use-case by wrapping a subgraph as a periodic op that only
actually executes the underlying computation once every N evaluations of the op,
behaving as a no-op in all other calls.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def periodically(body, period, name="periodically"):
  """Periodically performs a tensorflow op.

  The body tensorflow op will be executed every `period` times the periodically
  op is executed. More specifically, with `n` the number of times the op has
  been executed, the body will be executed when `n` is a non zero positive
  multiple of `period` (i.e. there exist an integer `k > 0` such that
  `k * period == n`).

  If `period` is 0 or `None`, it would not perform any op and would return a
  `tf.no_op()`.

  Args:
    body: callable that returns the tensorflow op to be performed every time
      an internal counter is divisible by the period. The op must have no
      output (for example, a tf.group()).
    period: inverse frequency with which to perform the op.
    name: name of the variable_scope.

  Raises:
    TypeError: if body is not a callable.
    ValueError: if period is negative.

  Returns:
    An op that periodically performs the specified op.
  """
  if not callable(body):
    raise TypeError("body must be callable.")

  if period is None or period == 0:
    return tf.no_op()

  if period < 0:
    raise ValueError("period cannot be less than 0.")

  if period == 1:
    return body()

  with tf.variable_scope(None, default_name=name):
    counter = tf.get_variable(
        "counter",
        shape=[],
        dtype=tf.int64,
        trainable=False,
        initializer=tf.constant_initializer(period, dtype=tf.int64))

    def _wrapped_body():
      with tf.control_dependencies([body()]):
        # Done the deed, resets the counter.
        return counter.assign(1)

    update = tf.cond(
        tf.equal(counter, period), _wrapped_body, lambda: counter.assign_add(1))

  return update
