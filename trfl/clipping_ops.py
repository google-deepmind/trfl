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
"""Ops to implement gradient clipping."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def huber_loss(input_tensor, quadratic_linear_boundary, name=None):
  """Calculates huber loss of `input_tensor`.

  For each value x in `input_tensor`, the following is calculated:

  ```
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```

  where d is `quadratic_linear_boundary`.

  When `input_tensor` is a loss this results in a form of gradient clipping.
  This is, for instance, how gradients are clipped in DQN and its variants.

  Args:
    input_tensor: `Tensor`, input values to calculate the huber loss on.
    quadratic_linear_boundary: `float`, the point where the huber loss function
      changes from a quadratic to linear.
    name: `string`, name for the operation (optional).

  Returns:
    `Tensor` of the same shape as `input_tensor`, containing values calculated
    in the manner described above.

  Raises:
    ValueError: if quadratic_linear_boundary <= 0.
  """
  if quadratic_linear_boundary < 0:
    raise ValueError("quadratic_linear_boundary must be > 0.")

  with tf.name_scope(
      name, default_name="huber_loss",
      values=[input_tensor, quadratic_linear_boundary]):
    abs_x = tf.abs(input_tensor)
    delta = quadratic_linear_boundary
    quad = tf.minimum(abs_x, delta)
    # The following expression is the same in value as
    # tf.maximum(abs_x - delta, 0), but importantly the gradient for the
    # expression when abs_x == delta is 0 (for tf.maximum it would be 1). This
    # is necessary to avoid doubling the gradient, since there is already a
    # non-zero contribution to the gradient from the quadratic term.
    lin = (abs_x - quad)
    return 0.5 * quad**2 + delta * lin
