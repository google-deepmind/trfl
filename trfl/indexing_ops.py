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
"""Indexing ops.

These ops support indexing the 2D tensors representing batches of values
(shape: [B, dim]) or 3D tensors representing batches of sequences
of values (shape: [T, B, dim]. `T` is the length of the rollout, `B` is the
batch size, and `dim` the size of the dimension that must be indexed.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def assert_compatible_shapes(value_shape, index_shape):
  """Check shapes of the indices and the tensor to be indexed.

  If all input shapes are known statically, obtain shapes of arguments and
  perform compatibility checks. Otherwise, print a warning. The only check
  we cannot perform statically (and do not attempt elsewhere) is making
  sure that each action index in actions is in [0, num_actions).

  Args:
    value_shape: static shape of the values.
    index_shape: static shape of the indices.
  """
  # note: rank-0 "[]" TensorShape is still True.
  if value_shape and index_shape:
    try:
      msg = ("Shapes of \"values\" and \"indices\" do not correspond to "
             "minibatch (2-D) or sequence-minibatch (3-D) indexing")
      assert (value_shape.ndims, index_shape.ndims) in [(2, 1), (3, 2)], msg
      msg = ("\"values\" and \"indices\" have incompatible shapes of {} "
             "and {}, respectively").format(value_shape, index_shape)
      assert value_shape[:-1].is_compatible_with(index_shape), msg
    except AssertionError as e:
      raise ValueError(e)  # Convert AssertionError to ValueError.

  else:  # No shape information is known ahead of time.
    tf.logging.warning(
        "indexing function cannot get shapes for tensors \"values\" and "
        "\"indices\" at construction time, and so can't check that their "
        "shapes are valid or compatible. Incorrect indexing may occur at "
        "runtime without error!")


def batched_index(values, indices):
  """Equivalent to `values[:, indices]`.

  Performs indexing on batches and sequence-batches by reducing over
  zero-masked values. Compared to indexing with `tf.gather` this approach is
  more general and TPU-friendly, but may be less efficient if `num_values`
  is large. It works with tensors whose shapes are unspecified or
  partially-specified, but this op will only do shape checking on shape
  information available at graph construction time. When complete shape
  information is absent, certain shape incompatibilities may not be detected at
  runtime! See `indexing_ops_test` for detailed examples.

  Args:
    values: tensor of shape `[B, num_values]` or `[T, B, num_values]`
    indices: tensor of shape `[B]` or `[T, B]` containing indices.

  Returns:
    Tensor of shape `[B]` or `[T, B]` containing values for the given indices.

  Raises: ValueError if values and indices have sizes that are known
    statically (i.e. during graph construction), and those sizes are not
    compatible (see shape descriptions in Args list above).
  """
  with tf.name_scope("batch_indexing", values=[values, indices]):
    values = tf.convert_to_tensor(values)
    indices = tf.convert_to_tensor(indices)
    assert_compatible_shapes(values.shape, indices.shape)

    one_hot_indices = tf.one_hot(
        indices, tf.shape(values)[-1], dtype=values.dtype)
    return tf.reduce_sum(values * one_hot_indices, axis=-1)
