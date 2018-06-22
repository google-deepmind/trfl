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
"""Utilities for Reinforcement Learning ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
# Dependency imports

from six.moves import zip
import tensorflow as tf

LossOutput = collections.namedtuple("loss_output", ["loss", "extra"])


def best_effort_shape(tensor, with_rank=None):
  """Extract as much static shape information from a tensor as possible.

  Args:
    tensor: A `Tensor`. If `with_rank` is None, must have statically-known
        number of dimensions.
    with_rank: Optional, an integer number of dimensions to force the shape to
        be. Useful for tensors with no static shape information that must be
        of a particular rank. Default is None (number of dimensions must be
        statically known).

  Returns:
    An iterable with length equal to the number of dimensions in `tensor`,
    containing integers for the dimensions with statically-known size, and
    scalar `Tensor`s for dimensions with size only known at run-time.

  Raises:
    ValueError: If `with_rank` is None and `tensor` does not have
      statically-known number of dimensions.
  """
  tensor_shape = tensor.get_shape()
  if with_rank:
    tensor_shape = tensor_shape.with_rank(with_rank)
  if tensor_shape.ndims is None:
    raise ValueError(
        "`tensor` does not have statically-known number of dimensions.")
  shape_list = tensor_shape.as_list()
  for idx, dim in enumerate(shape_list):
    if not dim:
      shape_list[idx] = tf.shape(tensor)[idx]
  return shape_list


def assert_rank_and_shape_compatibility(tensors, rank):
  """Asserts that the tensors have the correct rank and compatible shapes.

  Shapes (of equal rank) are compatible if corresponding dimensions are all
  equal or unspecified. E.g. `[2, 3]` is compatible with all of `[2, 3]`,
  `[None, 3]`, `[2, None]` and `[None, None]`.

  Args:
    tensors: List of tensors.
    rank: A scalar specifying the rank that the tensors passed need to have.

  Raises:
    ValueError: If the list of tensors is empty or fail the rank and mutual
      compatibility asserts.
  """
  if not tensors:
    raise ValueError("List of tensors should be non-empty.")

  union_of_shapes = tf.TensorShape(None)
  for tensor in tensors:
    tensor_shape = tensor.get_shape()
    tensor_shape.assert_has_rank(rank)
    union_of_shapes = union_of_shapes.merge_with(tensor_shape)


def wrap_rank_shape_assert(tensors_list, expected_ranks, op_name):
  try:
    for tensors, rank in zip(tensors_list, expected_ranks):
      assert_rank_and_shape_compatibility(tensors, rank)
  except ValueError as e:
    error_message = ("{}: Error in rank and/or "
                     "compatibility check, {}".format(op_name, e))
    tf.logging.error(error_message)
    raise ValueError(error_message)


def assert_arg_bounded(value, min_value, max_value, op_name, arg_name):
  if not min_value <= value <= max_value:
    raise ValueError(
        (op_name + ": " + arg_name + " has to lie in " +
         "[" + str(min_value) + ", " + str(max_value) + "]."))
