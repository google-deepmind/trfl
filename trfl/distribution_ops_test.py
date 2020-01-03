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
"""Tests for distribution_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from trfl import distribution_ops


l2_project = distribution_ops.l2_project
hard_cumulative_project = distribution_ops.hard_cumulative_project
_MULTIVARIATE_GAUSSIAN_TYPES = [
    tfp.distributions.MultivariateNormalDiagPlusLowRank,
    tfp.distributions.MultivariateNormalDiag,
    tfp.distributions.MultivariateNormalTriL,
    tfp.distributions.MultivariateNormalFullCovariance
]


def _l2_project_reference(z_p, p, z_q):
  """Projects distribution (z_p, p) onto support z_q under L2-metric over CDFs.

  The supports z_p and z_q are specified as tensors of distinct atoms (given
  in ascending order).

  Let Kq be len(z_q) and Kp be len(z_p). This projection works for any
  support z_q, in particular Kq need not be equal to Kp.

  Args:
    z_p: Tensor holding support of distribution p, shape `[batch_size, Kp]`.
    p: Tensor holding probability values p(z_p[i]), shape `[batch_size, Kp]`.
    z_q: Tensor holding support to project onto, shape `[Kq]`.

  Returns:
    Projection of (z_p, p) onto support z_q under Cramer distance.
  """
  # Broadcasting of tensors is used extensively in the code below. To avoid
  # accidental broadcasting along unintended dimensions, tensors are defensively
  # reshaped to have equal number of dimensions (3) throughout and intended
  # shapes are indicated alongside tensor definitions. To reduce verbosity,
  # extra dimensions of size 1 are inserted by indexing with `None` instead of
  # `tf.expand_dims()` (e.g., `x[:, None, :]` reshapes a tensor of shape
  # `[k, l]' to one of shape `[k, 1, l]`).

  # Extract vmin and vmax and construct helper tensors from z_q
  vmin, vmax = z_q[0], z_q[-1]
  d_pos = tf.concat([z_q, vmin[None]], 0)[1:]  # 1 x Kq x 1
  d_neg = tf.concat([vmax[None], z_q], 0)[:-1]  # 1 x Kq x 1
  # Clip z_p to be in new support range (vmin, vmax).
  z_p = tf.clip_by_value(z_p, vmin, vmax)[:, None, :]  # B x 1 x Kp

  # Get the distance between atom values in support.
  d_pos = (d_pos - z_q)[None, :, None]  # z_q[i+1] - z_q[i]. 1 x B x 1
  d_neg = (z_q - d_neg)[None, :, None]  # z_q[i] - z_q[i-1]. 1 x B x 1
  z_q = z_q[None, :, None]  # 1 x Kq x 1

  # Ensure that we do not divide by zero, in case of atoms of identical value.
  d_neg = tf.where(d_neg > 0, 1./d_neg, tf.zeros_like(d_neg))  # 1 x Kq x 1
  d_pos = tf.where(d_pos > 0, 1./d_pos, tf.zeros_like(d_pos))  # 1 x Kq x 1

  delta_qp = z_p - z_q   # clip(z_p)[j] - z_q[i]. B x Kq x Kp
  d_sign = tf.cast(delta_qp >= 0., dtype=p.dtype)  # B x Kq x Kp

  # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i].
  # Shape  B x Kq x Kp.
  delta_hat = (d_sign * delta_qp * d_pos) - ((1. - d_sign) * delta_qp * d_neg)
  p = p[:, None, :]  # B x 1 x Kp.
  return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * p, 2)


def _generate_support(size, sort=False):
  """Generate a random support vector."""
  # Note that support does not have to be monotonic.
  support = np.random.normal(-1.0, 1.0, size=size).astype(np.float32)
  if sort:
    return np.sort(support, axis=-1)
  return support


def _generate_weights(size, positive=False):
  """Generates a weight distribution where half of entries are zero."""
  normal = np.random.normal(-1.0, 1.0, size=size).astype(np.float32)
  if positive:
    normal = np.absolute(normal)
  mask = (np.random.random(size=size) > 0.5).astype(np.float32)
  return normal * mask


def _merge_supports(s1, s2):
  rank = len(s1.shape)
  assert len(s2.shape) == rank
  return np.concatenate((s1, s2), rank - 1)


def _mean_and_variance(support, weights):
  rank = len(weights.shape)
  sumw = np.sum(weights, axis=rank-1)
  mean = np.sum(support * weights, axis=rank - 1) / sumw
  zero_mean_support = support - np.expand_dims(mean, rank - 1)
  variance = np.sum(zero_mean_support * zero_mean_support * weights,
                    axis=rank-1) / sumw
  return mean, variance


def _imitate_broadcasting_by_tiling(tensors, broadcastable_dims):
  dims_expanded = 0
  expanded = []
  rank = 1
  for tensor in tensors:
    rank = np.max([rank, len(tensor.shape)])
  for tensor in tensors:
    # If rank is too low, we expand tensors.
    current_rank = len(tensor.shape)
    for i in range(rank - current_rank):
      tensor = np.expand_dims(tensor, 0)
    expanded.append(tensor)
  for d in broadcastable_dims:
    dim_expanded = 0
    size = 1
    for tensor in expanded:
      cs = tensor.shape[d]
      if size == 1:
        size = cs
      elif cs > 1:
        assert cs == size
    if size > 1:
      for i in range(len(expanded)):
        if expanded[i].shape[d] == 1:
          dim_expanded = 1
          expanded[i] = np.repeat(expanded[i], size, axis=d)
    dims_expanded += dim_expanded
  return expanded, dims_expanded


class L2ProjectTest(tf.test.TestCase):

  def _checkShapeInference(self, arg_shapes, expected_output_shape):
    placeholders = [tf.placeholder(tf.float32, shape=s) for s in arg_shapes]
    output = l2_project(*placeholders)
    output_shape = output.get_shape().as_list()
    self.assertAllEqual(output_shape, expected_output_shape)

  def testStaticShapeInference(self):
    # Check shape inference on non-bin dimensions.
    self._checkShapeInference([[7], [2, 7], [5]], [2, 5])
    self._checkShapeInference([[7], [None, 7], [5]], [None, 5])
    self._checkShapeInference([[6, 7], [None, 7], [5]], [6, 5])
    self._checkShapeInference([[6, 7], [None, 7], [None, 5]], [6, 5])
    self._checkShapeInference([[7], [None, 7], [3, 5]], [3, 5])
    self._checkShapeInference([[7], [None, 7], [None, 5]], [None, 5])
    self._checkShapeInference([[1, 3], [None, 3], [7, 4]], [7, 4])
    self._checkShapeInference([[1, 3], [None, 3], [None, 4]], [None, 4])
    self._checkShapeInference([[2, 3], [None, 3], [None, 4]], [2, 4])
    self._checkShapeInference([[None, 3], [None, 3], [None, 4]], [None, 4])
    self._checkShapeInference(
        [[2, None, None, 1, 9], [2, 3, None, None, 9], [2, 3, 4, 5, 7]],
        [2, 3, 4, 5, 7])
    # Check shape inference on bin dimensions.
    self._checkShapeInference([[7], [2, None], [5]], [2, 5])
    self._checkShapeInference([[3, 7], [3, None], [5]], [3, 5])
    self._checkShapeInference([[None], [2, 7], [5]], [2, 5])
    self._checkShapeInference([[3, None], [1, 7], [5]], [3, 5])
    self._checkShapeInference([[3, None], [1, 7], [None]], [3, None])
    self._checkShapeInference([[1, None], [1, 7], [7, None]], [7, None])

  def testOneHotProbabilities(self):
    support_new = tf.constant([-1.0, 0.0, 2.0], dtype=tf.float32)
    support_old = tf.constant(
        [-1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0], dtype=tf.float32)
    probabilities = tf.constant(np.identity(9), dtype=tf.float32)
    expected = [[1.0, 0.0, 0.0],  # This is an underflow example.
                [1.0, 0.0, 0.0],
                [0.75, 0.25, 0.0],
                [0.5, 0.5, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.75, 0.25],
                [0.0, 0.25, 0.75],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0]  # This is an overflow example.
               ]
    new_probabilities = l2_project(support_old, probabilities, support_new)
    with self.test_session():
      result = new_probabilities.eval()
      self.assertAllClose(result, expected)

  def testOneHotProbabilitiesOnNonMonotonicSupports(self):
    support_new = tf.constant([0.0, 2.0, -1.0], dtype=tf.float32)
    support_old = tf.constant(
        [0.0, -1.0, 1.5, 3.0, -1.5, 0.5, -0.75, 2.0, -0.5], dtype=tf.float32)
    probabilities = tf.constant(np.identity(9), dtype=tf.float32)
    expected = [[1.0, 0.0, 0.0],  # This is an underflow example.
                [0.0, 0.0, 1.0],
                [0.25, 0.75, 0.0],
                [0.0, 1.0, 0.0],  # This is an overflow example.
                [0.0, 0.0, 1.0],
                [0.75, 0.25, 0.0],
                [0.25, 0.0, 0.75],
                [0.0, 1.0, 0.0],
                [0.5, 0.0, 0.5]]
    new_probabilities = l2_project(support_old, probabilities, support_new)
    with self.test_session():
      result = new_probabilities.eval()
      self.assertAllClose(result, expected)

  def _permute_bins(self, support, weights):
    """Randomly permutes 2D support and weight tensors along each row."""
    support_new = np.zeros_like(support)
    weights_new = np.zeros_like(support)
    for i in range(support.shape[0]):
      permutation = np.random.permutation(support.shape[1])
      support_new[i, :] = support[i][permutation]
      weights_new[i, :] = weights[i][permutation]
    return support_new, weights_new

  def _checkBinPermutationConsistency(
      self, support_old, weights, support_new):
    """Checks if results are invariant under reshuffling of support bins."""
    # This is useful to check if the algorithm can handle non-monotonically
    # increasing supports. It indeed can, but it becomes O(n ln n) in the
    # number of bins.
    output1 = l2_project(tf.constant(support_old, tf.float32),
                         tf.constant(weights, tf.float32),
                         tf.constant(support_new, tf.float32))
    with self.test_session():
      result1 = output1.eval()
    support_old_p, weights_p = self._permute_bins(support_old, weights)
    support_new_p, target_p = self._permute_bins(support_new, result1)
    output2 = l2_project(tf.constant(support_old_p, tf.float32),
                         tf.constant(weights_p, tf.float32),
                         tf.constant(support_new_p, tf.float32))
    with self.test_session():
      result2 = output2.eval()
    self.assertAllEqual(result2, target_p)

  def testNonMonotonicSupports(self):
    dim = 50
    bins1 = 50
    bins2 = 60
    support_old = _generate_support(size=(dim, bins1), sort=True)
    support_new = _generate_support(size=(dim, bins2), sort=True)
    weights = _generate_weights(size=(dim, bins1))
    self._checkBinPermutationConsistency(support_old, weights, support_new)

  def _checkBroadcastingConsistency(self, args, broadcastable_dims):
    """Checks whether explicit tiling is consistent with broadcasting."""
    expanded_args, dims_expanded = _imitate_broadcasting_by_tiling(
        args, broadcastable_dims)
    self.assertGreater(dims_expanded, 0)
    original = [tf.constant(t, tf.float32) for t in args]
    expanded = [tf.constant(t, tf.float32) for t in expanded_args]
    output1 = l2_project(*original)
    output2 = l2_project(*expanded)
    self.assertAllEqual(output1.get_shape().as_list(),
                        output2.get_shape().as_list())
    with self.test_session():
      result1 = output1.eval()
      result2 = output2.eval()
      self.assertAllEqual(result1, result2)

  def testBroadcasting(self):
    """Check whether broadcasting works on a hand-crafted data."""
    support_new = np.zeros((3, 1, 1, 3))
    support_new[0, 0, 0, :] = [-1.0, 0.0, 2.0]
    support_new[1, 0, 0, :] = [-2.0, 1.0, 3.0]
    support_new[2, 0, 0, :] = [0.0, 1.0, 3.0]
    support_new_c = tf.constant(support_new, dtype=tf.float32)
    support_old = np.zeros((1, 2, 1, 5))
    support_old[0, 0, 0, :] = [-1.5, -0.75, 0.0, 0.5, 1.5]
    support_old[0, 1, 0, :] = [1.0, 2.0, 3.0, 4.0, 5.0]
    support_old_c = tf.constant(support_old, dtype=tf.float32)
    weights = np.zeros((1, 1, 4, 5))
    weights[0, 0, 0, :] = [1.0, -1.0, 0.0, 0.0, 0.0]
    weights[0, 0, 1, :] = [0.0, 0.0, 1.0, 0.0, 1.0]
    weights[0, 0, 2, :] = [0.0, 0.0, 0.0, 1.0, 0.0]
    weights[0, 0, 3, :] = [0.0, -1.0, 0.0, 0.0, 1.0]
    weights_c = tf.constant(weights, dtype=tf.float32)
    new_probabilities = l2_project(support_old_c, weights_c, support_new_c)
    shape = new_probabilities.get_shape().as_list()
    self.assertAllEqual(shape, [3, 2, 4, 3])
    expected000 = [0.25, -0.25, 0.0]
    expected012 = [0.0, 0.0, 1.0]
    expected111 = [0.0, 0.0, 2.0]
    expected203 = [-1.0, 0.75, 0.25]
    with self.test_session():
      result = new_probabilities.eval()
      self.assertAllEqual(result[0, 0, 0], expected000)
      self.assertAllEqual(result[0, 1, 2], expected012)
      self.assertAllEqual(result[1, 1, 1], expected111)
      self.assertAllEqual(result[2, 0, 3], expected203)
    self._checkBroadcastingConsistency([support_old, weights, support_new],
                                       range(3))

  def testBroadcastingLarge(self):
    """Check whether broadcasting is consistent with explicit expansion."""
    dim1 = 10
    dim2 = 20
    dim3 = 15
    bins1 = 200
    bins2 = 300
    support_old = _generate_support(size=(1, 1, dim3, bins1))
    support_new = _generate_support(size=(dim1, 1, 1, bins2))
    weights = _generate_weights(size=(1, dim2, 1, bins1))
    self._checkBroadcastingConsistency([support_old, weights, support_new],
                                       range(3))

  def testInverseConsistencyOnSupersetSupport(self):
    """Projection should be reversible when projecting onto a superset sup."""
    dim1 = 16
    dim2 = 8
    bins1 = 15
    bins2 = 28
    support_old = _generate_support((dim1, dim2, bins1))
    additional = _generate_support((dim1, dim2, bins2 - bins1))
    support_new = _merge_supports(additional, support_old)
    weights = _generate_weights(size=(dim1, dim2, bins1))
    support_old = tf.constant(support_old, tf.float32)
    support_new = tf.constant(support_new, tf.float32)
    weights = tf.constant(weights, tf.float32)
    new_probabilities = l2_project(support_old, weights, support_new)
    recovered_probabilities = l2_project(support_new, new_probabilities,
                                         support_old)
    with self.test_session() as session:
      p1, p2 = session.run([new_probabilities, recovered_probabilities])
      self.assertAllEqual(p1.shape, [dim1, dim2, bins2])
      self.assertAllEqual(p2.shape, [dim1, dim2, bins1])
      self.assertAllEqual(p2, weights.eval())

  def testShouldPreserveMeanWhenProjectingOntoSufficientlyWideSupport(self):
    """Mean should be preserved when projecting onto a wider support."""
    dim1 = 100
    bins1 = 28
    bins2 = 15
    support_old = _generate_support((dim1, bins1), sort=True)
    support_new = _generate_support((dim1, bins2), sort=True)
    # Force the new support to be at least as wide as.
    support_new[:, 0] = np.minimum(support_new[:, 0], support_old[:, 0])
    support_new[:, bins2 - 1] = np.maximum(support_new[:, bins2 - 1],
                                           support_old[:, bins1 - 1])
    # Generate positive weights.
    weights = _generate_weights(size=(dim1, bins1), positive=True)
    support_old = tf.constant(support_old, tf.float32)
    support_new = tf.constant(support_new, tf.float32)
    weights = tf.constant(weights, tf.float32)
    new_weights = l2_project(support_old, weights, support_new)
    with self.test_session():
      m1, v1 = _mean_and_variance(support_old.eval(), weights.eval())
      m2, v2 = _mean_and_variance(support_new.eval(), new_weights.eval())
      # Means should be preserved.
      self.assertAllClose(m1, m2)
      # Variance should increase due to spreding encountered while rebinning.
      dv = v2 - v1
      mdv = np.amin(dv)
      self.assertGreater(mdv, 0.0)

  def _checkConsistencyWithReference(self, rows, bins1, bins2):
    support_old = tf.constant(_generate_support((rows, bins1), sort=True),
                              tf.float32)
    support_new = tf.constant(_generate_support((bins2,), sort=True),
                              tf.float32)
    weights = tf.constant(_generate_weights(size=(rows, bins1)), tf.float32)
    ref_output = _l2_project_reference(support_old, weights, support_new)
    our_output = l2_project(support_old, weights, support_new)
    with self.test_session() as session:
      ref, out = session.run([ref_output, our_output])
      self.assertAllClose(ref, out)

  def testConsistencyWithReference(self):
    self._checkConsistencyWithReference(100, 7, 21)
    self._checkConsistencyWithReference(100, 13, 6)


class ProjectDistributionTest(tf.test.TestCase):

  def testOneHotProbabilitiesCumulative(self):
    support_new = tf.constant([-2.0, -1.0, -0.8, -0.5, 0.1, 2.0, 3.0],
                              dtype=tf.float32)
    support_old = tf.constant(
        [-2.1, -2.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.5, 2.0, 3.0, 3.5],
        dtype=tf.float32)
    weights = np.zeros((12, 12), dtype=np.float32)
    np.fill_diagonal(weights, [1, 2, 3, 4, 5, 6, 7, 8, 9, 1.1, 1.2, 1.3])
    probabilities = tf.constant(weights, dtype=tf.float32)
    expected_lt = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   [0.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
                   [0.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
                   [0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                   [0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0],
                   [0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0],
                   [0.0, 0.0, 0.0, 0.0, 7.0, 7.0, 7.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 8.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    expected_gt = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0],
                   [6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0],
                   [7.0, 7.0, 7.0, 7.0, 0.0, 0.0, 0.0],
                   [8.0, 8.0, 8.0, 8.0, 8.0, 0.0, 0.0],
                   [9.0, 9.0, 9.0, 9.0, 9.0, 0.0, 0.0],
                   [1.1, 1.1, 1.1, 1.1, 1.1, 0.0, 0.0],
                   [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.0],
                   [1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3]]
    # Cumulative from the left to the right (i.e. of weights less than target).
    cumulative_lt = hard_cumulative_project(support_old, probabilities,
                                            support_new, reverse=False)
    # Cumulative from the right to the left (i.e. of weights greater than).
    cumulative_gt = hard_cumulative_project(support_old, probabilities,
                                            support_new, reverse=True)
    with self.test_session():
      result_lt = cumulative_lt.eval()
      result_gt = cumulative_gt.eval()
      self.assertAllClose(result_lt, expected_lt)
      self.assertAllClose(result_gt, expected_gt)

  def testOneHotProbabilitiesOnNonMonotonicSupports(self):
    """Tests whether the op can deal with non-monotonic supports."""
    support_new = tf.constant([3.0, -1.0, -0.8, -0.5, -2.0, 0.1, 2.0],
                              dtype=tf.float32)
    support_old = tf.constant(
        [-2.1, -2.0, -1.5, -1.0, 0.5, 1.5, 2.0, 3.0, 3.5],
        dtype=tf.float32)
    weights = np.zeros((9, 9), dtype=np.float32)
    np.fill_diagonal(weights, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    probabilities = tf.constant(weights, dtype=tf.float32)
    expected_lt = [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                   [2.0, 2.0, 2.0, 2.0, 0.0, 2.0, 2.0],
                   [3.0, 3.0, 3.0, 3.0, 0.0, 3.0, 3.0],
                   [4.0, 0.0, 4.0, 4.0, 0.0, 4.0, 4.0],
                   [5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
                   [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0],
                   [7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    expected_gt = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
                   [0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 0.0],
                   [0.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0],
                   [0.0, 7.0, 7.0, 7.0, 7.0, 7.0, 0.0],
                   [0.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
                   [9.0, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0]]
    # Cumulative from the left to the right (i.e. of weights less than target).
    cumulative_lt = hard_cumulative_project(support_old, probabilities,
                                            support_new, reverse=False)
    # Cumulative from the right to the left (i.e. of weights greater than).
    cumulative_gt = hard_cumulative_project(support_old, probabilities,
                                            support_new, reverse=True)
    with self.test_session():
      result_lt = cumulative_lt.eval()
      result_gt = cumulative_gt.eval()
      self.assertAllClose(result_lt, expected_lt)
      self.assertAllClose(result_gt, expected_gt)

  def testSimpleCumulative(self):
    """Tests covering both non-monotonic supports and simple cases."""
    # Test coverage by lines:
    # 1-3 - permute order of source bins.
    # 4-5 - permute order of target bins.
    support_new = tf.constant([[-2.0, -1.0, -0.8, -0.5, 0.1, 1.2, 3.0],
                               [-2.0, -1.0, -0.8, -0.5, 0.1, 1.2, 3.0],
                               [-2.0, -1.0, -0.8, -0.5, 0.1, 1.2, 3.0],
                               [3.0, 1.2, 0.1, -0.5, -0.8, -1.0, -2.0],
                               [1.2, 3.0, -0.5, -1.0, 0.1, -2.0, -0.8]],
                              dtype=tf.float32)
    support_old = tf.constant(
        [[-2.1, -2.0, -1.0, -0.7, -0.5, 0.0, 1.3, 1.5, 2.0, 3.0],
         [3.0, 2.0, 1.5, 1.3, 0.0, -0.5, -0.7, -1.0, -2.0, -2.1],
         [-2.0, 1.5, 1.3, 0.0, 2.0, -0.5, -0.7, -1.0, -2.1, 3.0],
         [-2.0, 1.5, 1.3, 0.0, 2.0, -0.5, -0.7, -1.0, -2.1, 3.0],
         [3.0, 2.0, 1.5, 1.3, 0.0, -0.5, -0.7, -1.0, -2.0, -2.1]],
        dtype=tf.float32)
    weights = tf.constant(
        [[0.1, -0.2, 1.5, 1.0, 1.7, -0.5, 0.2, -0.5, 1.5, 0.2],
         [0.2, 1.5, -0.5, 0.2, -0.5, 1.7, 1.0, 1.5, -0.2, 0.1],
         [-0.2, -0.5, 0.2, -0.5, 1.5, 1.7, 1.0, 1.5, 0.1, 0.2],
         [-0.2, -0.5, 0.2, -0.5, 1.5, 1.7, 1.0, 1.5, 0.1, 0.2],
         [0.2, 1.5, -0.5, 0.2, -0.5, 1.7, 1.0, 1.5, -0.2, 0.1]],
        dtype=tf.float32)
    expected_lt = [[0.1, -0.1, 1.4, 2.4, 3.6, 3.6, 4.8],
                   [0.1, -0.1, 1.4, 2.4, 3.6, 3.6, 4.8],
                   [0.1, -0.1, 1.4, 2.4, 3.6, 3.6, 4.8],
                   [4.8, 3.6, 3.6, 2.4, 1.4, -0.1, 0.1],
                   [3.6, 4.8, 2.4, -0.1, 3.6, 0.1, 1.4]]
    expected_gt = [[5.1, 3.6, 3.6, 0.9, 1.4, 1.4, 0.0],
                   [5.1, 3.6, 3.6, 0.9, 1.4, 1.4, 0.0],
                   [5.1, 3.6, 3.6, 0.9, 1.4, 1.4, 0.0],
                   [0.0, 1.4, 1.4, 0.9, 3.6, 3.6, 5.1],
                   [1.4, 0.0, 0.9, 3.6, 1.4, 5.1, 3.6]]
    # Cumulative from the left to the right (i.e. of weights less than target).
    cumulative_lt = hard_cumulative_project(support_old, weights,
                                            support_new, reverse=False)
    # Cumulative from the right to the left (i.e. of weights greater than).
    cumulative_gt = hard_cumulative_project(support_old, weights,
                                            support_new, reverse=True)
    with self.test_session():
      result_lt = cumulative_lt.eval()
      result_gt = cumulative_gt.eval()
      self.assertAllClose(result_lt, expected_lt)
      self.assertAllClose(result_gt, expected_gt)


class FactorisedKLGaussianTest(tf.test.TestCase, parameterized.TestCase):

  def _create_gaussian(self, gaussian_type):
    mu = tf.random_normal([3])
    if gaussian_type == tfp.distributions.MultivariateNormalDiag:
      scale_diag = tf.random_normal([3])
      dist = tfp.distributions.MultivariateNormalDiag(mu, scale_diag)
    if gaussian_type == tfp.distributions.MultivariateNormalDiagPlusLowRank:
      scale_diag = tf.random_normal([3])
      perturb_factor = tf.random_normal([3, 2])
      scale_perturb_diag = tf.random_normal([2])
      dist = tfp.distributions.MultivariateNormalDiagPlusLowRank(
          mu,
          scale_diag,
          scale_perturb_factor=perturb_factor,
          scale_perturb_diag=scale_perturb_diag)
    if gaussian_type == tfp.distributions.MultivariateNormalTriL:
      cov = tf.random_uniform([3, 3], minval=0, maxval=1.0)
      # Create a PSD matrix.
      cov = 0.5 * (cov + tf.transpose(cov)) + 3 * tf.eye(3)
      scale = tf.cholesky(cov)
      dist = tfp.distributions.MultivariateNormalTriL(mu, scale)
    if gaussian_type == tfp.distributions.MultivariateNormalFullCovariance:
      cov = tf.random_uniform([3, 3], minval=0, maxval=1.0)
      # Create a PSD matrix.
      cov = 0.5 * (cov + tf.transpose(cov)) + 3 * tf.eye(3)
      dist = tfp.distributions.MultivariateNormalFullCovariance(mu, cov)
    return (dist, mu, dist.covariance())

  @parameterized.parameters(
      itertools.product(_MULTIVARIATE_GAUSSIAN_TYPES,
                        _MULTIVARIATE_GAUSSIAN_TYPES))
  def testFactorisedKLGaussian(self, dist1_type, dist2_type):
    """Tests that the factorised KL terms sum up to the true KL."""
    dist1, dist1_mean, dist1_cov = self._create_gaussian(dist1_type)
    dist2, dist2_mean, dist2_cov = self._create_gaussian(dist2_type)
    both_diagonal = _is_diagonal(dist1.scale) and _is_diagonal(dist2.scale)
    if both_diagonal:
      dist1_cov = dist1.parameters['scale_diag']
      dist2_cov = dist2.parameters['scale_diag']
    kl = tfp.distributions.kl_divergence(dist1, dist2)
    kl_mean, kl_cov = distribution_ops.factorised_kl_gaussian(
        dist1_mean,
        dist1_cov,
        dist2_mean,
        dist2_cov,
        both_diagonal=both_diagonal)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      actual_kl, kl_mean_np, kl_cov_np = sess.run([kl, kl_mean, kl_cov])
      self.assertAllClose(actual_kl, kl_mean_np + kl_cov_np, rtol=1e-4)

  def testShapeAssertion(self):
    dist_type = tfp.distributions.MultivariateNormalDiag
    _, dist1_mean, dist1_cov = self._create_gaussian(dist_type)
    _, dist2_mean, dist2_cov = self._create_gaussian(dist_type)
    shape_error_regexp = 'Shape (.*) must have rank [0-9]+'
    with self.assertRaisesRegexp(ValueError, shape_error_regexp):
      distribution_ops.factorised_kl_gaussian(
          dist1_mean, dist1_cov, dist2_mean, dist2_cov, both_diagonal=True)

  def testConsistentGradientsBothDiagonal(self):
    dist_type = tfp.distributions.MultivariateNormalDiag
    dist1, dist1_mean, _ = self._create_gaussian(dist_type)
    dist2, dist2_mean, _ = self._create_gaussian(dist_type)

    kl = tfp.distributions.kl_divergence(dist1, dist2)
    dist1_scale = dist1.parameters['scale_diag']
    dist2_scale = dist2.parameters['scale_diag']
    kl_mean, kl_cov = distribution_ops.factorised_kl_gaussian(
        dist1_mean, dist1_scale, dist2_mean, dist2_scale, both_diagonal=True)

    dist_params = [dist1_mean, dist2_mean, dist1_scale, dist2_scale]
    actual_kl_gradients = tf.gradients(kl, dist_params)
    factorised_kl_gradients = tf.gradients(kl_mean + kl_cov, dist_params)

    # Check that no gradients flow into the mean terms from `kl_cov` and
    # vice-versa.
    gradients = tf.gradients(kl_mean, [dist1_scale])
    self.assertListEqual(gradients, [None])
    gradients = tf.gradients(kl_cov, [dist1_mean, dist2_mean])
    self.assertListEqual(gradients, [None, None])

    with self.test_session() as sess:
      np_actual_kl, np_factorised_kl = sess.run(
          [actual_kl_gradients, factorised_kl_gradients])
      self.assertAllClose(np_actual_kl, np_factorised_kl)

  def testConsistentGradientsFullCovariance(self):
    dist_type = tfp.distributions.MultivariateNormalFullCovariance
    dist1, dist1_mean, dist1_cov = self._create_gaussian(dist_type)
    dist2, dist2_mean, dist2_cov = self._create_gaussian(dist_type)

    kl = tfp.distributions.kl_divergence(dist1, dist2)
    kl_mean, kl_cov = distribution_ops.factorised_kl_gaussian(
        dist1_mean, dist1_cov, dist2_mean, dist2_cov, both_diagonal=False)

    dist1_cov = dist1.parameters['covariance_matrix']
    dist2_cov = dist2.parameters['covariance_matrix']
    dist_params = [
        dist1_mean,
        dist2_mean,
        dist1_cov,
        dist2_cov,
    ]
    actual_kl_gradients = tf.gradients(kl, dist_params)
    factorised_kl_gradients = tf.gradients(kl_mean + kl_cov, dist_params)

    # Check that no gradients flow into the mean terms from `kl_cov` and
    # vice-versa.
    gradients = tf.gradients(kl_mean, [dist1_cov])
    self.assertListEqual(gradients, [None])
    gradients = tf.gradients(kl_cov, [dist1_mean, dist2_mean])
    self.assertListEqual(gradients, [None, None])

    with self.test_session() as sess:
      np_actual_kl, np_factorised_kl = sess.run(
          [actual_kl_gradients, factorised_kl_gradients])
      self.assertAllClose(np_actual_kl, np_factorised_kl)


# Check for diagonal Gaussian distributions. Based on the definition in
# tensorflow_probability/python/distributions/mvn_linear_operator.py
def _is_diagonal(x):
  """Helper to identify if `LinearOperator` has only a diagonal component."""
  return (isinstance(x, tf.linalg.LinearOperatorIdentity) or
          isinstance(x, tf.linalg.LinearOperatorScaledIdentity) or
          isinstance(x, tf.linalg.LinearOperatorDiag))


if __name__ == '__main__':
  tf.test.main()
