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
"""TensorFlow ops for various distribution projection operations.

All ops support multidimensional tensors. All dimensions except for the last
one can be considered as batch dimensions. They are processed in parallel
and are fully independent. The last dimension represents the number of bins.

The op supports broadcasting across all dimensions except for the last one.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
from trfl import gen_distribution_ops


def l2_project(support, weights, new_support):
  """Projects distribution (support, weights) onto new_support.

  Args:
    support: Tensor defining support of a categorical distribution(s). Must be
      of rank 1 or of the same rank as `weights`. The size of the last dimension
      has to match that of `weights`.
    weights: Tensor defining weights on the support points.
    new_support: Tensor holding positions of a new support.
  Returns:
    Projection of (support, weights) onto the new_support.
  """
  return gen_distribution_ops.project_distribution(
      support, weights, new_support, 1)


def hard_cumulative_project(support, weights, new_support, reverse):
  """Produces a cumulative categorical distribution on a new support.

  Args:
    support: Tensor defining support of a categorical distribution(s). Must be
      of rank 1 or of the same rank as `weights`. The size of the last dimension
      has to match that of `weights`.
    weights: Tensor defining weights on the support points.
    new_support: Tensor holding positions of a new support.
    reverse: Whether to evalute cumulative from the left or right.
  Returns:
    Cumulative distribution on the supplied support.
    The foolowing invariant is maintained across the last dimension:
    result[i] = (sum_j weights[j] for all j where support[j] < new_support[i])
                if reverse == False else
                (sum_j weights[j] for all j where support[j] > new_support[i])
  """
  return gen_distribution_ops.project_distribution(
      support, weights, new_support, 3 if reverse else 2)


def factorised_kl_gaussian(dist1_mean,
                           dist1_covariance_or_scale,
                           dist2_mean,
                           dist2_covariance_or_scale,
                           both_diagonal=False):
  """Compute the KL divergence KL(dist1, dist2) between two Gaussians.

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


  Args:
    dist1_mean: The mean of the first Multivariate Gaussian distribution.
    dist1_covariance_or_scale: The covariance or scale of the first Multivariate
      Gaussian distribution. In cases where *both* distributions are Gaussians
      with diagonal covariance matrices (for instance, if both are instances of
      `tfp.distributions.MultivariateNormalDiag`), then the `scale` can be
      passed in instead and the `both_diagonal` flag must be set to `True`.
      A more efficient sparse computation path is used in this case. For all
      other cases, the full covariance matrix must be passed in.
    dist2_mean: The mean of the second Multivariate Gaussian distribution.
    dist2_covariance_or_scale: The covariance or scale tensor of the second
      Multivariate Gaussian distribution, as for `dist1_covariance_or_scale`.
    both_diagonal: A `bool` indicating that both dist1 and dist2 are diagonal
      matrices. A more efficient sparse computation is used in this case.

  Returns:
    A tuple consisting of (`kl_mean`, `kl_cov`) which correspond to the mean and
    the covariance factorisation of the KL.
  """
  if both_diagonal:
    dist1_mean_rank = dist1_mean.get_shape().ndims
    dist1_covariance_or_scale.get_shape().assert_has_rank(dist1_mean_rank)
    dist2_mean_rank = dist2_mean.get_shape().ndims
    dist2_covariance_or_scale.get_shape().assert_has_rank(dist2_mean_rank)

    dist_type = tfp.distributions.MultivariateNormalDiag
  else:
    dist_type = tfp.distributions.MultivariateNormalFullCovariance

  # Recreate the distributions but with stop gradients on the mean and cov.
  dist1_stop_grad_mean = dist_type(
      tf.stop_gradient(dist1_mean), dist1_covariance_or_scale)
  dist2 = dist_type(dist2_mean, dist2_covariance_or_scale)

  # Now create a third distribution with the mean of dist1 and the variance of
  # dist2 and appropriate stop_gradients.
  dist3 = dist_type(dist1_mean, dist2_covariance_or_scale)
  dist3_stop_grad_mean = dist_type(
      tf.stop_gradient(dist1_mean), dist2_covariance_or_scale)

  # Finally get the two components of the KL between dist1 and dist2
  # using dist3
  kl_mean = tfp.distributions.kl_divergence(dist3, dist2)
  kl_cov = tfp.distributions.kl_divergence(dist1_stop_grad_mean,
                                           dist3_stop_grad_mean)
  return kl_mean, kl_cov
