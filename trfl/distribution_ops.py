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

import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


def l2_project(z_p, p, z_q):
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
