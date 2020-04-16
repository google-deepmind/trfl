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
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from trfl import distribution_ops


l2_project = distribution_ops.l2_project
_MULTIVARIATE_GAUSSIAN_TYPES = [
    tfp.distributions.MultivariateNormalDiagPlusLowRank,
    tfp.distributions.MultivariateNormalDiag,
    tfp.distributions.MultivariateNormalTriL,
    tfp.distributions.MultivariateNormalFullCovariance
]


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
