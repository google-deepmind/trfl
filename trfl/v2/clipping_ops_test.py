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
"""Tests for clipping_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
import tensorflow.compat.v2 as tf
from trfl import clipping_ops


class HuberLossTest(tf.test.TestCase):

  def testValue(self):
    quadratic_linear_boundary = 2
    xs = np.array(
        [-3.5, -2.1, 2, -1.9 - 1, -0.5, 0, 0.5, 1, 1.9, 2, 2.1, 3.5])
    xs_tf = tf.constant(xs)
    ys = clipping_ops.huber_loss(xs_tf, quadratic_linear_boundary)

    d = quadratic_linear_boundary

    # Check values for x <= -2
    ys_lo = ys[xs <= -d]
    xs_lo = xs[xs <= -d]
    expected_ys_lo = [0.5 * d**2 + d * (-x - d) for x in xs_lo]
    self.assertAllClose(ys_lo, expected_ys_lo)

    # Check values for x >= 2
    ys_hi = ys[xs >= d]
    xs_hi = xs[xs >= d]
    expected_ys_hi = [0.5 * d**2 + d * (x - d) for x in xs_hi]
    self.assertAllClose(ys_hi, expected_ys_hi)

    # Check values for x in (-2, 2)
    ys_mid = ys[np.abs(xs) < d]
    xs_mid = xs[np.abs(xs) < d]
    expected_ys_mid = [0.5 * x**2 for x in xs_mid]
    self.assertAllClose(ys_mid, expected_ys_mid)

  def testGradient(self):
    quadratic_linear_boundary = 3
    xs = np.array(
        [-5, -4, -3.1, -3, -2.9, 2, -1, 0, 1, 2, 2.9, 3, 3.1, 4, 5])
    xs_tf = tf.constant(xs)

    with tf.GradientTape() as tape:
      tape.watch(xs_tf)
      loss = clipping_ops.huber_loss(xs_tf, quadratic_linear_boundary)

    grads = tape.gradient(loss, xs_tf)
    self.assertTrue(np.all(np.abs(grads) <= quadratic_linear_boundary))

    # Everything <= -3 should have gradient -3.
    grads_lo = grads[xs <= -quadratic_linear_boundary]
    self.assertAllEqual(grads_lo,
                        [-quadratic_linear_boundary] * grads_lo.shape[0])

    # Everything >= 3 should have gradient 3.
    grads_hi = grads[xs >= quadratic_linear_boundary]
    self.assertAllEqual(grads_hi,
                        [quadratic_linear_boundary] * grads_hi.shape[0])

    # x in (-3, 3) should have gradient x.
    grads_mid = grads[np.abs(xs) <= quadratic_linear_boundary]
    xs_mid = xs[np.abs(xs) <= quadratic_linear_boundary]
    self.assertAllEqual(grads_mid, xs_mid)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
