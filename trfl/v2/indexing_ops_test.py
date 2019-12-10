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
"""Tests for indexing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf
from trfl import indexing_ops


class BatchIndexingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([None, True, False])
  def testOrdinaryValues(self, keepdims):
    """Indexing value functions by action for a minibatch of values."""
    values = [[1.1, 1.2, 1.3],
              [1.4, 1.5, 1.6],
              [2.1, 2.2, 2.3],
              [2.4, 2.5, 2.6],
              [3.1, 3.2, 3.3],
              [3.4, 3.5, 3.6],
              [4.1, 4.2, 4.3],
              [4.4, 4.5, 4.6]]
    action_indices = [0, 2, 1, 0, 2, 1, 0, 2]
    result = indexing_ops.batched_index(
        values, action_indices, keepdims=keepdims)
    expected_result = [1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6]
    if keepdims:
      expected_result = np.expand_dims(expected_result, axis=-1)

    self.assertAllClose(result, expected_result)

  def testValueSequence(self):
    """Indexing value functions by action with a minibatch of sequences."""
    values = [[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
              [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
              [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
              [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]]]
    action_indices = [[0, 2],
                      [1, 0],
                      [2, 1],
                      [0, 2]]
    result = indexing_ops.batched_index(values, action_indices)
    expected_result = [[1.1, 1.6],
                       [2.2, 2.4],
                       [3.3, 3.5],
                       [4.1, 4.6]]

    self.assertAllClose(result, expected_result)

  def testInputShapeChecks(self):
    """Input shape checks can catch some, but not all, shape problems."""
    # 1. Inputs have incorrect or incompatible ranks:
    for args in [dict(values=[[5, 5]], indices=1),
                 dict(values=[5, 5], indices=[1]),
                 dict(values=[[[5, 5]]], indices=[1]),
                 dict(values=[[5, 5]], indices=[[[1]]]),]:
      with self.assertRaisesRegexp(ValueError, 'do not correspond'):
        indexing_ops.batched_index(**args)

    # 2. Inputs have correct, compatible ranks but incompatible sizes:
    for args in [dict(values=[[5, 5]], indices=[1, 1]),
                 dict(values=[[5, 5], [5, 5]], indices=[1]),
                 dict(values=[[[5, 5], [5, 5]]], indices=[[1, 1], [1, 1]]),
                 dict(values=[[[5, 5], [5, 5]]], indices=[[1], [1]]),]:
      with self.assertRaisesRegexp(ValueError, 'incompatible shapes'):
        indexing_ops.batched_index(**args)

    # (Correct ranks and sizes work fine, though):
    indexing_ops.batched_index(
        values=[[5, 5]], indices=[1])
    indexing_ops.batched_index(
        values=[[[5, 5], [5, 5]]], indices=[[1, 1]])

  def testPartiallySpecifiedInputShapeChecks(self):
    # 3. Shape-checking works with fully-specified TensorSpecs, or even
    # partially-specified TensorSpecs that still provide evidence of having
    # incompatible shapes or incorrect ranks.
    for sizes in [dict(q_size=[4, 3], a_size=[4, 1]),
                  dict(q_size=[4, 2, 3], a_size=[4, 1]),
                  dict(q_size=[4, 3], a_size=[5, None]),
                  dict(q_size=[None, 2, 3], a_size=[4, 1]),
                  dict(q_size=[4, 2, 3], a_size=[None, 1]),
                  dict(q_size=[4, 2, 3], a_size=[5, None]),
                  dict(q_size=[None, None], a_size=[None, None]),
                  dict(q_size=[None, None, None], a_size=[None]),]:

      with self.assertRaises(ValueError):
        @tf.function(input_signature=[
            tf.TensorSpec(sizes['q_size'], tf.float32),
            tf.TensorSpec(sizes['a_size'], tf.int32)
        ])
        def f1(q_values, actions):
          return indexing_ops.batched_index(q_values, actions)

        q_size = [d if d is not None else 1 for d in sizes['q_size']]
        a_size = [d if d is not None else 1 for d in sizes['a_size']]
        f1(tf.zeros(q_size, tf.float32), tf.zeros(a_size, tf.int32))

    # But it can't work with 100% certainty if full shape information is not
    # known ahead of time.
    # And it can't detect invalid indices at runtime, either.
    indexing_ops.batched_index(values=[[5, 5, 5]], indices=[1000000000])

  def testFullShapeAvailableAtRuntimeOnly(self):
    """What happens when shape information isn't available statically?

    The short answer is: it still works. The long answer is: it still works, but
    arguments that shouldn't work due to argument shape mismatch can sometimes
    work without raising any errors! This can cause insidious bugs. This test
    verifies correct behaviour and also demonstrates kinds of shape mismatch
    that can go undetected. Look for `!!!DANGER!!!` below.

    Why this is possible: internally, `batched_index` flattens its inputs,
    then transforms the action indices you provide into indices into its
    flattened Q values tensor. So long as these flattened indices don't go
    out-of-bounds, and so long as your arguments are compatible with a few
    other bookkeeping operations, the operation will succeed.

    The moral: always provide as much shape information as you can! See also
    `testInputShapeChecks` for more on what shape checking can accomplish when
    only partial shape information is available.
    """

    ## 1. No shape information is available during construction time.
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32),
                                  tf.TensorSpec(None, tf.int32)])
    def f1(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    self.assertAllClose(
        [51],
        f1(q_values=[[50, 51]], actions=[1]))
    self.assertAllClose(
        [[51, 52]],
        f1(q_values=[[[50, 51], [52, 53]]], actions=[[1, 0]]))

    # !!!DANGER!!! These "incompatible" shapes are silently tolerated!
    # (These examples are probably not exhaustive, either!)
    qs_2x2 = [[5, 5], [5, 5]]
    qs_2x2x2 = [[[5, 5], [5, 5]],
                [[5, 5], [5, 5]]]
    f1(q_values=qs_2x2, actions=[0])
    f1(q_values=qs_2x2, actions=0)
    f1(q_values=qs_2x2x2, actions=[[0]])
    f1(q_values=qs_2x2x2, actions=[0])
    f1(q_values=qs_2x2x2, actions=0)

    ## 2a. Shape information is only known for the batch size (2-D case).
    @tf.function(input_signature=[
        tf.TensorSpec([2, None], tf.float32),
        tf.TensorSpec([2], tf.int32),
    ])
    def f2(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    # Correct and compatible Q values and indices work as intended.
    self.assertAllClose(
        [51, 52],
        f2(q_values=[[50, 51], [52, 53]], actions=[1, 0]))
    # There are no really terrible shape errors that go uncaught in this case.

    ## 2b. Shape information is only known for the batch size (3-D case).
    @tf.function(input_signature=[tf.TensorSpec([None, 2, None], tf.float32),
                                  tf.TensorSpec([None, 2], tf.int32)])
    def f3(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    # Correct and compatible Q values and indices work as intended.
    self.assertAllClose(
        [[51, 52]],
        f3(q_values=[[[50, 51], [52, 53]]], actions=[[1, 0]]))

    # !!!DANGER!!! This "incompatible" shape is silently tolerated!
    f3(q_values=qs_2x2x2, actions=[[0, 0]])

    ## 3. Shape information is only known for the sequence length.
    @tf.function(input_signature=[tf.TensorSpec([2, None, None], tf.float32),
                                  tf.TensorSpec([2, None], tf.int32)])
    def f4(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    # Correct and compatible Q values and indices work as intended.
    self.assertAllClose(
        [[51, 52], [54, 57]],
        f4(q_values=[[[50, 51], [52, 53]], [[54, 55], [56, 57]]],
           actions=[[1, 0], [0, 1]]))

    # !!!DANGER!!! This "incompatible" shape is silently tolerated!
    f4(q_values=qs_2x2x2, actions=[[0], [0]])

    ## 4a. Shape information is only known for the number of actions (2-D case).
    @tf.function(input_signature=[tf.TensorSpec([None, 2], tf.float32),
                                  tf.TensorSpec([None], tf.int32)])
    def f5(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    # Correct and compatible Q values and indices work as intended.
    self.assertAllClose(
        [51, 52],
        f5(q_values=[[50, 51], [52, 53]], actions=[1, 0]))

    # !!!DANGER!!! This "incompatible" shape is silently tolerated!
    f5(q_values=qs_2x2, actions=[0])

    ## 4b. Shape information is only known for the number of actions (3-D case).
    @tf.function(input_signature=[tf.TensorSpec([None, None, 2], tf.float32),
                                  tf.TensorSpec([None, None], tf.int32)])
    def f6(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    self.assertAllClose(
        [[51, 52]],
        f6(q_values=[[[50, 51], [52, 53]]], actions=[[1, 0]]))

    # !!!DANGER!!! These "incompatible" shapes are silently tolerated!
    f6(q_values=qs_2x2x2, actions=[[0, 0]])
    f6(q_values=qs_2x2x2, actions=[[0]])

    ## 5a. Value shape is not known ahead of time.
    @tf.function(input_signature=[tf.TensorSpec(None, tf.float32),
                                  tf.TensorSpec([2], tf.int32)])
    def f7(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    # Correct and compatible Q values and indices work as intended.
    self.assertAllClose(
        [51, 52],
        f7(q_values=[[50, 51], [52, 53]], actions=[1, 0]))

    # !!!DANGER!!! This "incompatible" shape is silently tolerated!
    f7(q_values=qs_2x2x2, actions=[0, 0])

    ## 5b. Action shape is not known ahead of time.
    @tf.function(input_signature=[tf.TensorSpec([None, None, 2], tf.float32),
                                  tf.TensorSpec(None, tf.int32)])
    def f8(q_values, actions):
      return indexing_ops.batched_index(q_values, actions)

    self.assertAllClose(
        [[51, 52], [54, 57]],
        f8(q_values=[[[50, 51], [52, 53]], [[54, 55], [56, 57]]],
           actions=[[1, 0], [0, 1]]))

    # !!!DANGER!!! This "incompatible" shape is silently tolerated!
    f8(q_values=qs_2x2x2, actions=[0, 0])


if __name__ == '__main__':
  tf.enable_v2_behavior()
  tf.test.main()
