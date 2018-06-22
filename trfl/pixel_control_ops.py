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
"""TensorFlow ops for implementing Pixel Control.

Pixel Control is an auxiliary task introduced in the UNREAL agent.
In Pixel Control an additional agent head is trained off-policy to predict
action-value functions for a host of pseudo rewards derived from the stream of
observations. This leads to better state representations and therefore improved
performance, both in terms of data efficiency and final performance.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Dependency imports
import sonnet as snt
import tensorflow as tf
from trfl import action_value_ops
from trfl import base_ops


PixelControlExtra = collections.namedtuple(
    "pixel_control_extra", ["spatial_loss", "pseudo_rewards"])


def pixel_control_rewards(observations, cell_size):
  """Calculates pixel control task rewards from observation sequence.

  The observations are first split in a grid of KxK cells. For each cell a
  distinct pseudo reward is computed as the average absolute change in pixel
  intensity for all pixels in the cell. The change in intensity is averaged
  across both pixels and channels (e.g. RGB).

  The `observations` provided to this function should be cropped suitably, to
  ensure that the observations' height and width are a multiple of `cell_size`.
  The values of the `observations` tensor should be rescaled to [0, 1]. In the
  UNREAL agent observations are cropped to 80x80, and each cell is 4x4 in size.

  See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
  Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

  Args:
    observations: A tensor of shape `[T+1,B,H,W,C...]`, where
      * `T` is the sequence length, `B` is the batch size.
      * `H` is height, `W` is width.
      * `C...` is at least one channel dimension (e.g., colour, stack).
      * `T` and `B` can be statically unknown.
    cell_size: The size of each cell.

  Returns:
    A tensor of pixel control rewards calculated from the observation. The
    shape is `[T,B,H',W']`, where `H'` and `W'` are determined by the
    `cell_size`. If evenly-divisible, `H' = H/cell_size`, and similar for `W`.
  """
  # Calculate the absolute differences across the sequence.
  abs_observation_diff = tf.abs(observations[1:] - observations[:-1])
  # Average over cells. abs_observation_diff has shape [T,B,H,W,C...], e.g.,
  # [T,B,H,W,C] if we have a colour channel. We want to use the TF avg_pool
  # op, but it expects 4D inputs. We collapse T and B then collapse all channel
  # dimensions. After pooling, we can then undo the sequence/batch collapse.
  obs_shape = abs_observation_diff.get_shape().as_list()
  # Collapse sequence and batch into one: [TB,H,W,C...].
  abs_diff = tf.reshape(abs_observation_diff, [-1] + obs_shape[2:])
  # Merge remaining dimensions after W: [TB,H,W,C'].
  abs_diff = snt.FlattenTrailingDimensions(dim_from=3)(abs_diff)
  # Apply the averaging using average pooling and reducing over channel.
  avg_abs_diff = tf.nn.avg_pool(
      abs_diff,
      ksize=[1, cell_size, cell_size, 1],
      strides=[1, cell_size, cell_size, 1],
      padding="VALID")  # [TB, H', W', C'].
  avg_abs_diff = tf.reduce_mean(avg_abs_diff, axis=[3])  # [TB,H',W'].
  # Restore sequence and batch dimensions, and static shape info where possible.
  pseudo_rewards = tf.reshape(
      avg_abs_diff, [
          tf.shape(abs_observation_diff)[0], tf.shape(abs_observation_diff)[1],
          tf.shape(avg_abs_diff)[1], tf.shape(avg_abs_diff)[2]
      ],
      name="pseudo_rewards")  # [T,B,H',W'].
  sequence_batch = abs_observation_diff.get_shape()[:2]
  new_height_width = avg_abs_diff.get_shape()[1:]
  pseudo_rewards.set_shape(sequence_batch.concatenate(new_height_width))
  return pseudo_rewards


def pixel_control_loss(
    observations, actions, action_values, cell_size, discount_factor,
    scale, crop_height_dim=(None, None), crop_width_dim=(None, None)):
  """Calculate n-step Q-learning loss for pixel control auxiliary task.

  For each pixel-based pseudo reward signal, the corresponding action-value
  function is trained off-policy, using Q(lambda). A discount of 0.9 is
  commonly used for learning the value functions.

  Note that, since pseudo rewards have a spatial structure, with neighbouring
  cells exhibiting strong correlations, it is convenient to predict the action
  values for all the cells through a deconvolutional head.

  See "Reinforcement Learning with Unsupervised Auxiliary Tasks" by Jaderberg,
  Mnih, Czarnecki et al. (https://arxiv.org/abs/1611.05397).

  Args:
    observations: A tensor of shape `[T+1,B, ...]`; `...` is the observation
      shape, `T` the sequence length, and `B` the batch size. `T` and `B` can
      be statically unknown for `observations`, `actions` and `action_values`.
    actions: A tensor, shape `[T,B]`, of the actions across each sequence.
    action_values: A tensor, shape `[T+1,B,H,W,N]` of pixel control action
      values, where `H`, `W` are the number of pixel control cells/tasks, and
      `N` is the number of actions.
    cell_size: size of the cells used to derive the pixel based pseudo-rewards.
    discount_factor: discount used for learning the value function associated
      to the pseudo rewards; must be a scalar or a Tensor of shape [T,B].
    scale: scale factor for pixels in `observations`.
    crop_height_dim: tuple (min_height, max_height) specifying how
      to crop the input observations before computing the pseudo-rewards.
    crop_width_dim: tuple (min_width, max_width) specifying how
      to crop the input observations before computing the pseudo-rewards.

  Returns:
    A namedtuple with fields:

    * `loss`: a tensor containing the batch of losses, shape [B].
    * `extra`: a namedtuple with fields:
        * `target`: batch of target values for `q_tm1[a_tm1]`, shape [B].
        * `td_error`: batch of temporal difference errors, shape [B].

  Raises:
    ValueError: if the shape of `action_values` is not compatible with that of
      the pseudo-rewards derived from the observations.
  """
  # Useful shapes.
  sequence_length, batch_size = base_ops.best_effort_shape(actions)
  num_actions = action_values.get_shape().as_list()[-1]
  height_width_q = action_values.get_shape().as_list()[2:-1]
  # Calculate rewards using the observations. Crop observations if appropriate.
  if crop_height_dim[0] is not None:
    h_low, h_high = crop_height_dim
    observations = observations[:, :, h_low:h_high, :]
  if crop_width_dim[0] is not None:
    w_low, w_high = crop_width_dim
    observations = observations[:, :, :, w_low:w_high]
  # Rescale observations by a constant factor.
  observations *= tf.constant(scale)
  # Compute pseudo-rewards and get their shape.
  pseudo_rewards = pixel_control_rewards(observations, cell_size)
  height_width = pseudo_rewards.get_shape().as_list()[2:]
  # Check that pseudo-rewards and Q-values are compatible in shape.
  if height_width != height_width_q:
    raise ValueError(
        "Pixel Control values are not compatible with the shape of the"
        "pseudo-rewards derived from the observation. Pseudo-rewards have shape"
        "{}, while Pixel Control values have shape {}".format(
            height_width, height_width_q))
  # We now have Q(s,a) and rewards, so can calculate the n-step loss. The
  # QLambda loss op expects inputs of shape [T,B,N] and [T,B], but our tensors
  # are in a variety of incompatible shapes. The state-action values have
  # shape [T,B,H,W,N] and rewards have shape [T,B,H,W]. We can think of the
  # [H,W] dimensions as extra batch dimensions for the purposes of the loss
  # calculation, so we first collapse [B,H,W] into a single dimension.
  q_tm1 = tf.reshape(
      action_values[:-1],  # [T,B,H,W,N].
      [sequence_length, -1, num_actions],
      name="q_tm1")  # [T,BHW,N].
  r_t = tf.reshape(
      pseudo_rewards,  # [T,B,H,W].
      [sequence_length, -1],
      name="r_t")  # [T,BHW].
  q_t = tf.reshape(
      action_values[1:],  # [T,B,H,W,N].
      [sequence_length, -1, num_actions],
      name="q_t")  # [T,BHW,N].
  # The actions tensor is of shape [T,B], and is the same for each H and W.
  # We thus expand it to be same shape as the reward tensor, [T,BHW].
  expanded_actions = tf.expand_dims(tf.expand_dims(actions, -1), -1)
  a_tm1 = tf.tile(
      expanded_actions, multiples=[1, 1] + height_width)  # [T,B,H,W].
  a_tm1 = tf.reshape(a_tm1, [sequence_length, -1])  # [T,BHW].
  # We similarly expand-and-tile the discount to [T,BHW].
  discount_factor = tf.convert_to_tensor(discount_factor)
  if discount_factor.shape.ndims == 0:
    pcont_t = tf.reshape(discount_factor, [1, 1])  # [1,1].
    pcont_t = tf.tile(pcont_t, tf.shape(a_tm1))  # [T,BHW].
  elif discount_factor.shape.ndims == 2:
    tiled_pcont = tf.tile(
        tf.expand_dims(tf.expand_dims(discount_factor, -1), -1),
        [1, 1] + height_width)
    pcont_t = tf.reshape(tiled_pcont, [sequence_length, -1])
  else:
    raise ValueError(
        "The discount_factor must be a scalar or a tensor of rank 2."
        "instead is a tensor of shape {}".format(
            discount_factor.shape.as_list()))
  # Compute a QLambda loss of shape [T,BHW]
  loss, _ = action_value_ops.qlambda(q_tm1, a_tm1, r_t, pcont_t, q_t, lambda_=1)
  # Take sum over sequence, sum over cells.
  expanded_shape = [sequence_length, batch_size] + height_width
  spatial_loss = tf.reshape(loss, expanded_shape)  # [T,B,H,W].
  # Return.
  extra = PixelControlExtra(
      spatial_loss=spatial_loss, pseudo_rewards=pseudo_rewards)
  return base_ops.LossOutput(
      tf.reduce_sum(spatial_loss, axis=[0, 2, 3]), extra)  # [B]
