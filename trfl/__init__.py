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
"""Flattened namespace for trfl."""

from trfl.action_value_ops import double_qlearning
from trfl.action_value_ops import persistent_qlearning
from trfl.action_value_ops import qlambda
from trfl.action_value_ops import qlearning
from trfl.action_value_ops import sarsa
from trfl.action_value_ops import sarse
from trfl.base_ops import assert_rank_and_shape_compatibility
from trfl.base_ops import best_effort_shape
from trfl.discrete_policy_gradient_ops import discrete_policy_entropy_loss
from trfl.discrete_policy_gradient_ops import discrete_policy_gradient
from trfl.discrete_policy_gradient_ops import discrete_policy_gradient_loss
from trfl.discrete_policy_gradient_ops import sequence_advantage_actor_critic_loss
from trfl.dist_value_ops import categorical_dist_double_qlearning
from trfl.dist_value_ops import categorical_dist_qlearning
from trfl.dist_value_ops import categorical_dist_td_learning
from trfl.dpg_ops import dpg
from trfl.indexing_ops import batched_index
from trfl.periodic_ops import periodically
from trfl.pixel_control_ops import pixel_control_loss
from trfl.pixel_control_ops import pixel_control_rewards
from trfl.policy_gradient_ops import policy_entropy_loss
from trfl.policy_gradient_ops import policy_gradient
from trfl.policy_gradient_ops import policy_gradient_loss
from trfl.policy_gradient_ops import sequence_a2c_loss
from trfl.retrace_ops import retrace
from trfl.retrace_ops import retrace_core
from trfl.sequence_ops import multistep_forward_view
from trfl.sequence_ops import scan_discounted_sum
from trfl.value_ops import generalized_lambda_returns
from trfl.value_ops import td_lambda
from trfl.value_ops import td_learning
from trfl.vtrace_ops import vtrace_from_importance_weights
from trfl.vtrace_ops import vtrace_from_logits
