# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SELAUR — Self-Evolving LLM Agent via Uncertainty-aware Rewards.

SELAUR is a critic-free RL algorithm for multi-turn LLM agents. It extends
GiGPO's two-level group advantage with a third signal read straight off the
rollout: how certain the policy was while it acted.

Typical use from the trainer::

    from selaur import SELAURConfig, compute_selaur_outcome_advantage

    settings = SELAURConfig.from_hydra(config)
    result = compute_selaur_outcome_advantage(
        token_level_rewards=batch.batch["token_level_rewards"],
        step_rewards=batch.batch["step_rewards"],
        response_mask=batch.batch["response_mask"],
        anchor_obs=batch.non_tensor_batch["anchor_obs"],
        index=batch.non_tensor_batch["uid"],
        traj_index=batch.non_tensor_batch["traj_uid"],
        logprobs=batch.non_tensor_batch["logprobs"],
        success=batch.non_tensor_batch["success"],
        settings=settings,
    )

Module map:

``selaur.advantage``
    The estimator itself; start here.
``selaur.config``
    Typed view over the ``selaur.*`` Hydra section.
``selaur.uncertainty``
    Signals, trajectory smoothing, shaping transforms, and the estimator that
    ties them together.
``selaur.grouping``
    Anchor-state grouping for the step level.
``selaur.normalization``
    The shared group-baseline operation.
``selaur.returns``
    Discounted step returns and the batch success rate.
``selaur.baselines``
    GiGPO, expressed as SELAUR with shaping switched off.

See ``selaur/README.md`` for the algorithm walkthrough.
"""

from selaur.advantage import AdvantageOutput, compute_selaur_outcome_advantage
from selaur.config import SELAURConfig
from selaur.grouping import build_step_group
from selaur.normalization import SingletonPolicy, group_normalize
from selaur.returns import (
    compute_step_discounted_returns,
    compute_trajectory_success_rate,
)
from selaur.uncertainty import UncertaintyEstimator, UncertaintyTrace

__all__ = [
    "AdvantageOutput",
    "SELAURConfig",
    "SingletonPolicy",
    "UncertaintyEstimator",
    "UncertaintyTrace",
    "build_step_group",
    "compute_selaur_outcome_advantage",
    "compute_step_discounted_returns",
    "compute_trajectory_success_rate",
    "group_normalize",
]
