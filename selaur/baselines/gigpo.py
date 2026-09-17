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

"""GiGPO: the baseline SELAUR is measured against.

`GiGPO <https://arxiv.org/abs/2505.10978>`_ contributes the two-level,
critic-free advantage — episode groups from shared prompts, step groups from
shared anchor states — that SELAUR builds on. The difference between the two
methods is exactly one thing: whether policy uncertainty is allowed to shape
the group scores.

This module therefore does not reimplement anything. It calls
:func:`~selaur.advantage.compute_selaur_outcome_advantage` with
``apply_uncertainty_shaping=False``, so that:

* the advantages are plain GiGPO, and
* the uncertainty diagnostics are still computed and logged, which is what
  makes the two runs directly comparable in W&B.

Keeping one implementation for both means an improvement to the grouping or the
normalisation cannot silently apply to only one arm of the comparison.
"""

from __future__ import annotations

import numpy as np
import torch

from selaur.advantage import AdvantageOutput, compute_selaur_outcome_advantage
from selaur.config import SELAURConfig
from selaur.typedefs import LogProbsBatch, SuccessFlags

__all__ = ["GIGPO_DIAGNOSTIC_SETTINGS", "compute_gigpo_outcome_advantage"]

#: Settings used purely to produce comparable uncertainty *diagnostics* for the
#: baseline. They are deliberately fixed rather than read from the config: the
#: baseline must not change when a SELAUR hyper-parameter sweep changes, and
#: with shaping off none of these values can affect its advantages.
GIGPO_DIAGNOSTIC_SETTINGS = SELAURConfig(
    use_combined=True,
    w1=1.0 / 3.0,
    w2=1.0 / 3.0,
    w3=1.0 / 3.0,
    rho=1.0,
    lambda_uq=0.95,
    step_weight=0.85,
    transform="linear",
)


def compute_gigpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    logprobs: LogProbsBatch,
    success: SuccessFlags,
    *,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    epsilon: float = 1e-6,
) -> AdvantageOutput:
    """Compute GiGPO advantages, with SELAUR's shaping switched off.

    Args:
        token_level_rewards: Episode-level token rewards, shape ``(B, R)``.
        step_rewards: Discounted return behind each step, shape ``(B,)``.
        response_mask: Token mask, shape ``(B, R)``.
        anchor_obs: Per-step anchor observations, shape ``(B,)``.
        index: Episode group id (prompt ``uid``) per step, shape ``(B,)``.
        traj_index: Trajectory id per step, shape ``(B,)``.
        logprobs: Rollout log-probabilities per step, used for diagnostics only.
        success: Per-step episode success flag, shape ``(B,)``.
        step_advantage_w: Weight of the step-level advantage in the sum.
        mode: ``"mean_norm"`` or ``"mean_std_norm"``.
        epsilon: Guard added to the standard deviation in ``mean_std_norm``.

    Returns:
        The advantages, returns and uncertainty diagnostics for this batch.
    """
    return compute_selaur_outcome_advantage(
        token_level_rewards=token_level_rewards,
        step_rewards=step_rewards,
        response_mask=response_mask,
        anchor_obs=anchor_obs,
        index=index,
        traj_index=traj_index,
        logprobs=logprobs,
        success=success,
        settings=GIGPO_DIAGNOSTIC_SETTINGS,
        step_advantage_w=step_advantage_w,
        mode=mode,
        epsilon=epsilon,
        apply_uncertainty_shaping=False,
    )
