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

"""Trajectory-level statistics derived from a rollout batch.

Two quantities are needed before advantages can be computed:

* the **discounted return** behind every step, which is the raw score the
  step-level groups get normalised against, and
* the **batch success rate**, which gates whether uncertainty shaping is
  applied at all.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:  # pragma: no cover - keeps the package importable without verl
    from verl import DataProto

__all__ = ["compute_step_discounted_returns", "compute_trajectory_success_rate"]


def compute_step_discounted_returns(batch: "DataProto", gamma: float) -> torch.Tensor:
    """Compute the discounted return-to-go behind every step of the batch.

    Each trajectory is scanned backwards so that step ``t`` receives
    ``r_t + gamma * G_{t+1}``. The result is what the step-level groups treat
    as that step's score.

    Args:
        batch: Rollout batch carrying ``rewards``, ``traj_uid`` and
            ``active_masks`` in ``non_tensor_batch``.
        gamma: Discount factor.

    Returns:
        Per-step returns, shape ``(B,)``, on the same device as the batch's
        ``input_ids``.

    Raises:
        AssertionError: If a trajectory contains inactive steps. Padding is
            expected to have been dropped before this point, so an inactive
            step here means the batch was assembled incorrectly.
    """
    rewards = batch.non_tensor_batch["rewards"].astype(np.float32)
    traj_uids = batch.non_tensor_batch["traj_uid"]
    active_masks = batch.non_tensor_batch["active_masks"].astype(np.float32)

    all_returns = np.zeros_like(rewards)
    for uid in np.unique(traj_uids):
        rows = np.where(traj_uids == uid)[0]
        assert active_masks[rows].all(), (
            "active_masks should be all 1s within a trajectory"
        )

        running_return = 0.0
        traj_rewards = rewards[rows]
        traj_returns = np.zeros_like(traj_rewards)
        for t in reversed(range(len(traj_rewards))):
            running_return = traj_rewards[t] + gamma * running_return
            traj_returns[t] = running_return

        all_returns[rows] = traj_returns

    return torch.tensor(
        all_returns, dtype=torch.float32, device=batch.batch["input_ids"].device
    )


def compute_trajectory_success_rate(
    success: np.ndarray, traj_index: np.ndarray
) -> float:
    """Average episode success over the batch, weighting every episode equally.

    Averaging the flat per-step ``success`` array directly would over-weight
    long episodes, so the value is reduced within each trajectory first.

    Args:
        success: Per-step episode success flag, shape ``(B,)``.
        traj_index: Per-step trajectory id, shape ``(B,)``.

    Returns:
        The batch success rate in ``[0, 1]``.
    """
    per_trajectory = [
        np.mean(success[np.where(traj_index == uid)[0]])
        for uid in np.unique(traj_index)
    ]
    return float(np.mean(per_trajectory))
