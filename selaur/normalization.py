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

"""Critic-free group normalisation.

Both halves of the algorithm reduce to the same operation: take a scalar score
per row, subtract the mean of the rows sharing a group, and optionally divide
by the group's standard deviation. The group mean *is* the baseline — that is
what lets GiGPO and SELAUR train without a value network.

The two halves differ only in what a group is (episode group vs. anchor-state
group) and in how a singleton group should behave, which is what
:class:`SingletonPolicy` captures.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = ["SingletonPolicy", "broadcast_to_tokens", "group_normalize"]


class SingletonPolicy(str, Enum):
    """How to baseline a group that contains a single row.

    A singleton group has no peers to compare against, so its "advantage" is
    arbitrary and the two levels of the algorithm resolve it differently.

    Attributes:
        ZERO_BASELINE: Subtract nothing, keeping the raw score. Used at the
            episode level, where a lone rollout still carries the only
            information available about that task.
        SELF_BASELINE: Subtract the row's own score, yielding exactly zero.
            Used at the step level, where an anchor state seen once offers no
            comparison and should not push the policy either way.
    """

    ZERO_BASELINE = "zero"
    SELF_BASELINE = "self"


def group_normalize(
    scores: torch.Tensor,
    group_ids: np.ndarray,
    *,
    remove_std: bool,
    singleton_policy: SingletonPolicy,
    epsilon: float = 1e-6,
    dedup_keys: Optional[Sequence] = None,
) -> torch.Tensor:
    """Centre each score against the mean of its group.

    Args:
        scores: One scalar score per row, shape ``(B,)``.
        group_ids: Group id per row, shape ``(B,)``. Rows sharing an id are
            normalised together.
        remove_std: If ``True`` (``mode="mean_norm"``) only the group mean is
            subtracted. If ``False`` (``mode="mean_std_norm"``) the centred
            score is additionally divided by the group standard deviation.
        singleton_policy: Baseline to use for groups of size one; see
            :class:`SingletonPolicy`.
        epsilon: Added to the standard deviation to avoid division by zero.
        dedup_keys: Optional per-row key, shape ``(B,)``. When given, only the
            first row of each ``(group_id, key)`` pair contributes to the
            group's statistics. This is how the episode level can compute its
            baseline over *trajectories* rather than over steps, so that long
            episodes do not dominate the mean.

    Returns:
        The normalised scores, shape ``(B,)``, on the input device and dtype.

    Raises:
        ValueError: If a group ends up with no contributing rows, which would
            mean ``group_ids`` and ``scores`` disagree in length.
    """
    with torch.no_grad():
        group_scores: Dict[object, List[torch.Tensor]] = defaultdict(list)
        seen: set = set()
        for row in range(scores.shape[0]):
            group = group_ids[row]
            if dedup_keys is not None:
                key = (group, dedup_keys[row])
                if key in seen:
                    continue
                seen.add(key)
            group_scores[group].append(scores[row])

        group_stats = {
            group: _group_statistics(group, values, singleton_policy)
            for group, values in group_scores.items()
        }

        means = torch.stack(
            [group_stats[group][0] for group in group_ids]
        ).to(device=scores.device, dtype=scores.dtype)
        centred = scores - means

        if remove_std:
            return centred

        stds = torch.stack(
            [group_stats[group][1] for group in group_ids]
        ).to(device=scores.device, dtype=scores.dtype)
        return centred / (stds + epsilon)


def _group_statistics(
    group: object,
    values: List[torch.Tensor],
    singleton_policy: SingletonPolicy,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the ``(mean, std)`` baseline for one group.

    Args:
        group: The group id, used only for error reporting.
        values: The scores contributing to this group.
        singleton_policy: Baseline to use when ``values`` has a single entry.

    Returns:
        The group mean and standard deviation as 0-d CPU tensors. The standard
        deviation of a singleton group is defined as ``1.0`` so that
        ``mean_std_norm`` degrades to plain centring rather than to ``nan``.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError(f"No score collected for group {group!r}.")

    stacked = torch.stack([value.detach().float().reshape(()) for value in values])
    if len(values) == 1:
        mean = (
            torch.tensor(0.0)
            if singleton_policy is SingletonPolicy.ZERO_BASELINE
            else stacked.mean()
        )
        return mean, torch.tensor(1.0)
    return stacked.mean(), stacked.std()


def broadcast_to_tokens(
    scores: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """Spread one advantage per step across that step's response tokens.

    Args:
        scores: Per-step advantage, shape ``(B,)``.
        response_mask: Token mask, shape ``(B, R)``; padding tokens are zeroed.

    Returns:
        Token-level advantages, shape ``(B, R)``.
    """
    response_length = response_mask.shape[-1]
    return scores.unsqueeze(-1).tile([1, response_length]) * response_mask
