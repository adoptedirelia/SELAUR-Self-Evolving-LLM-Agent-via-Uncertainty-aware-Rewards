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

"""The SELAUR advantage estimator.

SELAUR keeps GiGPO's two-level, critic-free advantage and adds a third input to
it: how *certain* the policy was while acting.

.. code-block:: text

    token_level_rewards ──► episode score ─┐
                                           ├─► group-normalise ─► episode adv ─┐
    anchor states ──► step groups ─────────┘                                   │
                                                                               ├─► advantage
    step returns ──────► step score ───────┬─► group-normalise ─► step adv ────┘
                                           │                        × step_advantage_w
    rollout logprobs ──► uncertainty ──────┘
                         (shapes both scores, same direction)

Both levels are baselined by the mean of their group, so no value network is
needed. Uncertainty enters as a shaping term on the scores *before*
normalisation, which keeps it inside the baseline: shifting a whole group by
the same amount changes nothing, so only *relative* certainty within a group
moves the gradient. Both levels are shaped in the same direction -- which
direction that is, is the transform's decision, not the level's.

The public entry point is :func:`compute_selaur_outcome_advantage`. The GiGPO
baseline in :mod:`selaur.baselines.gigpo` reuses the same machinery with
shaping switched off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from selaur.config import SELAURConfig
from selaur.grouping import build_step_group
from selaur.normalization import (
    SingletonPolicy,
    broadcast_to_tokens,
    group_normalize,
)
from selaur.returns import compute_trajectory_success_rate
from selaur.typedefs import LogProbsBatch, SuccessFlags
from selaur.uncertainty.estimator import UncertaintyTrace
from selaur.uncertainty.transforms import ADDITIVE_TRANSFORMS

__all__ = ["AdvantageOutput", "compute_selaur_outcome_advantage"]

#: Normalisation modes accepted by ``algorithm.gigpo.mode``, mapped to whether
#: the group standard deviation is divided out.
_NORMALIZATION_MODES = {"mean_norm": True, "mean_std_norm": False}


@dataclass
class AdvantageOutput:
    """Result of one advantage computation.

    Attributes:
        advantages: Token-level advantages, shape ``(B, R)``.
        returns: Token-level returns, shape ``(B, R)``. Identical to
            :attr:`advantages`: without a critic there is no separate return
            target, and the field exists so the trainer's PPO path stays
            uniform across estimators.
        step_uncertainty: Step-level diagnostic uncertainty, split into
            ``"success"`` and ``"failure"`` arrays.
        episode_uncertainty: Episode-level diagnostic uncertainty, split the
            same way.
    """

    advantages: torch.Tensor
    returns: torch.Tensor
    step_uncertainty: Dict[str, np.ndarray]
    episode_uncertainty: Dict[str, np.ndarray]


def compute_selaur_outcome_advantage(
    token_level_rewards: torch.Tensor,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    anchor_obs: np.ndarray,
    index: np.ndarray,
    traj_index: np.ndarray,
    logprobs: LogProbsBatch,
    success: SuccessFlags,
    *,
    settings: SELAURConfig,
    step_advantage_w: float = 1.0,
    mode: str = "mean_norm",
    epsilon: float = 1e-6,
    apply_uncertainty_shaping: bool = True,
    baseline_over_trajectories: bool = False,
) -> AdvantageOutput:
    """Compute two-level, uncertainty-aware advantages for a rollout batch.

    Args:
        token_level_rewards: Episode-level token rewards, shape ``(B, R)``.
            Summed over tokens to give each step's episode score.
        step_rewards: Discounted return behind each step, shape ``(B,)``; see
            :func:`~selaur.returns.compute_step_discounted_returns`.
        response_mask: Token mask, shape ``(B, R)``.
        anchor_obs: Per-step anchor observations used to form step-level
            groups, shape ``(B,)``.
        index: Episode group id (prompt ``uid``) per step, shape ``(B,)``.
        traj_index: Trajectory id per step, shape ``(B,)``.
        logprobs: Rollout log-probabilities per step; see
            :mod:`selaur.uncertainty.signals`.
        success: Per-step episode success flag, shape ``(B,)``.
        settings: Uncertainty-shaping settings.
        step_advantage_w: Weight of the step-level advantage in the sum.
        mode: ``"mean_norm"`` to subtract the group mean only, or
            ``"mean_std_norm"`` to also divide by the group standard deviation.
        epsilon: Guard added to the standard deviation in ``mean_std_norm``.
        apply_uncertainty_shaping: Set to ``False`` to recover plain GiGPO.
            Uncertainty is still measured and reported, but does not touch the
            scores. This is what :mod:`selaur.baselines.gigpo` uses.
        baseline_over_trajectories: If ``True``, the episode-level baseline is
            computed over trajectories instead of over steps, so that long
            episodes do not dominate the group mean. Defaults to ``False``,
            which pools every step and is the more stable of the two.

    Returns:
        The advantages, returns and uncertainty diagnostics for this batch.

    Raises:
        ValueError: If ``mode`` is not a known normalisation mode.
    """
    if mode not in _NORMALIZATION_MODES:
        raise ValueError(
            f"Unknown mode {mode!r}. Choose from {tuple(_NORMALIZATION_MODES)}."
        )
    remove_std = _NORMALIZATION_MODES[mode]

    estimator = settings.build_estimator()
    trace = estimator.compute(logprobs, success, traj_index)

    # The gate is a property of the batch, not of a single episode: shaping is
    # off until the policy solves the task often enough for "how sure was it"
    # to carry information.
    success_rate = compute_trajectory_success_rate(success, traj_index)
    shape_scores = (
        apply_uncertainty_shaping and success_rate > settings.threshold_success_rate
    )

    episode_advantages = _episode_advantages(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
        traj_index=traj_index,
        trace=trace,
        settings=settings,
        shape_scores=shape_scores,
        remove_std=remove_std,
        epsilon=epsilon,
        baseline_over_trajectories=baseline_over_trajectories,
    )

    step_advantages = _step_advantages(
        step_rewards=step_rewards,
        response_mask=response_mask,
        step_group_ids=build_step_group(anchor_obs, index),
        trace=trace,
        settings=settings,
        shape_scores=shape_scores,
        remove_std=remove_std,
        epsilon=epsilon,
    )

    advantages = episode_advantages + step_advantage_w * step_advantages
    return AdvantageOutput(
        advantages=advantages,
        returns=advantages,
        step_uncertainty=trace.step_diagnostics.as_arrays(),
        episode_uncertainty=trace.episode_diagnostics.as_arrays(),
    )


def _episode_advantages(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    trace: UncertaintyTrace,
    settings: SELAURConfig,
    shape_scores: bool,
    remove_std: bool,
    epsilon: float,
    baseline_over_trajectories: bool,
) -> torch.Tensor:
    """Normalise episode outcomes against the other rollouts of the same task.

    The episode score is the summed token reward. Uncertainty is subtracted
    from it, so an uncertain episode is scored *below* an equally-rewarded
    confident one — at this level the policy is asked to convert luck into
    conviction.

    Args:
        token_level_rewards: Episode-level token rewards, shape ``(B, R)``.
        response_mask: Token mask, shape ``(B, R)``.
        index: Episode group id per step, shape ``(B,)``.
        traj_index: Trajectory id per step, shape ``(B,)``.
        trace: Uncertainty computed for this batch.
        settings: Uncertainty-shaping settings.
        shape_scores: Whether shaping is active for this batch.
        remove_std: Whether to skip the standard-deviation division.
        epsilon: Guard added to the standard deviation.
        baseline_over_trajectories: Deduplicate steps of the same trajectory
            before computing group statistics.

    Returns:
        Token-level episode advantages, shape ``(B, R)``.
    """
    scores = token_level_rewards.sum(dim=-1)
    if shape_scores:
        scores = _shape(scores, trace.episode, settings)

    normalized = group_normalize(
        scores,
        index,
        remove_std=remove_std,
        singleton_policy=SingletonPolicy.ZERO_BASELINE,
        epsilon=epsilon,
        dedup_keys=traj_index if baseline_over_trajectories else None,
    )
    return broadcast_to_tokens(normalized, response_mask)


def _step_advantages(
    *,
    step_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    step_group_ids: np.ndarray,
    trace: UncertaintyTrace,
    settings: SELAURConfig,
    shape_scores: bool,
    remove_std: bool,
    epsilon: float,
) -> torch.Tensor:
    """Normalise each step against other steps taken from the same state.

    The step score is the discounted return behind the step, shaped by that
    step's smoothed uncertainty in the same direction as the episode level and
    scaled by ``step_weight``.

    Args:
        step_rewards: Discounted return behind each step, shape ``(B,)``.
        response_mask: Token mask, shape ``(B, R)``.
        step_group_ids: Anchor-state group id per step, shape ``(B,)``.
        trace: Uncertainty computed for this batch.
        settings: Uncertainty-shaping settings.
        shape_scores: Whether shaping is active for this batch.
        remove_std: Whether to skip the standard-deviation division.
        epsilon: Guard added to the standard deviation.

    Returns:
        Token-level step advantages, shape ``(B, R)``.
    """
    scores = step_rewards.clone()
    if shape_scores:
        scores = _shape(scores, trace.step, settings)

    normalized = group_normalize(
        scores,
        step_group_ids,
        remove_std=remove_std,
        singleton_policy=SingletonPolicy.SELF_BASELINE,
        epsilon=epsilon,
    )
    return broadcast_to_tokens(normalized, response_mask)


def _shape(
    scores: torch.Tensor,
    uncertainty: np.ndarray,
    settings: SELAURConfig,
) -> torch.Tensor:
    """Fold the uncertainty term into a group score.

    Additive transforms (``linear``, ``neg``) contribute ``+shaping_scale * u``;
    the multiplicative transform (``exp``) scales the score by ``u``, which lies
    in ``(0, 1]`` and therefore shrinks rather than shifts it.

    Both levels shape in the *same* direction. The direction itself is the
    transform's job: ``linear`` credits uncertainty, ``neg`` charges for it.
    Keeping the sign here fixed means an ablation only has to flip one knob.

    Args:
        scores: Group scores, shape ``(B,)``.
        uncertainty: Matching shaping term, shape ``(B,)``. Episodes the
            configured transform does not apply to already carry its neutral
            term, so they pass through unchanged.
        settings: Uncertainty-shaping settings.

    Returns:
        The shaped scores, shape ``(B,)``.
    """
    term = torch.as_tensor(uncertainty, dtype=scores.dtype, device=scores.device)
    if settings.transform in ADDITIVE_TRANSFORMS:
        return scores + settings.shaping_scale * term
    return scores * term
