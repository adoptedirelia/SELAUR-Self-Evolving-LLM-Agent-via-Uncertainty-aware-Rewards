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

"""The SELAUR uncertainty estimator.

:class:`UncertaintyEstimator` is the single object that turns raw rollout
log-probabilities into the two shaping terms the advantage estimator consumes:

1. **per-step** uncertainty, used to shape step-level group advantages, and
2. **per-episode** uncertainty, used to shape episode-level group advantages.

It owns the full pipeline for one training batch::

    logprobs --signal--> per-step u --smooth--> trace --transform--> shaping term

and applies it trajectory by trajectory, because both the smoothing and the
episode summary are only meaningful within a single episode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from selaur.typedefs import LogProbsBatch, LogProbsInput, SuccessFlags
from selaur.uncertainty.aggregation import discounted_smooth
from selaur.uncertainty.signals import SIGNAL_REGISTRY, combined_uncertainty
from selaur.uncertainty.transforms import (
    DIAGNOSTIC_TRANSFORM,
    apply_episode_transform,
    apply_step_transform,
    neutral_term,
    shapes_failures,
)

__all__ = ["OutcomeSplit", "UncertaintyTrace", "UncertaintyEstimator"]


@dataclass
class OutcomeSplit:
    """Diagnostic uncertainty values split by episode outcome.

    These are *observational* only: they are always computed with the
    ``linear`` transform regardless of what the run actually shapes with, so
    the logged curves stay comparable across ablations.

    Attributes:
        success: Values belonging to steps of successful episodes.
        failure: Values belonging to steps of failed episodes.
    """

    success: List[float] = field(default_factory=list)
    failure: List[float] = field(default_factory=list)

    def record(self, values: np.ndarray, succeeded: bool) -> None:
        """Append a trajectory's values to the matching bucket.

        Args:
            values: Values to record, one per step of the trajectory.
            succeeded: Whether the trajectory ended in success.
        """
        bucket = self.success if succeeded else self.failure
        bucket.extend(np.asarray(values, dtype=float).ravel().tolist())

    def as_arrays(self) -> Dict[str, np.ndarray]:
        """Return the two buckets as numpy arrays, ready for metric logging."""
        return {
            "success": np.array(self.success, dtype=float),
            "failure": np.array(self.failure, dtype=float),
        }


@dataclass
class UncertaintyTrace:
    """Everything :class:`UncertaintyEstimator` produces for one batch.

    Attributes:
        step: Step-level shaping term, shape ``(B,)``. Trajectories the
            configured transform does not apply to carry its *neutral* term
            (``0`` for the additive family, ``1`` for the multiplicative one),
            so folding the term into a score leaves them untouched.
        episode: Episode-level shaping term, shape ``(B,)``, constant within a
            trajectory. Neutral under the same condition as :attr:`step`.
        step_diagnostics: Un-gated step-level values split by outcome.
        episode_diagnostics: Un-gated episode-level values split by outcome.
    """

    step: np.ndarray
    episode: np.ndarray
    step_diagnostics: OutcomeSplit
    episode_diagnostics: OutcomeSplit


class UncertaintyEstimator:
    """Configurable policy-uncertainty estimator over rollout log-probabilities.

    Args:
        method: Name of the single signal to use when ``use_combined`` is
            ``False``; one of :data:`~selaur.uncertainty.signals.SIGNAL_REGISTRY`.
        step_weight: Strength of the step-level shaping term relative to the
            episode-level one.
        lambda_uq: Discount used by
            :func:`~selaur.uncertainty.aggregation.discounted_smooth` when
            spreading uncertainty backwards along a trajectory.
        rho: Mixing coefficient between the weighted average and the maximum in
            :func:`~selaur.uncertainty.signals.combined_uncertainty`.
        transform: Shaping transform; see :mod:`selaur.uncertainty.transforms`.
        use_combined: If ``True`` (the SELAUR default) blend all three signals;
            if ``False`` use the single signal named by ``method``.
        w1: Weight of the entropy signal in the blend.
        w2: Weight of the confidence-shortfall signal in the blend.
        w3: Weight of the top-2 margin signal in the blend.

    Raises:
        ValueError: If ``transform`` is unknown, or if ``method`` is unknown
            while ``use_combined`` is ``False``.
    """

    def __init__(
        self,
        method: str = "entropy",
        step_weight: float = 0.85,
        lambda_uq: float = 0.95,
        rho: float = 1.0,
        transform: str = "linear",
        use_combined: bool = True,
        w1: float = 1.0 / 3.0,
        w2: float = 1.0 / 3.0,
        w3: float = 1.0 / 3.0,
    ) -> None:
        if not use_combined and method not in SIGNAL_REGISTRY:
            raise ValueError(
                f"Unknown method {method!r}. Choose from {tuple(SIGNAL_REGISTRY)}."
            )
        # Validates ``transform`` and caches which outcome bucket it shapes.
        self._shapes_failures = shapes_failures(transform)

        self.method = method
        self.step_weight = float(step_weight)
        self.lambda_uq = float(lambda_uq)
        self.rho = float(rho)
        self.transform = transform
        self.use_combined = use_combined
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)

    def __repr__(self) -> str:
        signal = "combined" if self.use_combined else self.method
        return (
            f"{type(self).__name__}(signal={signal!r}, transform={self.transform!r}, "
            f"step_weight={self.step_weight}, lambda_uq={self.lambda_uq}, "
            f"rho={self.rho}, weights=({self.w1:.4g}, {self.w2:.4g}, {self.w3:.4g}))"
        )

    def score_step(self, logprobs: LogProbsInput, **signal_kwargs: Any) -> float:
        """Score the uncertainty of a single agent step.

        Args:
            logprobs: Log-probabilities of that step's generated response.
            **signal_kwargs: Extra keyword arguments forwarded to the signal.
                Only consulted when ``use_combined`` is ``False``.

        Returns:
            The step's scalar uncertainty. Higher means less decisive.
        """
        if self.use_combined:
            return float(
                combined_uncertainty(
                    logprobs, w1=self.w1, w2=self.w2, w3=self.w3, rho=self.rho
                )
            )
        return float(SIGNAL_REGISTRY[self.method](logprobs, **signal_kwargs).mean())

    def compute(
        self,
        logprobs: LogProbsBatch,
        success: SuccessFlags,
        traj_index: np.ndarray,
        **signal_kwargs: Any,
    ) -> UncertaintyTrace:
        """Compute step- and episode-level shaping terms for a whole batch.

        Trajectories are processed independently. For each one the per-step
        signal is smoothed along the episode, and the result is turned into a
        shaping term *only if* the configured transform is responsible for that
        trajectory's outcome (additive transforms shape failures,
        multiplicative ones shape successes). Trajectories the transform does
        not own receive its neutral term, so their advantages come through
        unchanged.

        Diagnostics are recorded for every trajectory either way, always under
        the ``linear`` transform.

        Args:
            logprobs: One log-prob container per step of the flat batch.
            success: Per-step episode success flag, shape ``(B,)``.
            traj_index: Per-step trajectory id, shape ``(B,)``.
            **signal_kwargs: Extra keyword arguments forwarded to the signal.

        Returns:
            The :class:`UncertaintyTrace` for this batch.
        """
        batch_size = len(logprobs)
        # Pre-fill with the neutral term so that skipped trajectories pass
        # through the shaping unchanged, whichever family the transform is in.
        neutral = neutral_term(self.transform)
        step_term = np.full(batch_size, neutral, dtype=float)
        episode_term = np.full(batch_size, neutral, dtype=float)
        step_diagnostics = OutcomeSplit()
        episode_diagnostics = OutcomeSplit()

        for uid in np.unique(traj_index):
            rows = np.where(traj_index == uid)[0]
            succeeded = bool(success[rows][0])

            raw = np.array(
                [self.score_step(logprobs[i], **signal_kwargs) for i in rows],
                dtype=float,
            )
            smoothed = discounted_smooth(raw, self.lambda_uq)
            episode_summary = float(np.mean(smoothed))

            # Only the transform that owns this outcome bucket gets to shape it.
            if succeeded != self._shapes_failures:
                step_term[rows] = apply_step_transform(
                    smoothed, self.transform, self.step_weight
                )
                episode_term[rows] = apply_episode_transform(
                    episode_summary, self.transform
                )

            step_diagnostics.record(
                apply_step_transform(
                    smoothed, DIAGNOSTIC_TRANSFORM, self.step_weight
                ),
                succeeded,
            )
            episode_diagnostics.record(
                np.full(
                    len(rows),
                    apply_episode_transform(episode_summary, DIAGNOSTIC_TRANSFORM),
                ),
                succeeded,
            )

        return UncertaintyTrace(
            step=step_term,
            episode=episode_term,
            step_diagnostics=step_diagnostics,
            episode_diagnostics=episode_diagnostics,
        )
