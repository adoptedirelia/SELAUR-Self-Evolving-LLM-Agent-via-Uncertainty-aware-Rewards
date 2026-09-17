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

"""Transforms that turn a raw uncertainty trace into a reward-shaping term.

A transform decides *how* uncertainty enters the reward, and therefore also
*which* episodes it is allowed to touch:

``linear``
    Additive shaping, uncertainty kept as-is. Applied to **failed** episodes:
    an uncertain failure is a step the policy was not committed to, so the
    penalty it receives is softened.
``neg``
    Additive shaping with the sign flipped. Also applied to **failed**
    episodes, but it rewards *certainty* instead — the ablation that checks the
    direction of the effect actually matters.
``exp``
    Multiplicative shaping, ``exp(-u)`` in ``(0, 1]``. Applied to **successful**
    episodes, where it discounts a win that the policy stumbled into rather
    than chose.

Each transform owns exactly one outcome bucket, and the episodes it does not
own must come through untouched. What "untouched" means depends on the family,
which is why :func:`neutral_term` exists: adding ``0`` leaves a score alone,
but *multiplying* by ``0`` destroys it. The estimator fills the episodes a
transform skips with that transform's neutral term rather than with zeros.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "ADDITIVE_TRANSFORMS",
    "MULTIPLICATIVE_TRANSFORMS",
    "TRANSFORMS",
    "apply_episode_transform",
    "apply_step_transform",
    "neutral_term",
    "shapes_failures",
]

#: Transforms that add their term to the group score.
ADDITIVE_TRANSFORMS: Tuple[str, ...] = ("linear", "neg")

#: Transforms that multiply the group score by their term.
MULTIPLICATIVE_TRANSFORMS: Tuple[str, ...] = ("exp",)

#: Every transform SELAUR understands.
TRANSFORMS: Tuple[str, ...] = ADDITIVE_TRANSFORMS + MULTIPLICATIVE_TRANSFORMS

#: Transform used for the diagnostics that get logged to W&B. Held fixed so the
#: logged uncertainty curves stay comparable across runs that shape differently.
DIAGNOSTIC_TRANSFORM = "linear"


def neutral_term(transform: str) -> float:
    """Return the shaping term that leaves a score unchanged.

    Additive transforms are neutral at ``0`` and multiplicative ones at ``1``.
    The distinction matters: the transform only produces a term for the outcome
    bucket it owns, and filling the other bucket with ``0`` would silently wipe
    out those episodes' scores under ``exp`` instead of passing them through.

    Args:
        transform: One of :data:`TRANSFORMS`.

    Returns:
        ``0.0`` for the additive family, ``1.0`` for the multiplicative one.

    Raises:
        ValueError: If ``transform`` is unknown.
    """
    _check(transform)
    return 0.0 if transform in ADDITIVE_TRANSFORMS else 1.0


def shapes_failures(transform: str) -> bool:
    """Report which outcome bucket a transform is responsible for.

    Args:
        transform: One of :data:`TRANSFORMS`.

    Returns:
        ``True`` if the transform shapes failed episodes (the additive family),
        ``False`` if it shapes successful ones (the multiplicative family).

    Raises:
        ValueError: If ``transform`` is unknown.
    """
    _check(transform)
    return transform in ADDITIVE_TRANSFORMS


def apply_step_transform(
    values: np.ndarray, transform: str, step_weight: float
) -> np.ndarray:
    """Map a per-step uncertainty trace to its step-level shaping term.

    ``step_weight`` scales how strongly step-level advantages react to
    uncertainty relative to the episode-level term, which is left unscaled.

    Args:
        values: Per-step uncertainty of one trajectory, shape ``(T,)``.
        transform: One of :data:`TRANSFORMS`.
        step_weight: Strength of the step-level shaping.

    Returns:
        The shaping term, same shape as ``values``.

    Raises:
        ValueError: If ``transform`` is unknown.
    """
    _check(transform)
    scaled = values * step_weight
    if transform == "linear":
        return scaled
    if transform == "neg":
        return -scaled
    return np.exp(-scaled)  # "exp"


def apply_episode_transform(value: float, transform: str) -> float:
    """Map an episode-level uncertainty summary to its shaping term.

    Mirrors :func:`apply_step_transform` without the ``step_weight`` scaling:
    the episode-level term is the reference magnitude that ``step_weight`` is
    expressed relative to.

    Args:
        value: Mean uncertainty over the steps of one trajectory.
        transform: One of :data:`TRANSFORMS`.

    Returns:
        The shaping term.

    Raises:
        ValueError: If ``transform`` is unknown.
    """
    _check(transform)
    if transform == "linear":
        return value
    if transform == "neg":
        return -value
    return float(np.exp(-value))  # "exp"


def _check(transform: str) -> None:
    """Validate a transform name.

    Args:
        transform: Candidate transform name.

    Raises:
        ValueError: If ``transform`` is not in :data:`TRANSFORMS`.
    """
    if transform not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform {transform!r}. Choose from {TRANSFORMS}."
        )
