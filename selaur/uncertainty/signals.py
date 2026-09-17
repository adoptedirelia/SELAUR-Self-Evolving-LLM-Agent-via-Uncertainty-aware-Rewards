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

"""Token-level uncertainty signals for a single generated action.

Every signal in this module consumes the ``(top1, top2)`` log-probability pair
recorded by the rollout collector for one agent step and reduces it to a scalar
"how unsure was the policy here" score. Higher is always *more* uncertain.

Three complementary views of the same distribution are provided. These are the
three metrics of the paper (arXiv:2602.21158), named here after what they
compute:

``step_entropy`` -- the paper's **entropy**-based metric
    Spread of the top-1 branch: ``-p * log p``. Large when probability mass is
    smeared out over many tokens.
``step_nll`` -- the paper's **least-confidence** metric
    Confidence shortfall: ``1 - p``. Large when the chosen token was unlikely.
    Selected with ``selaur.method=nll``.
``step_top2_margin`` -- the paper's **margin**-based metric
    Decision margin: a squashed ``top1 - top2`` gap. Large when the two best
    candidates were nearly tied, i.e. the policy almost picked something else.
    Selected with ``selaur.method=top2``.

:func:`combined_uncertainty` mixes the three into the paper's *combined
token-level uncertainty estimate*, the single scalar SELAUR actually uses for
reward shaping.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from torch import Tensor

from selaur.typedefs import LogProbPair, LogProbsInput

__all__ = [
    "SIGNAL_REGISTRY",
    "combined_uncertainty",
    "step_entropy",
    "step_nll",
    "step_top2_margin",
    "unpack_logprobs",
]

#: Valid values for the ``reduction`` keyword shared by all signals.
_REDUCTIONS = ("batch", "mean", "none")


def unpack_logprobs(logprobs: LogProbsInput) -> LogProbPair:
    """Normalise any accepted log-prob container into a ``(top1, top2)`` pair.

    Rollout data survives several round-trips (vLLM -> DataProto -> numpy object
    array) and comes back in whichever shape the last hop produced. Accepted
    forms are a 2-element sequence, a ``(2, ...)`` tensor, or a ``(..., 2, L)``
    tensor.

    Args:
        logprobs: Log-probabilities of one generated response.

    Returns:
        The top-1 and top-2 log-probability tensors, promoted to a common dtype.

    Raises:
        TypeError: If ``logprobs`` is not a recognised container.
        ValueError: If the two-way dimension cannot be identified, or if the two
            resulting tensors disagree in shape.
    """
    if isinstance(logprobs, np.ndarray):
        top1, top2 = logprobs
    elif isinstance(logprobs, (tuple, list)):
        if len(logprobs) != 2:
            raise ValueError(f"Expected a 2-element sequence, got {len(logprobs)}.")
        top1, top2 = logprobs
    elif isinstance(logprobs, torch.Tensor):
        if logprobs.ndim < 2:
            raise ValueError("Stacked logprobs tensor must have at least 2 dims.")
        if logprobs.shape[-2] == 2:  # (..., 2, L)
            top1, top2 = logprobs[..., 0, :], logprobs[..., 1, :]
        elif logprobs.shape[0] == 2:  # (2, ...)
            top1, top2 = logprobs[0], logprobs[1]
        else:
            raise ValueError(
                f"Could not infer the 2-way dimension from shape {tuple(logprobs.shape)}."
            )
    else:
        raise TypeError(
            f"logprobs must be a (Tensor, Tensor) pair or a Tensor, got {type(logprobs)}."
        )

    if top1.shape != top2.shape:
        raise ValueError(
            f"top-1 and top-2 logprobs must have the same shape, "
            f"got {tuple(top1.shape)} and {tuple(top2.shape)}."
        )

    common_dtype = torch.promote_types(top1.dtype, top2.dtype)
    return top1.to(dtype=common_dtype), top2.to(dtype=common_dtype)


def _reduce(values: Tensor, reduction: str) -> Tensor:
    """Apply the shared reduction convention to a per-token signal.

    Args:
        values: Per-token signal values.
        reduction: ``"batch"`` averages over the last dim (one scalar per row),
            ``"mean"`` averages everything into a single scalar, and ``"none"``
            returns the per-token values untouched.

    Returns:
        The reduced tensor.

    Raises:
        ValueError: If ``reduction`` is not one of :data:`_REDUCTIONS`.
    """
    if reduction == "batch":
        return values.mean(dim=-1)
    if reduction == "mean":
        return values.mean()
    if reduction == "none":
        return values
    raise ValueError(f"Invalid reduction {reduction!r}. Choose from {_REDUCTIONS}.")


def step_entropy(logprobs: LogProbsInput, *, reduction: str = "batch") -> Tensor:
    """Point-wise entropy contribution of the sampled tokens.

    Computes ``-p * log p`` from the top-1 log-probabilities, which is the
    per-token term of the Shannon entropy restricted to the realised branch.

    Args:
        logprobs: Log-probabilities of one generated response.
        reduction: See :func:`_reduce`.

    Returns:
        The reduced entropy signal. Higher means flatter, less decisive output.
    """
    top1, _ = unpack_logprobs(logprobs)
    probs = torch.exp(top1)
    return _reduce(-(probs * top1), reduction)


def step_nll(logprobs: LogProbsInput, *, reduction: str = "batch") -> Tensor:
    """Least-confidence uncertainty of the sampled tokens.

    Computes ``1 - exp(logp)``, a bounded stand-in for the negative
    log-likelihood that stays in ``[0, 1)`` and therefore mixes cleanly with the
    other two signals in :func:`combined_uncertainty`.

    Args:
        logprobs: Log-probabilities of one generated response.
        reduction: See :func:`_reduce`.

    Returns:
        The reduced confidence-shortfall signal. Higher means less confident.
    """
    top1, _ = unpack_logprobs(logprobs)
    return _reduce(1.0 - torch.exp(top1), reduction)


def step_top2_margin(
    logprobs: LogProbsInput,
    *,
    temperature: float = 1.0,
    bias: float = 1.0,
    reduction: str = "batch",
) -> Tensor:
    """Margin-based uncertainty: the squashed top-1/top-2 decision margin.

    Computes ``sigmoid((bias - (logp1 - logp2)) / temperature)``. A wide margin
    between the best and runner-up token drives the value towards 0 (decisive);
    a near-tie drives it towards 1 (the policy could easily have acted
    differently).

    Args:
        logprobs: Log-probabilities of one generated response.
        temperature: Sigmoid sharpness; must be positive. Smaller values make
            the signal closer to a hard threshold on the margin.
        bias: Margin offset, i.e. the gap at which the signal crosses ``0.5``.
        reduction: See :func:`_reduce`.

    Returns:
        The reduced margin signal. Higher means a closer call.

    Raises:
        ValueError: If ``temperature`` is not positive.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")
    top1, top2 = unpack_logprobs(logprobs)
    margin = top1 - top2
    return _reduce(torch.special.expit((bias - margin) / temperature), reduction)


#: Name -> signal, for selecting a single signal via ``selaur.method``.
SIGNAL_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "entropy": step_entropy,
    "nll": step_nll,
    "top2": step_top2_margin,
}


def combined_uncertainty(
    logprobs: LogProbsInput,
    w1: float = 1.0 / 3.0,
    w2: float = 1.0 / 3.0,
    w3: float = 1.0 / 3.0,
    rho: float = 1.0,
) -> float:
    """The paper's combined token-level uncertainty estimate.

    The blend interpolates between a weighted average and a pessimistic
    worst-case view::

        u = rho * (w1*entropy + w2*nll + w3*top2) + (1 - rho) * max(entropy, nll, top2)

    ``rho = 1`` keeps the plain weighted average; ``rho = 0`` reports whichever
    single signal fired hardest, which is the conservative choice when any one
    view of uncertainty should be enough to flag a step.

    Args:
        logprobs: Log-probabilities of one generated response.
        w1: Weight of the entropy signal.
        w2: Weight of the confidence-shortfall signal.
        w3: Weight of the top-2 margin signal.
        rho: Mixing coefficient between the weighted average and the maximum.

    Returns:
        The combined uncertainty of this step as a plain Python float.
    """
    entropy = step_entropy(logprobs, reduction="mean").item()
    nll = step_nll(logprobs, reduction="mean").item()
    top2 = step_top2_margin(logprobs, reduction="mean").item()

    weighted = w1 * entropy + w2 * nll + w3 * top2
    return rho * weighted + (1.0 - rho) * max(entropy, nll, top2)
