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

"""Aggregation of per-step uncertainty along a trajectory.

A single step's uncertainty is noisy, and the step that *caused* a failure is
rarely the step right before it. :func:`discounted_smooth` therefore credits
each step with a forward-looking average of the uncertainty it and its
successors exhibited, so that hesitation later in an episode is partly
attributed to the decisions that led there.
"""

from __future__ import annotations

import numpy as np

__all__ = ["discounted_smooth"]


def discounted_smooth(values: np.ndarray, decay: float) -> np.ndarray:
    """Normalised backward discounted average of a per-step signal.

    For a trajectory of length ``T`` this returns

    .. math::

        \\tilde{u}_t = \\frac{\\sum_{k \\ge t} \\lambda^{k-t} u_k}
                            {\\sum_{k \\ge t} \\lambda^{k-t}}

    computed in a single reverse pass. Dividing by the weight sum is what
    separates this from a plain discounted return: the output stays on the same
    scale as the input, so a 5-step and a 50-step episode produce comparable
    magnitudes and ``step_weight`` means the same thing for both.

    Args:
        values: Per-step signal of one trajectory, shape ``(T,)``.
        decay: Discount ``lambda`` in ``[0, 1]``. ``0`` keeps each step's own
            value; values near ``1`` spread credit evenly over the remainder of
            the episode.

    Returns:
        The smoothed trace, shape ``(T,)``, dtype ``float64``.
    """
    smoothed = np.zeros_like(values, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for t in range(len(values) - 1, -1, -1):
        numerator = values[t] + decay * numerator
        denominator = 1.0 + decay * denominator
        smoothed[t] = numerator / denominator
    return smoothed
