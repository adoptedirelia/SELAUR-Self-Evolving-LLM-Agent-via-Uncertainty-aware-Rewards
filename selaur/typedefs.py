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

"""Shared type aliases for the SELAUR algorithm package.

These aliases exist to make the shapes flowing through the algorithm explicit.
The whole package works on a *flat batch of steps*: every row is one
agent-environment interaction step, and rows belonging to the same episode are
tied together by ``traj_index`` while rows belonging to the same task/prompt are
tied together by ``index``.

Naming convention used throughout the package::

    B  - number of steps in the flat batch
    L  - response length, i.e. tokens generated for one agent step
    R  - padded response length of the token-level reward/mask tensors
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "LogProbPair",
    "LogProbsInput",
    "LogProbsBatch",
    "GroupIds",
    "SuccessFlags",
]

#: Per-token log-probability of the sampled token and of the runner-up token,
#: for a single generated response. Both entries have identical shape ``(L,)``
#: and are recorded by the rollout collector as ``rollout_log_probs`` and
#: ``rollout_log_probs_2nd``, truncated to the un-padded answer length.
LogProbPair = Tuple[Tensor, Tensor]

#: Anything the uncertainty signals accept for a single step. Besides the plain
#: ``(top1, top2)`` pair we also tolerate a stacked tensor, because DataProto
#: round-trips sometimes collapse the pair into an array of shape ``(2, L)``.
LogProbsInput = Union[
    np.ndarray,
    LogProbPair,
    List[Tensor],
    Tensor,
]

#: One :data:`LogProbsInput` per step of the flat batch, indexable by ``int``
#: and by integer arrays (``np.ndarray`` of dtype ``object`` in practice).
LogProbsBatch = Sequence[LogProbsInput]

#: Per-step group identifiers, shape ``(B,)``, dtype ``object`` (UUID strings).
GroupIds = np.ndarray

#: Per-step episode success flags, shape ``(B,)``. Every step of an episode
#: carries the *episode-level* outcome, so the flag is constant within a
#: trajectory.
SuccessFlags = np.ndarray
