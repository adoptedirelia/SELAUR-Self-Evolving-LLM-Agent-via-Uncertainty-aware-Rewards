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

"""Anchor-state grouping — the step-level half of GiGPO/SELAUR.

Episode-level grouping is given to us: every rollout of the same task shares a
prompt ``uid``. Step-level grouping has to be *discovered*, and this module is
where that happens.

The idea is the anchor state. Two steps that observed exactly the same
environment state are, for credit-assignment purposes, the same decision point:
whatever happened afterwards is attributable to the action, not to the
situation. Collecting such steps into a group gives a free, on-policy baseline
for that state, which is what :mod:`selaur.normalization` then normalises
against.

Groups are therefore built by exact-matching observations *within* an episode
group, and never across tasks — identical-looking states from different tasks
are not comparable.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Any, Hashable, List

import numpy as np

from selaur.typedefs import GroupIds

__all__ = ["build_step_group", "summarize_group_size", "to_hashable"]


def to_hashable(value: Any) -> Hashable:
    """Convert an observation into something usable as a dict key.

    Observations arrive as whatever the environment emits — strings, nested
    lists, numpy arrays, dicts of those — so they need a canonical hashable
    form before identical states can be matched. Dicts are sorted by key so
    that key ordering never splits a group.

    Args:
        value: The observation, or any part of it, to canonicalise.

    Returns:
        A hashable representation with the same equality semantics.

    Raises:
        TypeError: If ``value`` contains a type with no defined canonical form.
    """
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(value.flatten())
    if isinstance(value, (list, tuple)):
        return tuple(to_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((k, to_hashable(v)) for k, v in value.items()))
    raise TypeError(f"Unsupported type for anchor-state hashing: {type(value)}")


def build_step_group(
    anchor_obs: np.ndarray, index: np.ndarray, summarize: bool = False
) -> GroupIds:
    """Assign a step-level group id to every step of the batch.

    Steps are grouped when they share both the episode group (``index``, i.e.
    the same task/prompt) and an identical anchor observation. Each resulting
    cluster gets a fresh UUID, so group ids are unique across the batch and can
    be used directly as normalisation keys.

    Args:
        anchor_obs: Per-step anchor observations, shape ``(B,)``.
        index: Per-step episode group id (prompt ``uid``), shape ``(B,)``.
        summarize: If ``True``, print the group-size histogram. Useful when
            tuning ``env.rollout.n``: groups of size 1 carry no signal, so a
            distribution concentrated at 1 means the step-level term is mostly
            inert.

    Returns:
        Per-step group ids, shape ``(B,)``, dtype ``object``.

    Raises:
        ValueError: If any step was left without a group id.
    """
    step_group_uids = np.empty(len(anchor_obs), dtype=object)
    group_sizes: List[int] = []

    for episode_group in np.unique(index):
        rows = np.where(index == episode_group)[0]

        clusters = defaultdict(list)
        for row, obs in zip(rows, anchor_obs[rows]):
            clusters[to_hashable(obs)].append(row)

        for cluster_rows in clusters.values():
            group_sizes.append(len(cluster_rows))
            step_group_uids[cluster_rows] = str(uuid.uuid4())

    missing = np.where(step_group_uids == None)[0]  # noqa: E711 - object array
    if len(missing):
        raise ValueError(
            f"Failed to assign step-group UIDs to all observations. "
            f"Missing at indices: {missing}"
        )

    if summarize:
        summarize_group_size(group_sizes)
    print(f"Avg size of step-level group: {np.mean(group_sizes)}")
    return step_group_uids


def summarize_group_size(group_sizes: List[int]) -> None:
    """Print how step-level group sizes are distributed across the batch.

    Args:
        group_sizes: One entry per step-level group, giving its size.
    """
    counts = Counter(group_sizes)
    total = sum(counts.values())

    print("Summary of step-level group sizes:")
    print("Size | Count | Proportion")
    print("-------------------------")
    for size in range(1, max(counts, default=0) + 1):
        count = counts.get(size, 0)
        if not count:
            continue
        print(f"{size:>4} | {count:>5} | {count / total:>9.2%}")
