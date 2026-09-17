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

"""Tests for the shared group-baseline operation."""

import numpy as np
import pytest
import torch

from selaur.normalization import (
    SingletonPolicy,
    broadcast_to_tokens,
    group_normalize,
)


def _groups(*names):
    return np.array(names, dtype=object)


def test_each_group_is_centred_independently():
    """The group mean is the baseline, so every group must sum to zero."""
    scores = torch.tensor([1.0, 3.0, 10.0, 20.0])
    groups = _groups("a", "a", "b", "b")

    out = group_normalize(
        scores, groups, remove_std=True, singleton_policy=SingletonPolicy.ZERO_BASELINE
    )
    assert out.tolist() == pytest.approx([-1.0, 1.0, -5.0, 5.0])


def test_mean_std_norm_divides_by_the_group_spread():
    """`mean_std_norm` makes advantages comparable across groups."""
    scores = torch.tensor([1.0, 3.0])
    groups = _groups("a", "a")

    out = group_normalize(
        scores, groups, remove_std=False, singleton_policy=SingletonPolicy.ZERO_BASELINE
    )
    expected = (scores - scores.mean()) / (scores.std() + 1e-6)
    assert out.tolist() == pytest.approx(expected.tolist(), abs=1e-5)


def test_a_uniform_shift_within_a_group_is_invisible():
    """This is why shaping is applied before normalisation, not after."""
    scores = torch.tensor([1.0, 3.0, 5.0])
    groups = _groups("a", "a", "a")

    base = group_normalize(
        scores, groups, remove_std=True, singleton_policy=SingletonPolicy.ZERO_BASELINE
    )
    shifted = group_normalize(
        scores + 100.0,
        groups,
        remove_std=True,
        singleton_policy=SingletonPolicy.ZERO_BASELINE,
    )
    assert base.tolist() == pytest.approx(shifted.tolist())


def test_self_baseline_zeroes_a_singleton_group():
    """An anchor state seen once offers no comparison, so it must not push."""
    scores = torch.tensor([7.0])
    out = group_normalize(
        scores,
        _groups("lonely"),
        remove_std=True,
        singleton_policy=SingletonPolicy.SELF_BASELINE,
    )
    assert out.tolist() == pytest.approx([0.0])


def test_zero_baseline_keeps_a_singleton_score():
    """A lone rollout is still the only evidence about its task."""
    scores = torch.tensor([7.0])
    out = group_normalize(
        scores,
        _groups("lonely"),
        remove_std=True,
        singleton_policy=SingletonPolicy.ZERO_BASELINE,
    )
    assert out.tolist() == pytest.approx([7.0])


def test_singleton_std_is_one_so_mean_std_norm_does_not_produce_nan():
    """torch.std of a single element is nan; the policy must guard against it."""
    out = group_normalize(
        torch.tensor([7.0]),
        _groups("lonely"),
        remove_std=False,
        singleton_policy=SingletonPolicy.SELF_BASELINE,
    )
    assert torch.isfinite(out).all()


def test_dedup_keys_weight_trajectories_rather_than_steps():
    """Without dedup a long episode would drag the group mean towards itself."""
    scores = torch.tensor([0.0, 0.0, 0.0, 12.0])
    groups = _groups("a", "a", "a", "a")
    trajectories = _groups("long", "long", "long", "short")

    pooled = group_normalize(
        scores, groups, remove_std=True, singleton_policy=SingletonPolicy.ZERO_BASELINE
    )
    per_trajectory = group_normalize(
        scores,
        groups,
        remove_std=True,
        singleton_policy=SingletonPolicy.ZERO_BASELINE,
        dedup_keys=trajectories,
    )
    assert pooled[0].item() == pytest.approx(-3.0)  # mean over 4 steps
    assert per_trajectory[0].item() == pytest.approx(-6.0)  # mean over 2 episodes


def test_dtype_and_device_survive_normalization():
    """The trainer feeds the result straight back into the batch."""
    scores = torch.tensor([1.0, 2.0], dtype=torch.float64)
    out = group_normalize(
        scores, _groups("a", "a"), remove_std=True,
        singleton_policy=SingletonPolicy.ZERO_BASELINE,
    )
    assert out.dtype == scores.dtype


def test_broadcast_respects_the_response_mask():
    """Padding tokens must not carry advantage into the loss."""
    scores = torch.tensor([2.0, -1.0])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    out = broadcast_to_tokens(scores, mask)
    assert out.tolist() == [[2.0, 2.0, 0.0], [-1.0, 0.0, 0.0]]
