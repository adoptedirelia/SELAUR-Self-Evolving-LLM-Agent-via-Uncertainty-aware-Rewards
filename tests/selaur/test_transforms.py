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

"""Tests for the reward-shaping transforms."""

import numpy as np
import pytest

from selaur.uncertainty.transforms import (
    ADDITIVE_TRANSFORMS,
    MULTIPLICATIVE_TRANSFORMS,
    TRANSFORMS,
    apply_episode_transform,
    apply_step_transform,
    shapes_failures,
)


def test_the_two_families_partition_the_transforms():
    """Every transform belongs to exactly one family."""
    assert set(ADDITIVE_TRANSFORMS) | set(MULTIPLICATIVE_TRANSFORMS) == set(TRANSFORMS)
    assert not set(ADDITIVE_TRANSFORMS) & set(MULTIPLICATIVE_TRANSFORMS)


def test_additive_transforms_shape_failures_and_multiplicative_shape_successes():
    """The outcome bucket a transform owns follows from its family."""
    assert all(shapes_failures(name) for name in ADDITIVE_TRANSFORMS)
    assert not any(shapes_failures(name) for name in MULTIPLICATIVE_TRANSFORMS)


def test_neg_is_the_sign_flip_of_linear():
    """`neg` exists to test the direction of the effect, nothing more."""
    values = np.array([0.2, 0.5])
    assert apply_step_transform(values, "neg", 0.85) == pytest.approx(
        -apply_step_transform(values, "linear", 0.85)
    )
    assert apply_episode_transform(0.4, "neg") == pytest.approx(
        -apply_episode_transform(0.4, "linear")
    )


def test_step_weight_scales_only_the_step_level_term():
    """The episode-level term is the reference `step_weight` is relative to."""
    values = np.array([0.3, 0.6])
    assert apply_step_transform(values, "linear", 0.5) == pytest.approx(values * 0.5)
    assert apply_episode_transform(0.3, "linear") == pytest.approx(0.3)


def test_exp_is_a_bounded_discount():
    """`exp` shrinks a score instead of shifting it, so it must stay in (0, 1]."""
    values = np.array([0.0, 0.5, 5.0])
    shaped = apply_step_transform(values, "exp", 1.0)
    assert shaped[0] == pytest.approx(1.0)
    assert np.all((shaped > 0.0) & (shaped <= 1.0))
    assert np.all(np.diff(shaped) < 0)


@pytest.mark.parametrize(
    "call",
    [
        lambda: apply_step_transform(np.zeros(2), "bogus", 1.0),
        lambda: apply_episode_transform(0.0, "bogus"),
        lambda: shapes_failures("bogus"),
    ],
)
def test_unknown_transforms_are_rejected(call):
    """A typo in `selaur.transform` must fail loudly, not silently pick a default."""
    with pytest.raises(ValueError, match="Unknown transform"):
        call()
