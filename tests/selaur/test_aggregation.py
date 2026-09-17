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

"""Tests for backward discounted smoothing along a trajectory."""

import numpy as np
import pytest

from selaur.uncertainty.aggregation import discounted_smooth


def test_zero_decay_leaves_each_step_alone():
    """With no discount every step keeps exactly its own value."""
    values = np.array([0.1, 0.9, 0.4])
    assert discounted_smooth(values, 0.0) == pytest.approx(values)


def test_unit_decay_is_the_plain_suffix_mean():
    """lambda = 1 weights the whole remainder of the episode equally."""
    values = np.array([1.0, 2.0, 6.0])
    expected = [np.mean(values[t:]) for t in range(len(values))]
    assert discounted_smooth(values, 1.0) == pytest.approx(expected)


def test_the_last_step_is_never_smoothed():
    """Nothing follows the final step, so it has nothing to average with."""
    values = np.array([0.2, 0.7, 0.55])
    for decay in (0.0, 0.5, 1.0):
        assert discounted_smooth(values, decay)[-1] == pytest.approx(values[-1])


def test_output_stays_within_the_input_range():
    """Normalising by the weight sum keeps the scale, unlike a discounted sum."""
    values = np.array([0.3, 0.8, 0.1, 0.5, 0.6])
    smoothed = discounted_smooth(values, 0.95)
    assert smoothed.min() >= values.min() - 1e-12
    assert smoothed.max() <= values.max() + 1e-12


def test_scale_is_independent_of_episode_length():
    """A short and a long episode of the same signal must smooth alike.

    This is the property that lets ``step_weight`` mean the same thing for a
    5-step WebShop episode and a 50-step ALFWorld one.
    """
    for length in (2, 10, 50):
        constant = np.full(length, 0.42)
        assert discounted_smooth(constant, 0.95) == pytest.approx(constant)


def test_single_step_episode_is_a_no_op():
    """Trajectories of length one are common and must not blow up."""
    assert discounted_smooth(np.array([0.7]), 0.95) == pytest.approx([0.7])
