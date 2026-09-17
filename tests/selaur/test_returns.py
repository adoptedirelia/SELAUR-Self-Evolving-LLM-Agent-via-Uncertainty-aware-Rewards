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

"""Tests for the trajectory-level statistics."""

import numpy as np
import pytest

from selaur.returns import compute_trajectory_success_rate


def test_every_episode_counts_once_regardless_of_length():
    """Averaging the flat array would let a long episode outvote a short one."""
    success = np.array([1.0, 1.0, 1.0, 1.0, 0.0])  # 4-step win, 1-step loss
    traj = np.array(["long"] * 4 + ["short"], dtype=object)
    assert compute_trajectory_success_rate(success, traj) == pytest.approx(0.5)


def test_all_failures_give_zero():
    """This is the regime the shaping threshold is meant to gate out."""
    success = np.zeros(6)
    traj = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    assert compute_trajectory_success_rate(success, traj) == 0.0


def test_all_successes_give_one():
    success = np.ones(4)
    traj = np.array(["a", "a", "b", "b"], dtype=object)
    assert compute_trajectory_success_rate(success, traj) == 1.0
