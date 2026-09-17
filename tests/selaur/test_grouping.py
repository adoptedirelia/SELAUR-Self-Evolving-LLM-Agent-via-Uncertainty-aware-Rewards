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

"""Tests for anchor-state grouping."""

import numpy as np
import pytest

from selaur.grouping import build_step_group, to_hashable


def test_identical_states_within_a_task_share_a_group():
    """Two rollouts that reached the same state are comparable."""
    anchors = np.array(["s0", "s1", "s0"], dtype=object)
    index = np.array(["task", "task", "task"], dtype=object)

    uids = build_step_group(anchors, index)
    assert uids[0] == uids[2]
    assert uids[0] != uids[1]


def test_identical_states_from_different_tasks_never_group():
    """A state means something different in a different task."""
    anchors = np.array(["same", "same"], dtype=object)
    index = np.array(["task-a", "task-b"], dtype=object)

    uids = build_step_group(anchors, index)
    assert uids[0] != uids[1]


def test_every_step_receives_a_group():
    """A missing group id would silently drop a step from normalisation."""
    anchors = np.array([f"s{i % 3}" for i in range(12)], dtype=object)
    index = np.array(["task"] * 12, dtype=object)

    uids = build_step_group(anchors, index)
    assert all(uid is not None for uid in uids)
    assert len(set(uids)) == 3


@pytest.mark.parametrize(
    "left, right",
    [
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}),  # key order must not split a group
        ([1, [2, 3]], [1, [2, 3]]),  # nested sequences
        (np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])),  # arrays
        ("text", "text"),
    ],
)
def test_equal_observations_hash_equal(left, right):
    """Grouping is exact matching, so the hash must follow observation equality."""
    assert to_hashable(left) == to_hashable(right)


def test_different_observations_hash_differently():
    """The converse: distinct states must not collapse into one group."""
    assert to_hashable([1, 2]) != to_hashable([2, 1])
    assert to_hashable({"a": 1}) != to_hashable({"a": 2})


def test_unsupported_observation_types_are_rejected():
    """Silently grouping by object identity would be worse than failing."""
    with pytest.raises(TypeError, match="Unsupported type"):
        to_hashable(object())


def test_numpy_scalars_match_their_python_equivalents():
    """Environments mix numpy and plain scalars in the same observation dict."""
    assert to_hashable(np.int64(3)) == to_hashable(3)
    assert to_hashable(np.float32(1.5)) == to_hashable(1.5)
