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

"""Tests for the batch-level uncertainty estimator."""

import numpy as np
import pytest

from conftest import make_rollout
from selaur.uncertainty import UncertaintyEstimator
from selaur.uncertainty.transforms import neutral_term


def _trace(transform, **kwargs):
    """Run the estimator over a 2-failure / 1-success batch."""
    rollout = make_rollout(episode_lengths=[3, 2, 4], successes=[0, 1, 0])
    estimator = UncertaintyEstimator(transform=transform, **kwargs)
    trace = estimator.compute(
        rollout["logprobs"], rollout["success"], rollout["traj_index"]
    )
    return trace, rollout


@pytest.mark.parametrize("transform", ["linear", "neg"])
def test_additive_transforms_shape_only_failed_episodes(transform):
    """Successes must be left untouched so the shaping stays one-sided."""
    trace, rollout = _trace(transform)
    succeeded = rollout["success"] == 1

    assert np.all(trace.step[succeeded] == 0.0)
    assert np.all(trace.episode[succeeded] == 0.0)
    assert np.any(trace.step[~succeeded] != 0.0)


def test_exp_transform_shapes_only_successful_episodes():
    """`exp` is the mirror image: it discounts wins, not losses."""
    trace, rollout = _trace("exp")
    succeeded = rollout["success"] == 1

    assert np.any(trace.step[succeeded] != 1.0)
    assert np.all(trace.step[~succeeded] == 1.0)
    assert np.all(trace.episode[~succeeded] == 1.0)


@pytest.mark.parametrize("transform", ["linear", "neg", "exp"])
def test_skipped_episodes_receive_the_neutral_term(transform):
    """The bucket a transform skips must pass through the shaping unchanged.

    For the additive family the neutral term is 0, but for `exp` it is 1 --
    filling it with 0 would multiply those episodes' scores away instead of
    leaving them alone.
    """
    trace, rollout = _trace(transform)
    from selaur.uncertainty.transforms import shapes_failures

    skipped = (rollout["success"] == 1) if shapes_failures(transform) else (
        rollout["success"] == 0
    )
    assert np.all(trace.step[skipped] == neutral_term(transform))
    assert np.all(trace.episode[skipped] == neutral_term(transform))


def test_episode_term_is_constant_within_a_trajectory():
    """The episode-level term summarises the whole episode, so it cannot vary."""
    trace, rollout = _trace("linear")
    for uid in np.unique(rollout["traj_index"]):
        rows = rollout["traj_index"] == uid
        assert len(np.unique(trace.episode[rows])) == 1


def test_diagnostics_cover_every_step_regardless_of_shaping():
    """Logging must stay complete even for the bucket the transform skips."""
    trace, rollout = _trace("exp")
    total = len(trace.step_diagnostics.success) + len(trace.step_diagnostics.failure)
    assert total == len(rollout["traj_index"])


def test_diagnostics_do_not_depend_on_the_configured_transform():
    """Fixed-transform diagnostics are what makes ablation curves comparable."""
    linear, _ = _trace("linear")
    flipped, _ = _trace("neg")
    assert linear.step_diagnostics.as_arrays()["failure"] == pytest.approx(
        flipped.step_diagnostics.as_arrays()["failure"]
    )


def test_an_unknown_method_is_rejected_only_when_it_would_be_used():
    """`method` is dead config while `use_combined` is on, so it is not validated."""
    UncertaintyEstimator(method="bogus", use_combined=True)
    with pytest.raises(ValueError, match="Unknown method"):
        UncertaintyEstimator(method="bogus", use_combined=False)


def test_an_unknown_transform_is_always_rejected():
    """Unlike `method`, the transform is always live."""
    with pytest.raises(ValueError, match="Unknown transform"):
        UncertaintyEstimator(transform="bogus")


def test_confident_episodes_get_a_smaller_term_than_unsure_ones(
    confident_logprobs, unsure_logprobs
):
    """The whole point: hesitation must register as a larger shaping term."""
    success = np.zeros(2)
    traj = np.array(["a", "b"], dtype=object)
    logprobs = np.empty(2, dtype=object)
    logprobs[0] = confident_logprobs
    logprobs[1] = unsure_logprobs

    trace = UncertaintyEstimator(transform="linear").compute(logprobs, success, traj)
    assert trace.step[0] < trace.step[1]
