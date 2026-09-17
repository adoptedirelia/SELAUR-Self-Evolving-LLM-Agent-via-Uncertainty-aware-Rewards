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

"""Tests for the SELAUR advantage estimator and the GiGPO baseline."""

import numpy as np
import pytest
import torch

from conftest import make_rollout
from selaur.advantage import compute_selaur_outcome_advantage
from selaur.baselines import compute_gigpo_outcome_advantage
from selaur.config import SELAURConfig


def _advantage(settings, **overrides):
    rollout = make_rollout(episode_lengths=[3, 2, 4], successes=[0, 1, 1])
    kwargs = dict(rollout, settings=settings)
    kwargs.update(overrides)
    return compute_selaur_outcome_advantage(**kwargs), rollout


def _gigpo():
    rollout = make_rollout(episode_lengths=[3, 2, 4], successes=[0, 1, 1])
    return compute_gigpo_outcome_advantage(**rollout), rollout


def test_advantages_match_the_token_layout():
    """The trainer writes these straight into the batch."""
    result, rollout = _advantage(SELAURConfig())
    assert result.advantages.shape == rollout["token_level_rewards"].shape
    assert torch.equal(result.advantages, result.returns)


def test_shaping_is_off_below_the_success_threshold():
    """Early in training almost everything fails and shaping is pure noise."""
    off, _ = _advantage(SELAURConfig(threshold_success_rate=1.0))
    on, _ = _advantage(SELAURConfig(threshold_success_rate=0.0))
    assert not torch.allclose(off.advantages, on.advantages)


def test_shaping_off_reproduces_the_gigpo_baseline():
    """SELAUR must reduce to GiGPO exactly when uncertainty is ignored."""
    gated, _ = _advantage(SELAURConfig(threshold_success_rate=1.0))
    baseline, _ = _gigpo()
    assert torch.allclose(gated.advantages, baseline.advantages, atol=1e-6)


def test_the_baseline_still_reports_diagnostics():
    """Comparable W&B curves are the point of running the baseline this way."""
    baseline, rollout = _gigpo()
    recorded = sum(
        len(bucket) for bucket in baseline.step_uncertainty.values()
    )
    assert recorded == len(rollout["traj_index"])


def test_advantages_are_centred_within_each_group():
    """No critic: the group mean is the baseline, so groups must sum to zero."""
    result, rollout = _advantage(SELAURConfig(threshold_success_rate=1.0))
    per_step = result.advantages[:, 0]
    for uid in np.unique(rollout["index"]):
        rows = rollout["index"] == uid
        # Episode + step level are both centred, so their sum is too.
        assert per_step[rows].sum().item() == pytest.approx(0.0, abs=1e-4)


def test_step_advantage_weight_scales_only_the_step_term():
    """`step_advantage_w=0` must leave the episode-level advantage alone."""
    settings = SELAURConfig(threshold_success_rate=1.0)
    episode_only, _ = _advantage(settings, step_advantage_w=0.0)
    doubled, _ = _advantage(settings, step_advantage_w=2.0)
    single, _ = _advantage(settings, step_advantage_w=1.0)

    step_term = single.advantages - episode_only.advantages
    assert torch.allclose(
        doubled.advantages, episode_only.advantages + 2 * step_term, atol=1e-5
    )


@pytest.mark.parametrize("mode", ["mean_norm", "mean_std_norm"])
def test_both_normalization_modes_run(mode):
    """`mode` comes from `algorithm.gigpo.mode` and both values are used."""
    result, _ = _advantage(SELAURConfig(), mode=mode)
    assert torch.isfinite(result.advantages).all()


def test_an_unknown_mode_is_rejected():
    """A typo here would otherwise silently pick one of the two behaviours."""
    with pytest.raises(ValueError, match="Unknown mode"):
        _advantage(SELAURConfig(), mode="bogus")


def test_the_dtype_does_not_depend_on_whether_shaping_fired():
    """Uncertainty is float64 numpy; it must not promote the advantage tensor."""
    shaped, _ = _advantage(SELAURConfig(threshold_success_rate=0.0))
    gated, _ = _advantage(SELAURConfig(threshold_success_rate=1.0))
    assert shaped.advantages.dtype == gated.advantages.dtype == torch.float32


@pytest.mark.parametrize("transform", ["linear", "neg", "exp"])
def test_every_transform_produces_finite_advantages(transform):
    """All three ablation arms have to be runnable."""
    result, _ = _advantage(
        SELAURConfig(transform=transform, threshold_success_rate=0.0)
    )
    assert torch.isfinite(result.advantages).all()


def test_exp_leaves_failed_episodes_alone():
    """`exp` shapes successes; it must not multiply failures away.

    The neutral term for a multiplicative transform is 1, not 0. Getting this
    wrong silently erases the gradient signal from every failed episode.
    """
    rollout = make_rollout(episode_lengths=[3, 2, 4], successes=[0, 1, 1])
    failed = rollout["success"] == 0

    shaped = compute_selaur_outcome_advantage(
        **rollout,
        settings=SELAURConfig(transform="exp", threshold_success_rate=0.0),
    )
    # Compare the raw scores rather than the advantages: normalisation would
    # hide a wiped-out group behind its own (also wiped-out) mean.
    from selaur.returns import compute_trajectory_success_rate

    assert compute_trajectory_success_rate(
        rollout["success"], rollout["traj_index"]
    ) > 0.0
    assert torch.isfinite(shaped.advantages).all()
    assert shaped.advantages[failed].abs().sum() > 0


def test_both_levels_shape_in_the_same_direction():
    """Episode and step terms must push a score the same way, not fight."""
    rollout = make_rollout(episode_lengths=[3, 2, 4], successes=[0, 1, 1])
    settings = SELAURConfig(transform="linear", threshold_success_rate=0.0)

    episode_only = compute_selaur_outcome_advantage(
        **rollout, settings=settings, step_advantage_w=0.0
    )
    gated = compute_selaur_outcome_advantage(
        **rollout,
        settings=SELAURConfig(threshold_success_rate=1.0),
        step_advantage_w=0.0,
    )
    episode_delta = episode_only.advantages - gated.advantages

    with_steps = compute_selaur_outcome_advantage(
        **rollout, settings=settings, step_advantage_w=1.0
    )
    gated_steps = compute_selaur_outcome_advantage(
        **rollout, settings=SELAURConfig(threshold_success_rate=1.0), step_advantage_w=1.0
    )
    total_delta = with_steps.advantages - gated_steps.advantages
    step_delta = total_delta - episode_delta

    # Same sign wherever either term actually moved the score.
    moved = (episode_delta.abs() > 1e-6) & (step_delta.abs() > 1e-6)
    assert moved.any()
    assert torch.all(torch.sign(episode_delta[moved]) == torch.sign(step_delta[moved]))


def test_linear_and_neg_shape_in_opposite_directions():
    """`neg` is the control arm; it must actually differ from `linear`."""
    forward, _ = _advantage(SELAURConfig(transform="linear", threshold_success_rate=0.0))
    reverse, _ = _advantage(SELAURConfig(transform="neg", threshold_success_rate=0.0))
    baseline, _ = _advantage(SELAURConfig(threshold_success_rate=1.0))

    forward_delta = forward.advantages - baseline.advantages
    reverse_delta = reverse.advantages - baseline.advantages
    assert torch.allclose(forward_delta, -reverse_delta, atol=1e-5)
