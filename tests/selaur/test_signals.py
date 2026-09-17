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

"""Tests for the token-level uncertainty signals."""

import numpy as np
import pytest
import torch

from selaur.uncertainty.signals import (
    SIGNAL_REGISTRY,
    combined_uncertainty,
    step_entropy,
    step_nll,
    step_top2_margin,
    unpack_logprobs,
)


@pytest.mark.parametrize("signal", list(SIGNAL_REGISTRY.values()))
def test_signal_is_higher_for_the_unsure_step(
    signal, confident_logprobs, unsure_logprobs
):
    """Every signal must agree on which of two steps was less decisive."""
    assert signal(unsure_logprobs, reduction="mean") > signal(
        confident_logprobs, reduction="mean"
    )


def test_nll_and_top2_are_bounded_to_the_unit_interval(unsure_logprobs):
    """Both bounded signals stay in [0, 1] so the blend weights are meaningful."""
    for signal in (step_nll, step_top2_margin):
        value = signal(unsure_logprobs, reduction="mean").item()
        assert 0.0 <= value <= 1.0


def test_entropy_peaks_at_the_flattest_distribution():
    """-p log p is maximised at p = 1/e, not at p = 1."""
    probs = torch.tensor([0.05, 1 / np.e, 0.9])
    values = step_entropy((torch.log(probs), torch.log(probs) - 1), reduction="none")
    assert values.argmax().item() == 1


def test_top2_margin_falls_as_the_runner_up_drops_away():
    """A widening top-1/top-2 gap means a more decisive step."""
    top1 = torch.full((4,), -0.5)
    margins = [
        step_top2_margin((top1, top1 - gap), reduction="mean").item()
        for gap in (0.0, 1.0, 5.0)
    ]
    assert margins == sorted(margins, reverse=True)


def test_top2_margin_rejects_a_non_positive_temperature(unsure_logprobs):
    """A zero temperature would divide by zero rather than sharpen the sigmoid."""
    with pytest.raises(ValueError, match="temperature"):
        step_top2_margin(unsure_logprobs, temperature=0.0)


@pytest.mark.parametrize(
    "container",
    [
        lambda a, b: (a, b),
        lambda a, b: [a, b],
        lambda a, b: torch.stack([a, b]),
    ],
)
def test_unpack_accepts_every_shape_the_rollout_may_produce(container):
    """Tuple, list and stacked-tensor forms must all round-trip identically."""
    top1, top2 = torch.tensor([-0.1, -0.2]), torch.tensor([-1.1, -1.2])
    unpacked = unpack_logprobs(container(top1, top2))
    assert torch.equal(unpacked[0], top1)
    assert torch.equal(unpacked[1], top2)


def test_unpack_rejects_mismatched_shapes():
    """A shape mismatch means the two branches came from different steps."""
    with pytest.raises(ValueError, match="same shape"):
        unpack_logprobs((torch.zeros(3), torch.zeros(4)))


def test_combined_lies_between_its_parts_when_rho_is_one(unsure_logprobs):
    """A weighted average cannot escape the range of the values it averages."""
    parts = [
        signal(unsure_logprobs, reduction="mean").item()
        for signal in SIGNAL_REGISTRY.values()
    ]
    blended = combined_uncertainty(unsure_logprobs, rho=1.0)
    assert min(parts) <= blended <= max(parts)


def test_rho_zero_reports_the_loudest_signal(unsure_logprobs):
    """rho = 0 is the pessimistic view: any one signal firing is enough."""
    parts = [
        signal(unsure_logprobs, reduction="mean").item()
        for signal in SIGNAL_REGISTRY.values()
    ]
    assert combined_uncertainty(unsure_logprobs, rho=0.0) == pytest.approx(max(parts))


def test_weights_select_a_single_signal(unsure_logprobs):
    """Putting all the weight on one term must reproduce that term exactly."""
    expected = step_nll(unsure_logprobs, reduction="mean").item()
    blended = combined_uncertainty(unsure_logprobs, w1=0.0, w2=1.0, w3=0.0, rho=1.0)
    assert blended == pytest.approx(expected)
