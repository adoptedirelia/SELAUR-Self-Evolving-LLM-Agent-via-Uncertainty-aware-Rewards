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

"""Shared fixtures for the SELAUR unit tests."""

import numpy as np
import pytest
import torch


@pytest.fixture
def confident_logprobs():
    """A step the policy was sure about: high top-1, distant runner-up."""
    top1 = torch.full((8,), -0.01)
    return top1, top1 - 6.0


@pytest.fixture
def unsure_logprobs():
    """A step the policy nearly got wrong: low top-1, near-tied runner-up."""
    top1 = torch.full((8,), -0.9)
    return top1, top1 - 0.01


def make_rollout(episode_lengths, successes, response_length=4, seed=0):
    """Build a synthetic flat rollout batch.

    Args:
        episode_lengths: Number of steps in each episode.
        successes: Per-episode success flag, aligned with ``episode_lengths``.
        response_length: Token count per step.
        seed: Seed for the generated log-probabilities.

    Returns:
        A dict of the arrays the advantage estimator takes.
    """
    generator = torch.Generator().manual_seed(seed)
    index, traj, success, anchors, logprobs = [], [], [], [], []
    for episode, (length, ok) in enumerate(zip(episode_lengths, successes)):
        for step in range(length):
            index.append("task-0")
            traj.append(f"episode-{episode}")
            success.append(float(ok))
            anchors.append(f"state-{step}")
            top1 = -torch.rand(response_length, generator=generator)
            logprobs.append((top1, top1 - torch.rand(response_length, generator=generator)))

    batch_size = len(index)
    logprob_array = np.empty(batch_size, dtype=object)
    for i, pair in enumerate(logprobs):
        logprob_array[i] = pair

    return {
        "index": np.array(index, dtype=object),
        "traj_index": np.array(traj, dtype=object),
        "success": np.array(success),
        "anchor_obs": np.array(anchors, dtype=object),
        "logprobs": logprob_array,
        "token_level_rewards": torch.randn(
            batch_size, response_length, generator=generator
        ),
        "step_rewards": torch.randn(batch_size, generator=generator),
        "response_mask": torch.ones(batch_size, response_length),
    }
