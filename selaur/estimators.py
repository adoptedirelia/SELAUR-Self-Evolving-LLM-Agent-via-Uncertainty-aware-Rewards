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

"""Canonical ``algorithm.adv_estimator`` names for the group-step algorithms.

These live outside the trainer because two independent places need them: the
trainer, to dispatch to the right advantage estimator, and the rollout
collector, to decide whether it must record the top-2 log-probabilities that
the uncertainty signals read. Importing the trainer from the collector would be
circular, so the names are single-sourced here instead.
"""

from __future__ import annotations

from typing import Tuple

__all__ = [
    "GIGPO",
    "GROUP_STEP_ESTIMATOR_NAMES",
    "SELAUR",
    "SELAUR_ESTIMATOR_NAMES",
    "SELAUR_LEGACY",
]

#: The SELAUR advantage estimator.
SELAUR = "selaur"

#: Pre-rename spelling of :data:`SELAUR`, still accepted so that existing
#: launch scripts and checkpoint paths keep working.
SELAUR_LEGACY = "uqgigpo"

#: The GiGPO baseline.
GIGPO = "gigpo"

#: Every name that resolves to SELAUR.
SELAUR_ESTIMATOR_NAMES: Tuple[str, ...] = (SELAUR, SELAUR_LEGACY)

#: Every estimator built on episode + anchor-state groups. These are the ones
#: that need per-step rollout log-probabilities and that report uncertainty
#: diagnostics alongside their advantages.
GROUP_STEP_ESTIMATOR_NAMES: Tuple[str, ...] = (GIGPO,) + SELAUR_ESTIMATOR_NAMES
