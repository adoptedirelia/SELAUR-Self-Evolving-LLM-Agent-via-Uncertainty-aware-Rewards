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

"""Policy-uncertainty estimation, in four steps.

``signals``
    Read one step's log-probabilities and score how unsure the policy was.
``aggregation``
    Spread that score backwards along the trajectory, so a step is credited
    with the hesitation it led to.
``transforms``
    Turn the smoothed trace into a reward-shaping term, which also decides
    whether successes or failures are the ones being shaped.
``estimator``
    Run the three over a whole batch and hand the result to the advantage
    estimator.
"""

from selaur.uncertainty.aggregation import discounted_smooth
from selaur.uncertainty.estimator import (
    OutcomeSplit,
    UncertaintyEstimator,
    UncertaintyTrace,
)
from selaur.uncertainty.signals import (
    SIGNAL_REGISTRY,
    combined_uncertainty,
    step_entropy,
    step_nll,
    step_top2_margin,
)
from selaur.uncertainty.transforms import (
    ADDITIVE_TRANSFORMS,
    MULTIPLICATIVE_TRANSFORMS,
    TRANSFORMS,
)

__all__ = [
    "ADDITIVE_TRANSFORMS",
    "MULTIPLICATIVE_TRANSFORMS",
    "OutcomeSplit",
    "SIGNAL_REGISTRY",
    "TRANSFORMS",
    "UncertaintyEstimator",
    "UncertaintyTrace",
    "combined_uncertainty",
    "discounted_smooth",
    "step_entropy",
    "step_nll",
    "step_top2_margin",
]
