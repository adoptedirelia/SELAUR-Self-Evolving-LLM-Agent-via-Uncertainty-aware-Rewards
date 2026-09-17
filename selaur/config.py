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

"""Typed view over the ``selaur.*`` Hydra config section.

Hydra hands the trainer an untyped :class:`~omegaconf.DictConfig`, which is
awkward to validate and impossible to construct in a unit test.
:class:`SELAURConfig` is the typed boundary: it is read once at the top of the
advantage computation, validates itself, and is the only thing the rest of the
package sees.

It also absorbs the two rough edges of the CLI surface:

* fractional weights written as strings (``selaur.w1=1/3``), and
* the legacy ``uqgigpo.*`` section, kept working as a deprecated alias.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any, Optional, Union

from selaur.uncertainty.estimator import UncertaintyEstimator
from selaur.uncertainty.signals import SIGNAL_REGISTRY
from selaur.uncertainty.transforms import TRANSFORMS

__all__ = ["SELAURConfig", "parse_weight"]

#: Config section SELAUR reads its settings from.
CONFIG_SECTION = "selaur"

#: Previous name of :data:`CONFIG_SECTION`, still accepted with a warning.
LEGACY_CONFIG_SECTION = "uqgigpo"


def parse_weight(value: Union[str, float, int]) -> float:
    """Parse a weight that may be written as a fraction.

    Shell scripts pass weights as ``w1=1/3`` because typing ``0.3333333`` three
    times is both ugly and slightly wrong. Hydra keeps such a value as a
    string, so it is resolved here rather than at every use site.

    Args:
        value: A number, or a string holding a number or a ``"a/b"`` fraction.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the string is neither a number nor a simple fraction.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass

    numerator, sep, denominator = text.partition("/")
    if sep:
        try:
            return float(numerator) / float(denominator)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Cannot parse weight {value!r}: {exc}") from exc

    raise ValueError(
        f"Cannot parse weight {value!r}; expected a number or an 'a/b' fraction."
    )


@dataclass(frozen=True)
class SELAURConfig:
    """Settings that control uncertainty-aware reward shaping.

    Attributes:
        method: Single uncertainty signal to use when ``use_combined`` is
            ``False``; one of ``entropy``, ``nll``, ``top2``.
        use_combined: Blend all three signals instead of using ``method`` alone.
            This is the SELAUR default.
        w1: Weight of the entropy signal in the blend.
        w2: Weight of the confidence-shortfall signal in the blend.
        w3: Weight of the top-2 margin signal in the blend.
        rho: Mix between the weighted average (``1.0``) and the per-step
            maximum (``0.0``) of the three signals.
        lambda_uq: Discount for spreading uncertainty backwards along a
            trajectory.
        step_weight: Strength of the step-level shaping term relative to the
            episode-level one.
        transform: How uncertainty enters the reward, which also selects the
            episodes it applies to; see :mod:`selaur.uncertainty.transforms`.
        threshold_success_rate: Shaping is skipped entirely while the batch
            success rate is at or below this value. Early in training almost
            everything fails, and shaping that regime only adds noise; the gate
            lets the policy first learn to solve the task, then refines *how
            confidently* it does so.
        shaping_scale: Multiplier applied to the shaping term before it is
            added to the group score, for the additive transforms only.
    """

    method: str = "entropy"
    use_combined: bool = True
    w1: float = 1.0 / 3.0
    w2: float = 1.0 / 3.0
    w3: float = 1.0 / 3.0
    rho: float = 1.0
    lambda_uq: float = 0.95
    step_weight: float = 0.85
    transform: str = "linear"
    threshold_success_rate: float = 0.0
    shaping_scale: float = 10.0

    #: Fields coerced to ``float`` on construction, so that an integer written
    #: in YAML (``rho: 1``) behaves identically to ``rho: 1.0``.
    _FLOAT_FIELDS = ("w1", "w2", "w3", "rho", "lambda_uq", "step_weight",
                     "threshold_success_rate", "shaping_scale")

    def __post_init__(self) -> None:
        """Coerce numeric fields and validate the enum-like ones.

        Raises:
            ValueError: If ``transform`` is unknown, or if ``method`` is
                unknown while ``use_combined`` is ``False``.
        """
        for name in self._FLOAT_FIELDS:
            object.__setattr__(self, name, float(getattr(self, name)))
        object.__setattr__(self, "use_combined", bool(self.use_combined))

        if self.transform not in TRANSFORMS:
            raise ValueError(
                f"Unknown selaur.transform {self.transform!r}. "
                f"Choose from {TRANSFORMS}."
            )
        if not self.use_combined and self.method not in SIGNAL_REGISTRY:
            raise ValueError(
                f"Unknown selaur.method {self.method!r}. "
                f"Choose from {tuple(SIGNAL_REGISTRY)}."
            )

    @classmethod
    def from_hydra(cls, config: Any) -> "SELAURConfig":
        """Build a :class:`SELAURConfig` from the trainer's root config.

        Reads the ``selaur`` section, then lets any explicitly-set key in the
        deprecated ``uqgigpo`` section override it. Keys left at ``null`` in
        the legacy section are ignored, which is what makes the alias
        transparent: ``uqgigpo.w1=1/3`` on the command line still wins, while
        an untouched legacy section changes nothing.

        Args:
            config: The root Hydra config of the PPO trainer.

        Returns:
            The parsed, validated settings.
        """
        settings = _section_to_dict(getattr(config, CONFIG_SECTION, None))

        legacy = _section_to_dict(getattr(config, LEGACY_CONFIG_SECTION, None))
        if legacy:
            warnings.warn(
                f"The '{LEGACY_CONFIG_SECTION}.*' config section is deprecated; "
                f"rename it to '{CONFIG_SECTION}.*'. Overriding: "
                f"{sorted(legacy)}.",
                DeprecationWarning,
                stacklevel=2,
            )
            settings.update(legacy)

        known = {field.name for field in fields(cls)}
        unknown = set(settings) - known
        if unknown:
            raise ValueError(
                f"Unknown SELAUR config keys: {sorted(unknown)}. "
                f"Known keys: {sorted(known)}."
            )

        for weight in ("w1", "w2", "w3"):
            if weight in settings:
                settings[weight] = parse_weight(settings[weight])

        return cls(**settings)

    def build_estimator(self) -> UncertaintyEstimator:
        """Instantiate the estimator these settings describe.

        Returns:
            An :class:`~selaur.uncertainty.estimator.UncertaintyEstimator`
            configured from this config.
        """
        return UncertaintyEstimator(
            method=self.method,
            step_weight=self.step_weight,
            lambda_uq=self.lambda_uq,
            rho=self.rho,
            transform=self.transform,
            use_combined=self.use_combined,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3,
        )


def _section_to_dict(section: Optional[Any]) -> dict:
    """Flatten a config section into a plain dict, dropping unset keys.

    Args:
        section: A ``DictConfig``/mapping, or ``None`` if the section is absent.

    Returns:
        The section's explicitly-set entries. ``None`` values are treated as
        "not set" so that the legacy alias section can be declared with null
        defaults without shadowing the real ones.
    """
    if section is None:
        return {}
    return {key: value for key, value in dict(section).items() if value is not None}
