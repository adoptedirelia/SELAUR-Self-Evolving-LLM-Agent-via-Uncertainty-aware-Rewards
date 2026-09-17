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

"""Tests for the typed view over the Hydra config."""

import pytest

from selaur.config import SELAURConfig, parse_weight

OmegaConf = pytest.importorskip("omegaconf").OmegaConf


@pytest.mark.parametrize(
    "written, expected",
    [("1/3", 1 / 3), ("1.0/3.0", 1 / 3), ("0.5", 0.5), (0.25, 0.25), (1, 1.0)],
)
def test_weights_may_be_written_as_fractions(written, expected):
    """Launch scripts pass `w1=1/3`; Hydra hands that over as a string."""
    assert parse_weight(written) == pytest.approx(expected)


@pytest.mark.parametrize("written", ["", "one third", "1/0", "1/2/3"])
def test_unparseable_weights_are_rejected(written):
    """A malformed weight must fail at startup, not silently become zero."""
    with pytest.raises(ValueError):
        parse_weight(written)


def test_yaml_integers_are_coerced_to_floats():
    """`rho: 1` in YAML must behave exactly like `rho: 1.0`."""
    settings = SELAURConfig.from_hydra(
        OmegaConf.create({"selaur": {"rho": 1, "threshold_success_rate": 0}})
    )
    assert isinstance(settings.rho, float)
    assert isinstance(settings.threshold_success_rate, float)


def test_the_legacy_section_still_overrides():
    """Existing `uqgigpo.*` launch scripts must keep working after the rename."""
    config = OmegaConf.create(
        {
            "selaur": {"transform": "linear", "w3": "1/3"},
            "uqgigpo": {"transform": "neg", "w3": None},
        }
    )
    with pytest.warns(DeprecationWarning, match="uqgigpo"):
        settings = SELAURConfig.from_hydra(config)

    assert settings.transform == "neg"  # taken from the legacy section
    assert settings.w3 == pytest.approx(1 / 3)  # null there, so `selaur` wins


def test_an_all_null_legacy_section_is_silent():
    """The alias exists in the shipped YAML and must not warn by itself."""
    config = OmegaConf.create(
        {"selaur": {"transform": "linear"}, "uqgigpo": {"transform": None}}
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert SELAURConfig.from_hydra(config).transform == "linear"


def test_a_missing_section_falls_back_to_the_defaults():
    """SELAUR must still run on a config that predates the section."""
    assert SELAURConfig.from_hydra(OmegaConf.create({})) == SELAURConfig()


def test_typos_in_the_section_are_reported():
    """A silently ignored key is a sweep that quietly measured nothing."""
    with pytest.raises(ValueError, match="Unknown SELAUR config keys"):
        SELAURConfig.from_hydra(OmegaConf.create({"selaur": {"step_wieght": 0.5}}))


def test_build_estimator_forwards_every_setting():
    """The config is the only place these values are set."""
    settings = SELAURConfig(transform="neg", step_weight=0.5, lambda_uq=0.1, rho=0.2)
    estimator = settings.build_estimator()

    assert estimator.transform == "neg"
    assert estimator.step_weight == 0.5
    assert estimator.lambda_uq == 0.1
    assert estimator.rho == 0.2


def test_the_shipped_yaml_parses():
    """Guards against the config file and the dataclass drifting apart."""
    config = OmegaConf.load("verl/trainer/config/ppo_trainer.yaml")
    settings = SELAURConfig.from_hydra(config)
    assert settings.transform == "linear"
    assert settings.w1 == pytest.approx(1 / 3)
