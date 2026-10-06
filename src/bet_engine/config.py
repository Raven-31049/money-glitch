"""Typed run settings loaded from YAML variant configs in configs/.

WHY a typed config layer: every backtest and report must be able to name the
exact configuration that produced it. pydantic coerces and validates the YAML,
so an out-of-range value or a misspelled key fails loudly at load time, with a
pointer to the offending file, instead of silently poisoning a long-running
analysis. Configs belong in configs/; see docs/PLAN.md and AGENTS.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

# Season codes are football-data style four-character codes ("1819" = 2019-20).
_YEAR_CODE = re.compile(r"^\d{4}$")


class _Base(BaseModel):
    # A misspelled key is a bug, not a suggestion. Forbid extras so typos like
    # "max_stake_fra" fail instead of being silently carried into a report.
    model_config = ConfigDict(extra="forbid")


class ModelSpec(_Base):
    """Which model runs this variant. `type` is a free-form registry key;
    whether it names a real model is the registry's business (models/base.py),
    not the config's."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def _type_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model.type must not be empty")
        return value.strip()


class StakingConfig(_Base):
    """Bet placement rules: edge filter, fractional Kelly, per-bet cap."""

    ev_threshold: float
    kelly_fraction: float
    max_stake_frac: float

    @model_validator(mode="after")
    def _check_ranges(self) -> "StakingConfig":
        if not (0 < self.kelly_fraction <= 1):
            raise ValueError(
                f"staking.kelly_fraction must be in (0, 1]; got {self.kelly_fraction}"
            )
        if self.ev_threshold < 0:
            raise ValueError(
                f"staking.ev_threshold must be >= 0; got {self.ev_threshold}"
            )
        if not (0 < self.max_stake_frac <= 1):
            raise ValueError(
                f"staking.max_stake_frac must be in (0, 1]; got {self.max_stake_frac}"
            )
        return self


class BankrollConfig(_Base):
    """Bankroll rules: starting amount, drawdown limit, and breach behaviour."""

    starting: float
    max_drawdown: float
    stop_on_breach: bool = False

    @model_validator(mode="after")
    def _check_ranges(self) -> "BankrollConfig":
        if self.starting <= 0:
            raise ValueError(f"bankroll.starting must be > 0; got {self.starting}")
        # A drawdown limit of 1.0 means "keep going until ruined", which is the
        # same as no limit and is almost certainly a mistake.
        if not (0 < self.max_drawdown < 1):
            raise ValueError(
                f"bankroll.max_drawdown must be in (0, 1); got {self.max_drawdown}"
            )
        return self


class BacktestConfig(_Base):
    """Walk-forward cadence. Days are CALENDAR days (invariant 1); refitting
    more often than daily is meaningless, hence the >= 1 clamp."""

    min_train_days: int
    refit_every_days: int = Field(default=1)

    @model_validator(mode="after")
    def _check_ranges(self) -> "BacktestConfig":
        if self.min_train_days < 1:
            raise ValueError(
                f"backtest.min_train_days must be >= 1; got {self.min_train_days}"
            )
        if self.refit_every_days < 1:
            raise ValueError(
                f"backtest.refit_every_days must be >= 1; got {self.refit_every_days}"
            )
        return self


class VariantConfig(_Base):
    """One runnable backtest variant: which market, league, model, and how to
    stake and evaluate it. `uncertainty_percentile` is None for the point
    estimate — uncertainty percentiles are never renormalized (invariant 5)."""

    name: str
    market: str
    league: str
    seasons: list[str]
    model: ModelSpec
    staking: StakingConfig
    bankroll: BankrollConfig
    uncertainty_percentile: float | None = None
    backtest: BacktestConfig

    @field_validator("name", "market", "league")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @field_validator("seasons", mode="before")
    @classmethod
    def _coerce_seasons(cls, value: Any) -> list[str]:
        # YAML authors often write seasons as bare numbers ([1819, 1920]); do
        # not require quoting.
        if not isinstance(value, (list, tuple)):
            raise ValueError("seasons must be a list")
        return [str(item) for item in value]

    @field_validator("seasons")
    @classmethod
    def _check_seasons(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("seasons must not be empty")
        for season in value:
            if not _YEAR_CODE.match(season):
                raise ValueError(
                    f"season {season!r} is not a 4-digit code like '1819'"
                )
        return value

    @field_validator("uncertainty_percentile")
    @classmethod
    def _check_percentile(cls, value: float | None) -> float | None:
        if value is not None and not (0 <= value <= 100):
            raise ValueError(
                f"uncertainty_percentile must be in [0, 100]; got {value}"
            )
        return value


def load_variant(path: str | Path) -> VariantConfig:
    """Load and validate one YAML variant config file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: YAML root must be a mapping, got {type(raw).__name__}"
        )
    try:
        return VariantConfig.model_validate(raw)
    except ValidationError as exc:
        details = "\n".join(
            "  " + _format_error(error) for error in exc.errors()
        )
        raise ValueError(f"{path}: invalid variant config\n{details}") from exc


def _format_error(error: dict) -> str:
    field = ".".join(str(part) for part in error["loc"])
    if error["type"] == "value_error" and "ctx" in error:
        message = str(error["ctx"]["error"])
    else:
        message = error["msg"]
    return f"{field}: {message}"


def load_all(directory: str | Path) -> dict[str, VariantConfig]:
    """Load every YAML variant config in a directory, keyed by variant name.

    Every file is loaded even if one fails, so a broken file can never be
    masked by an earlier valid one — and duplicate variant names across files
    are refused outright, because they would make reports ambiguous.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory}: not a directory")
    paths = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    seen: dict[str, Path] = {}
    variants: dict[str, VariantConfig] = {}
    for path in paths:
        config = load_variant(path)
        if config.name in seen:
            raise ValueError(
                f"duplicate variant name {config.name!r}: {seen[config.name]} and {path}"
            )
        seen[config.name] = path
        variants[config.name] = config
    return variants


__all__ = ["VariantConfig", "load_variant", "load_all"]