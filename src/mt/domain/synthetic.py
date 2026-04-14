from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ALLOWED_DEMO_MODELS = {"naive", "seasonal_naive"}


@dataclass(slots=True)
class SyntheticCalendarManifest:
	"""Календарная рамка synthetic weekly-датасета."""

	start_date: str = "2022-01-03"
	history_weeks: int = 104
	horizon_weeks: int = 8
	week_anchor: str = "W-MON"

	def __post_init__(self) -> None:
		if self.history_weeks < 16 or self.horizon_weeks < 1 or self.horizon_weeks > 8 or self.week_anchor != "W-MON":
			raise ValueError()


@dataclass(slots=True)
class SyntheticSeriesManifest:
	"""Параметры структуры набора рядов."""

	series_count: int = 24
	categories: list[str] = field(default_factory=lambda: ["foods", "household", "hobbies"])
	base_level_min: float = 80.0
	base_level_max: float = 240.0
	trend_slope_min: float = -0.35
	trend_slope_max: float = 0.60
	yearly_amplitude_min: float = 0.05
	yearly_amplitude_max: float = 0.22
	monthly_amplitude_min: float = 0.02
	monthly_amplitude_max: float = 0.08

	def __post_init__(self) -> None:
		if self.series_count < 1:
			raise ValueError()
		if not self.categories:
			raise ValueError()
		if self.base_level_min <= 0 or self.base_level_max <= self.base_level_min:
			raise ValueError()
		if self.yearly_amplitude_min < 0 or self.yearly_amplitude_max < self.yearly_amplitude_min:
			raise ValueError()
		if self.monthly_amplitude_min < 0 or self.monthly_amplitude_max < self.monthly_amplitude_min:
			raise ValueError()


@dataclass(slots=True)
class SyntheticPromoManifest:
	"""Управляемые known-in-advance promo-события."""

	enabled: bool = True
	probability: float = 0.12
	lift_min: float = 0.08
	lift_max: float = 0.35

	def __post_init__(self) -> None:
		if not 0.0 <= self.probability <= 1.0:
			raise ValueError()
		if self.lift_min < 0 or self.lift_max < self.lift_min:
			raise ValueError()


@dataclass(slots=True)
class SyntheticPriceManifest:
	"""Управляемый known-in-advance price-сценарий."""

	enabled: bool = True
	base_price_min: float = 4.0
	base_price_max: float = 18.0
	discount_probability: float = 0.10
	discount_depth_min: float = 0.03
	discount_depth_max: float = 0.20
	elasticity_min: float = -1.4
	elasticity_max: float = -0.3

	def __post_init__(self) -> None:
		if self.base_price_min <= 0 or self.base_price_max <= self.base_price_min:
			raise ValueError()
		if not 0.0 <= self.discount_probability <= 1.0:
			raise ValueError()
		if self.discount_depth_min < 0 or self.discount_depth_max < self.discount_depth_min:
			raise ValueError()
		if self.elasticity_max > 0:
			raise ValueError()
		if self.elasticity_min > self.elasticity_max:
			raise ValueError()


@dataclass(slots=True)
class SyntheticNoiseManifest:
	"""Шум и редкие искажения наблюдаемого спроса."""

	dispersion_alpha: float = 0.25
	outlier_probability: float = 0.02
	outlier_multiplier_min: float = 1.5
	outlier_multiplier_max: float = 2.4
	stockout_probability: float = 0.01
	stockout_depth_min: float = 0.35
	stockout_depth_max: float = 0.95

	def __post_init__(self) -> None:
		if self.dispersion_alpha <= 0:
			raise ValueError()
		if not 0.0 <= self.outlier_probability <= 1.0:
			raise ValueError()
		if self.outlier_multiplier_min < 1.0 or self.outlier_multiplier_max < self.outlier_multiplier_min:
			raise ValueError()
		if not 0.0 <= self.stockout_probability <= 1.0:
			raise ValueError()
		if (
			self.stockout_depth_min < 0.0
			or self.stockout_depth_max > 1.0
			or self.stockout_depth_max < self.stockout_depth_min
		):
			raise ValueError()


@dataclass(slots=True)
class SyntheticDemoForecastManifest:
	"""Параметры демонстрационного прогноза и overlay-графика."""

	enabled: bool = True
	model_name: str = "seasonal_naive"
	overlay_series_id: str | None = None
	plot_history_weeks: int = 32

	def __post_init__(self) -> None:
		if self.model_name not in ALLOWED_DEMO_MODELS:
			raise ValueError()
		if self.plot_history_weeks < 8:
			raise ValueError()


@dataclass(slots=True)
class SyntheticScenarioManifest:
	"""Модификаторы отдельного сценария synthetic-спроса."""

	name: str = "base"
	description: str = "Balanced synthetic retail scenario"
	level_scale: float = 1.0
	trend_scale: float = 1.0
	yearly_amplitude_scale: float = 1.0
	monthly_amplitude_scale: float = 1.0
	promo_probability_scale: float = 1.0
	promo_lift_scale: float = 1.0
	discount_probability_scale: float = 1.0
	price_elasticity_scale: float = 1.0
	noise_scale: float = 1.0
	outlier_probability_scale: float = 1.0
	stockout_probability_scale: float = 1.0
	zero_inflation_probability: float = 0.0

	def __post_init__(self) -> None:
		if not self.name.strip():
			raise ValueError()

		for field_name in (
				"level_scale",
				"trend_scale",
				"yearly_amplitude_scale",
				"monthly_amplitude_scale",
				"promo_probability_scale",
				"promo_lift_scale",
				"discount_probability_scale",
				"price_elasticity_scale",
				"noise_scale",
				"outlier_probability_scale",
				"stockout_probability_scale",
		):
			value = getattr(self, field_name)
			if value < 0:
				raise ValueError()
		if not 0.0 <= self.zero_inflation_probability <= 1.0:
			raise ValueError()


@dataclass(slots=True)
class SyntheticRuntimeManifest:
	"""Параметры сохранения synthetic-артефактов."""

	output_dir: str = "artifacts/synthetic"
	dataset_name: str = "synthetic"
	seed: int = 42

	def __post_init__(self) -> None:
		if not self.output_dir or not self.dataset_name:
			raise ValueError()


@dataclass(slots=True)
class SyntheticManifest:
	"""Корневой манифест генерации synthetic weekly retail-датасета."""

	calendar: SyntheticCalendarManifest = field(default_factory=SyntheticCalendarManifest)
	series: SyntheticSeriesManifest = field(default_factory=SyntheticSeriesManifest)
	promo: SyntheticPromoManifest = field(default_factory=SyntheticPromoManifest)
	price: SyntheticPriceManifest = field(default_factory=SyntheticPriceManifest)
	noise: SyntheticNoiseManifest = field(default_factory=SyntheticNoiseManifest)
	demo_forecast: SyntheticDemoForecastManifest = field(
		default_factory=SyntheticDemoForecastManifest)
	scenarios: list[SyntheticScenarioManifest] = field(
		default_factory=lambda: [SyntheticScenarioManifest()]
	)
	runtime: SyntheticRuntimeManifest = field(default_factory=SyntheticRuntimeManifest)

	def __post_init__(self) -> None:
		if not self.scenarios:
			raise ValueError()
		names = [scenario.name for scenario in self.scenarios]
		if len(set(names)) != len(names):
			raise ValueError()

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


def load_synthetic_manifest(path: str | Path) -> SyntheticManifest:
	"""Загрузить YAML-манифест synthetic generator."""

	data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
	return SyntheticManifest(
		calendar=SyntheticCalendarManifest(**_section(data, "calendar")),
		series=SyntheticSeriesManifest(**_section(data, "series")),
		promo=SyntheticPromoManifest(**_section(data, "promo")),
		price=SyntheticPriceManifest(**_section(data, "price")),
		noise=SyntheticNoiseManifest(**_section(data, "noise")),
		demo_forecast=SyntheticDemoForecastManifest(**_section(data, "demo_forecast")),
		scenarios=[
			SyntheticScenarioManifest(**item)
			for item in data.get("scenarios", [{}])
		],
		runtime=SyntheticRuntimeManifest(**_section(data, "runtime")),
	)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
	value = data.get(key, {})
	if not isinstance(value, dict):
		raise TypeError(f"Для секции '{key}' ожидался словарь, получено значение типа {type(value)!r}")
	return value
