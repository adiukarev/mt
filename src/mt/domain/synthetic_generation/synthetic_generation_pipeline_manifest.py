from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mt.domain.runtime.runtime_manifest import RuntimeManifest
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.helper.dict import get_required_mapping_section


@dataclass(slots=True)
class SyntheticGenerationCalendarManifest:
	start_date: str = "2022-01-03"
	history_weeks: int = 104
	horizon_weeks: int = 8
	week_anchor: str = "W-MON"

	def __post_init__(self) -> None:
		if self.history_weeks < 16:
			raise ValueError()
		if not 1 <= self.horizon_weeks <= 8:
			raise ValueError()
		if self.week_anchor != "W-MON":
			raise ValueError()


@dataclass(slots=True)
class SyntheticGenerationSeriesManifest:
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
class SyntheticGenerationNoiseManifest:
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
		if self.stockout_depth_min < 0.0 or self.stockout_depth_max > 1.0 or self.stockout_depth_max < self.stockout_depth_min:
			raise ValueError()


@dataclass(slots=True)
class SyntheticGenerationScenarioManifest:
	name: str = "base"
	description: str = "Balanced synthetic retail scenario"
	level_scale: float = 1.0
	trend_scale: float = 1.0
	yearly_amplitude_scale: float = 1.0
	monthly_amplitude_scale: float = 1.0
	noise_scale: float = 1.0
	outlier_probability_scale: float = 1.0
	stockout_probability_scale: float = 1.0
	zero_inflation_probability: float = 0.0
	recent_level_scale: float = 1.0
	recent_noise_scale: float = 1.0
	recent_outlier_probability_scale: float = 1.0
	recent_stockout_probability_scale: float = 1.0
	recent_zero_inflation_probability: float = 0.0

	def __post_init__(self) -> None:
		if not self.name.strip():
			raise ValueError()
		for field_name in (
				"level_scale",
				"trend_scale",
				"yearly_amplitude_scale",
				"monthly_amplitude_scale",
				"noise_scale",
				"outlier_probability_scale",
				"stockout_probability_scale",
				"recent_level_scale",
				"recent_noise_scale",
				"recent_outlier_probability_scale",
				"recent_stockout_probability_scale",
		):
			if getattr(self, field_name) < 0:
				raise ValueError()
		if not 0.0 <= self.zero_inflation_probability <= 1.0:
			raise ValueError()
		if not 0.0 <= self.recent_zero_inflation_probability <= 1.0:
			raise ValueError()


@dataclass(slots=True)
class SyntheticGenerationOutputManifest:
	dataset_dir: str = "data/synthetic"
	write_mode: str = "replace"
	scenario_policy: str = "first"
	scenario_cycle: list[str] | None = None

	def __post_init__(self) -> None:
		if not str(self.dataset_dir).strip():
			raise ValueError()
		if self.write_mode not in {"replace", "append"}:
			raise ValueError("write_mode must be replace or append")
		if self.scenario_policy not in {"first", "cycle"}:
			raise ValueError("scenario_policy must be first or cycle")
		if self.scenario_cycle is not None:
			normalized = [str(item).strip() for item in self.scenario_cycle if str(item).strip()]
			if not normalized:
				raise ValueError("scenario_cycle must not be empty")
			self.scenario_cycle = normalized


@dataclass(slots=True)
class SyntheticGenerationPipelineManifest:
	calendar: SyntheticGenerationCalendarManifest = field(
		default_factory=SyntheticGenerationCalendarManifest)
	series: SyntheticGenerationSeriesManifest = field(
		default_factory=SyntheticGenerationSeriesManifest)
	noise: SyntheticGenerationNoiseManifest = field(default_factory=SyntheticGenerationNoiseManifest)
	scenarios: list[SyntheticGenerationScenarioManifest] = field(
		default_factory=lambda: [SyntheticGenerationScenarioManifest()]
	)
	output: SyntheticGenerationOutputManifest = field(
		default_factory=SyntheticGenerationOutputManifest
	)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)

	def __post_init__(self) -> None:
		if not self.scenarios:
			raise ValueError()
		names = [scenario.name for scenario in self.scenarios]
		if len(set(names)) != len(names):
			raise ValueError()

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)

	@staticmethod
	def from_dict(data: dict[str, Any]) -> "SyntheticGenerationPipelineManifest":
		return SyntheticGenerationPipelineManifest(
			calendar=SyntheticGenerationCalendarManifest(**get_required_mapping_section(data, "calendar")),
			series=SyntheticGenerationSeriesManifest(**get_required_mapping_section(data, "series")),
			noise=SyntheticGenerationNoiseManifest(**get_required_mapping_section(data, "noise")),
			scenarios=[
				SyntheticGenerationScenarioManifest(**item)
				for item in data.get("scenarios", [{}])
			],
			output=SyntheticGenerationOutputManifest(**dict(data.get("output", {}))),
			runtime=RuntimeManifest(**get_required_mapping_section(data, "runtime")),
		)

	@staticmethod
	def load(source: str | Path | dict[str, Any]) -> "SyntheticGenerationPipelineManifest":
		if isinstance(source, dict):
			return SyntheticGenerationPipelineManifest.from_dict(source)

		return SyntheticGenerationPipelineManifest.from_dict(read_yaml_mapping(source))
