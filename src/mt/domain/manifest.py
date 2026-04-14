import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ALLOWED_AGGREGATION_LEVELS = {"category", "sku"}
ALLOWED_BACKTEST_MODES = {"direct", "recursive", "multi_output"}
MANDATORY_BASELINE = "seasonal_naive"
ALLOWED_MODEL_NAMES = {
	"naive",
	"seasonal_naive",
	"ets",
	"ridge",
	"lightgbm",
	"catboost",
	"mlp",
	"nbeats",
}
DL_MODEL_NAMES = {"mlp", "nbeats"}
FEATURE_SET_PATTERN = re.compile(r"^F[0-6](?:_[A-Za-z0-9]+)?$")


@dataclass(slots=True)
class DatasetManifest:
	"""Настройки загрузки и агрегации датасета"""

	path: str = "data/m5-forecasting-accuracy"
	aggregation_level: str = "category"
	target_name: str = "sales_units"
	week_anchor: str = "W-MON"
	sample_rows: int | None = None
	series_limit: int | None = None
	include_promo: bool = False
	allow_price_features: bool = False

	def __post_init__(self) -> None:
		if self.aggregation_level not in ALLOWED_AGGREGATION_LEVELS \
			or self.week_anchor != "W-MON" \
			or self.sample_rows is not None and self.sample_rows <= 0 \
			or self.series_limit is not None and self.series_limit <= 0:
			raise ValueError()


@dataclass(slots=True)
class BacktestManifest:
	"""Настройки rolling window backtesting"""

	horizon_min: int = 1
	horizon_max: int = 8
	min_train_weeks: int = 52
	step_weeks: int = 1
	max_windows: int | None = 8
	mode: str = "direct"

	def __post_init__(self) -> None:
		if self.horizon_min < 1 or self.horizon_max > 8 or self.horizon_min > self.horizon_max:
			raise ValueError()
		if self.min_train_weeks < 8 or self.step_weeks < 1:
			raise ValueError()
		if self.max_windows is not None and self.max_windows < 1:
			raise ValueError()
		if self.mode not in ALLOWED_BACKTEST_MODES:
			raise ValueError()


@dataclass(slots=True)
class FeatureManifest:
	"""Настройки признаков конкретной модели"""

	enabled: bool = False
	feature_set: str = "F4"
	lags: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 8, 12, 13, 26, 52])
	rolling_windows: list[int] = field(default_factory=lambda: [4, 8, 13, 26])
	use_calendar: bool = True
	use_category_encodings: bool = True
	use_price: bool = False
	use_promo: bool = False
	use_external: bool = False

	def __post_init__(self) -> None:
		if not FEATURE_SET_PATTERN.fullmatch(self.feature_set):
			raise ValueError()
		if any(lag <= 0 for lag in self.lags):
			raise ValueError()
		if any(window <= 1 for window in self.rolling_windows):
			raise ValueError()

		self.lags = sorted(set(self.lags))
		self.rolling_windows = sorted(set(self.rolling_windows))


@dataclass(slots=True)
class RuntimeManifest:
	"""Настройки выполнения эксперимента"""

	artifacts_dir: str = "artifacts/default_run"
	seed: int = 42
	bootstrap_samples: int = 300

	def __post_init__(self) -> None:
		if not self.artifacts_dir or self.bootstrap_samples < 1:
			raise ValueError()


@dataclass(slots=True)
class AuditRuntimeManifest:
	"""Настройки выполнения отдельного аудита данных"""

	output_dir: str = ""
	seed: int = 42

	def __post_init__(self) -> None:
		if self.seed < 0:
			raise ValueError()


@dataclass(slots=True)
class DLManifest:
	"""Настройки глубокой модели"""

	history_length: int = 52
	hidden_size: int = 128
	n_blocks: int = 3
	n_layers: int = 4
	epochs: int = 80
	batch_size: int = 32
	learning_rate: float = 1e-3
	weight_decay: float = 1e-5
	device: str = "cpu"

	def __post_init__(self) -> None:
		if self.history_length < 8 or self.hidden_size < 8 or self.n_blocks < 1 or self.n_layers < 1 or self.epochs < 1 or self.batch_size < 1:
			raise ValueError()


DL_CONFIG_FIELD_NAMES = {item.name for item in fields(DLManifest)}


@dataclass(slots=True)
class ModelManifest:
	"""Полная конфигурация одной модели внутри эксперимента"""

	name: str
	enabled: bool = True
	features: FeatureManifest = field(default_factory=FeatureManifest)
	config: dict[str, Any] = field(default_factory=dict)

	def __post_init__(self) -> None:
		self.name = self.name.strip().lower()
		if self.name not in ALLOWED_MODEL_NAMES:
			raise ValueError()
		if not isinstance(self.config, dict):
			raise TypeError()
		self.config = dict(self.config)

		if self.name in DL_MODEL_NAMES:
			missing_fields = sorted(DL_CONFIG_FIELD_NAMES.difference(self.config))
			if missing_fields:
				raise ValueError()

	@property
	def adapter_params(self) -> dict[str, Any]:
		return {
			key: value
			for key, value in self.config.items()
			if key not in DL_CONFIG_FIELD_NAMES
		}

	@property
	def dl_manifest(self) -> DLManifest | None:
		if self.name not in DL_MODEL_NAMES:
			return None
		return DLManifest(
			**{
				field_name: self.config[field_name]
				for field_name in DL_CONFIG_FIELD_NAMES
			}
		)


@dataclass(slots=True)
class ExperimentManifest:
	"""Корневой манифест эксперимента"""

	dataset: DatasetManifest = field(default_factory=DatasetManifest)
	backtest: BacktestManifest = field(default_factory=BacktestManifest)
	runtime: RuntimeManifest = field(default_factory=RuntimeManifest)
	models: list[ModelManifest] = field(
		default_factory=lambda: [ModelManifest(name="seasonal_naive")])

	def __post_init__(self) -> None:
		if not self.models:
			raise ValueError()

		normalized_names = [model.name for model in self.models]
		if len(set(normalized_names)) != len(normalized_names):
			raise ValueError()

		enabled_models = self.enabled_models
		if not enabled_models:
			raise ValueError()
		if MANDATORY_BASELINE not in {model.name for model in enabled_models}:
			raise ValueError()

		for model in enabled_models:
			if model.features.use_promo and not self.dataset.include_promo:
				raise ValueError()
			if model.features.use_price and not self.dataset.allow_price_features:
				raise ValueError()

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)

	@property
	def enabled_models(self) -> list[ModelManifest]:
		return [model for model in self.models if model.enabled]

	@property
	def enabled_model_names(self) -> list[str]:
		return [model.name for model in self.enabled_models]

	def get_model(self, model_name: str) -> ModelManifest:
		normalized_name = model_name.strip().lower()
		for model in self.models:
			if model.name == normalized_name:
				return model
		raise KeyError()

	def get_enabled_model(self, model_name: str) -> ModelManifest:
		model = self.get_model(model_name)
		if not model.enabled:
			raise ValueError()
		return model

	def build_combined_feature_manifest(self) -> FeatureManifest:
		enabled_feature_manifests = [
			model.features
			for model in self.enabled_models
			if model.features.enabled
		]
		if not enabled_feature_manifests:
			return FeatureManifest(enabled=False, feature_set="F0")

		return FeatureManifest(
			enabled=True,
			feature_set=_max_feature_set(enabled_feature_manifests),
			lags=sorted({lag for manifest in enabled_feature_manifests for lag in manifest.lags}),
			rolling_windows=sorted(
				{window for manifest in enabled_feature_manifests for window in manifest.rolling_windows}
			),
			use_calendar=any(manifest.use_calendar for manifest in enabled_feature_manifests),
			use_category_encodings=any(
				manifest.use_category_encodings for manifest in enabled_feature_manifests
			),
			use_price=any(manifest.use_price for manifest in enabled_feature_manifests),
			use_promo=any(manifest.use_promo for manifest in enabled_feature_manifests),
			use_external=any(manifest.use_external for manifest in enabled_feature_manifests),
		)


@dataclass(slots=True)
class AuditManifest:
	"""Корневой манифест аудита данных"""

	dataset: DatasetManifest = field(default_factory=DatasetManifest)
	runtime: AuditRuntimeManifest = field(default_factory=AuditRuntimeManifest)

	def __post_init__(self) -> None:
		if not self.runtime.output_dir:
			self.runtime.output_dir = (
				"artifacts/audit_category"
				if self.dataset.aggregation_level == "category"
				else "artifacts/audit_sku"
			)

	def as_dict(self) -> dict[str, Any]:
		return asdict(self)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
	data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
	return ExperimentManifest(
		dataset=DatasetManifest(**_section(data, "dataset")),
		backtest=BacktestManifest(**_section(data, "backtest")),
		runtime=RuntimeManifest(**_section(data, "runtime")),
		models=[
			_load_model_manifest(item) for item in data.get("models")
		],
	)


def load_audit_manifest(path: str | Path) -> AuditManifest:
	data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
	return AuditManifest(
		dataset=DatasetManifest(**_section(data, "dataset")),
		runtime=AuditRuntimeManifest(**_section(data, "runtime")),
	)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
	value = data.get(key, {})
	if not isinstance(value, dict):
		raise TypeError(f"Для секции '{key}' ожидался словарь, получено значение типа {type(value)!r}")
	return value


def _load_model_manifest(item: object) -> ModelManifest:
	data = dict(item)

	features_data = data.pop("features", {})
	config_data = _normalize_model_config(
		config=data.pop("config", None),
		params=data.pop("params", None),
		dl=data.pop("dl", None),
	)

	return ModelManifest(
		**data,
		features=FeatureManifest(**features_data),
		config=config_data,
	)


def _max_feature_set(manifests: list[FeatureManifest]) -> str:
	return f"F{max(_feature_set_rank(manifest.feature_set) for manifest in manifests)}"


def _feature_set_rank(feature_set: str) -> int:
	prefix = feature_set.split("_", maxsplit=1)[0]
	if not prefix.startswith("F"):
		return 4
	try:
		return int(prefix[1:])
	except ValueError:
		return 4


def _normalize_model_config(
	config: object | None,
	params: object | None,
	dl: object | None,
) -> dict[str, Any]:
	if config is not None:
		if params is not None or dl is not None:
			raise ValueError()
		if not isinstance(config, dict):
			raise TypeError()

		return dict(config)

	merged: dict[str, Any] = {}
	if params is not None:
		if not isinstance(params, dict):
			raise TypeError()
		merged.update(params)
	if dl is not None:
		if not isinstance(dl, dict):
			raise TypeError()

		merged.update(dl)
	return merged
