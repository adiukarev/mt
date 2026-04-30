from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TrackingFieldSpec:
	key: str
	source_paths: tuple[str, ...] = ()
	resolver_name: str | None = None
	transform_name: str | None = None


@dataclass(slots=True, frozen=True)
class ManifestTrackingContract:
	param_specs: tuple[TrackingFieldSpec, ...] = ()
	tag_specs: tuple[TrackingFieldSpec, ...] = ()


@dataclass(slots=True, frozen=True)
class MetricsTableSpec:
	source_path: str
	prefix: str
	use_selected_model_name: bool = False


@dataclass(slots=True, frozen=True)
class RowCountSpec:
	source_path: str
	metric_key: str


@dataclass(slots=True, frozen=True)
class MappingMetricSpec:
	source_path: str
	prefix: str
	required_paths: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class FinalMetricsContract:
	metrics_table_specs: tuple[MetricsTableSpec, ...] = ()
	row_count_specs: tuple[RowCountSpec, ...] = ()
	mapping_metric_specs: tuple[MappingMetricSpec, ...] = ()


_COMMON_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("runtime.artifacts_basename", resolver_name="artifact_basename"),
		TrackingFieldSpec("dataset.kind", source_paths=("dataset.kind",)),
		TrackingFieldSpec("dataset.aggregation_level", source_paths=("dataset.aggregation_level",)),
		TrackingFieldSpec("dataset.target_name", source_paths=("dataset.target_name",)),
		TrackingFieldSpec("runtime.seed", source_paths=("runtime.seed",)),
	),
	tag_specs=(
		TrackingFieldSpec("dataset_kind", source_paths=("dataset.kind",)),
		TrackingFieldSpec(
			"aggregation_level",
			source_paths=("dataset.aggregation_level",),
		),
		TrackingFieldSpec("target_name", source_paths=("dataset.target_name",)),
		TrackingFieldSpec("artifacts_basename", resolver_name="artifact_basename"),
	),
)

_AUDIT_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("dataset.series_limit", source_paths=("dataset.series_limit",)),
		TrackingFieldSpec(
			"dataset.series_allowlist_size",
			source_paths=("dataset.series_allowlist",),
			transform_name="len",
		),
	),
)

_EXPERIMENT_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("source.source_type", source_paths=("source.source_type",)),
		TrackingFieldSpec(
			"source.applied_batch_count",
			source_paths=("source.source_config.applied_batch_count",),
		),
		TrackingFieldSpec("backtest.horizon_start", source_paths=("backtest.horizon_start",)),
		TrackingFieldSpec("backtest.horizon_end", source_paths=("backtest.horizon_end",)),
		TrackingFieldSpec(
			"features.combined.enabled",
			resolver_name="experiment_feature_manifest_enabled",
		),
		TrackingFieldSpec(
			"features.combined.feature_set",
			resolver_name="experiment_feature_manifest_feature_set",
		),
		TrackingFieldSpec(
			"features.combined.lag_count",
			resolver_name="experiment_feature_manifest_lag_count",
		),
		TrackingFieldSpec(
			"features.combined.rolling_window_count",
			resolver_name="experiment_feature_manifest_rolling_window_count",
		),
		TrackingFieldSpec("models.total_count", source_paths=("models",), transform_name="len"),
		TrackingFieldSpec("models.enabled_count", resolver_name="experiment_enabled_model_count"),
		TrackingFieldSpec("models.enabled_names", resolver_name="experiment_enabled_model_names"),
		TrackingFieldSpec(
			"probabilistic.default_quantiles",
			resolver_name="default_probabilistic_quantiles",
		),
		TrackingFieldSpec(
			"probabilistic.default_interval_levels",
			resolver_name="default_probabilistic_interval_levels",
		),
	),
	tag_specs=(
		TrackingFieldSpec("source_type", source_paths=("source.source_type",)),
		TrackingFieldSpec("horizon_range", resolver_name="experiment_horizon_range"),
		TrackingFieldSpec(
			"enabled_models",
			resolver_name="experiment_enabled_model_names",
			transform_name="csv",
		),
	),
)

_FORECAST_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("model.source_preference", source_paths=("model.source_preference",)),
		TrackingFieldSpec("forecast.horizon_weeks", source_paths=("forecast.horizon_weeks",)),
		TrackingFieldSpec(
			"model.local.configured",
			source_paths=("model.local.model_dir",),
			transform_name="bool",
		),
		TrackingFieldSpec(
			"model.registry.selection_alias",
			source_paths=("model.registry.selection.alias", "model.registry.selection.registry_alias"),
		),
		TrackingFieldSpec(
			"model.registry.selection_metric",
			source_paths=("model.registry.selection.metric_name",),
		),
	),
	tag_specs=(
		TrackingFieldSpec("source_preference", source_paths=("model.source_preference",)),
		TrackingFieldSpec(
			"registry_selection_alias",
			source_paths=("model.registry.selection.alias", "model.registry.selection.registry_alias"),
		),
		TrackingFieldSpec(
			"registry_selection_metric",
			source_paths=("model.registry.selection.metric_name",),
		),
		TrackingFieldSpec("forecast_horizon_weeks", source_paths=("forecast.horizon_weeks",)),
	),
)

_SYNTHETIC_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("synthetic.scenario_count", source_paths=("scenarios",), transform_name="len"),
		TrackingFieldSpec("synthetic.scenario_names", resolver_name="synthetic_scenario_names"),
	),
	tag_specs=(
		TrackingFieldSpec("scenario_count", source_paths=("scenarios",), transform_name="len"),
	),
)

_MONITORING_MANIFEST_CONTRACT = ManifestTrackingContract(
	param_specs=(
		TrackingFieldSpec("model.source_preference", source_paths=("model.source_preference",)),
		TrackingFieldSpec(
			"model.local.configured",
			source_paths=("model.local.model_dir",),
			transform_name="bool",
		),
		TrackingFieldSpec(
			"model.registry.selection_alias",
			source_paths=("model.registry.selection.alias", "model.registry.selection.registry_alias"),
		),
	),
	tag_specs=(
		TrackingFieldSpec("model_source_preference", source_paths=("model.source_preference",)),
	),
)

_PIPELINE_MANIFEST_CONTRACTS = {
	"audit": _AUDIT_MANIFEST_CONTRACT,
	"experiment": _EXPERIMENT_MANIFEST_CONTRACT,
	"forecast": _FORECAST_MANIFEST_CONTRACT,
	"monitoring": _MONITORING_MANIFEST_CONTRACT,
	"synthetic_generation": _SYNTHETIC_MANIFEST_CONTRACT,
}


FINAL_METRICS_CONTRACT = FinalMetricsContract(
	metrics_table_specs=(
		MetricsTableSpec("overall_metrics", "point.overall", use_selected_model_name=True),
		MetricsTableSpec("by_horizon_metrics", "point.by_horizon", use_selected_model_name=True),
		MetricsTableSpec(
			"probabilistic_overall_metrics",
			"probabilistic.overall",
			use_selected_model_name=True,
		),
		MetricsTableSpec(
			"probabilistic_by_horizon_metrics",
			"probabilistic.by_horizon",
			use_selected_model_name=True,
		),
		MetricsTableSpec("metrics", "forecast.point.by_horizon"),
		MetricsTableSpec("probabilistic_metrics", "forecast.probabilistic.by_horizon"),
	),
	row_count_specs=(
		RowCountSpec("evaluation.metrics_by_segment", "evaluation.segment.rows"),
		RowCountSpec("evaluation.metrics_by_category", "evaluation.category.rows"),
		RowCountSpec(
			"evaluation.probabilistic_metrics_by_segment",
			"evaluation.probabilistic_segment.rows",
		),
		RowCountSpec(
			"evaluation.probabilistic_metrics_by_category",
			"evaluation.probabilistic_category.rows",
		),
		RowCountSpec("evaluation.bootstrap_ci", "evaluation.bootstrap.rows"),
		RowCountSpec("evaluation.error_cases", "evaluation.error_cases.rows"),
		RowCountSpec("evaluation.rolling_vs_holdout", "evaluation.rolling_vs_holdout.rows"),
		RowCountSpec(
			"evaluation.probabilistic_calibration_summary",
			"evaluation.calibration.rows",
		),
		RowCountSpec("audit_artifacts.summary", "audit.summary.rows"),
		RowCountSpec("audit_artifacts.dataset_profile", "audit.dataset_profile.rows"),
		RowCountSpec("audit_artifacts.category_summary", "audit.category_summary.rows"),
		RowCountSpec("audit_artifacts.sku_summary", "audit.category_summary.rows"),
		RowCountSpec(
			"audit_artifacts.series_diagnostic_table",
			"audit.series_diagnostic_table.rows",
		),
		RowCountSpec("metadata", "synthetic.metadata_rows"),
	),
	mapping_metric_specs=(
		MappingMetricSpec(
			"selected_model_metrics",
			"selected_model",
			required_paths=("overall_metrics",),
		),
		MappingMetricSpec(
			"selected_model_metrics",
			"model.selected",
			required_paths=("experiment_manifest",),
		),
		MappingMetricSpec("monitoring_metrics", "monitoring"),
	),
)


def resolve_manifest_tracking_contract(pipeline_type: str) -> ManifestTrackingContract:
	return _merge_manifest_contracts(
		_COMMON_MANIFEST_CONTRACT,
		_PIPELINE_MANIFEST_CONTRACTS.get(pipeline_type, ManifestTrackingContract()),
	)


def _merge_manifest_contracts(
	left: ManifestTrackingContract,
	right: ManifestTrackingContract,
) -> ManifestTrackingContract:
	return ManifestTrackingContract(
		param_specs=(*left.param_specs, *right.param_specs),
		tag_specs=(*left.tag_specs, *right.tag_specs),
	)
