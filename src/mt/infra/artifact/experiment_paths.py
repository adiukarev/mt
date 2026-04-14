from pathlib import PurePosixPath

RUN_FILES = {
	"run_catalog.csv",
	"run_summary.md",
	"comparison_report.md",
}

DATASET_FILES = {
	"dataset_preparation_summary.md",
	"experiment_segmentation.md",
}

FEATURE_FILES = {
	"feature_registry.csv",
	"experiment_feature_registry.md",
	"experiment_supervised_building.md",
	"model_feature_usage.csv",
}

VALIDATION_FILES = {
	"backtest_windows.csv",
	"backtest_window_generation.md",
	"rolling_vs_holdout_diagnostic.csv",
}

COMPARISON_FILES = {
	"overall_model_comparison.csv",
	"metrics_by_horizon.csv",
	"metrics_by_segment.csv",
	"metrics_by_category.csv",
	"bootstrap_ci_model_differences.csv",
	"selected_error_cases.csv",
	"leader_forecast.csv",
}


def experiment_artifact_relpath(filename: str) -> PurePosixPath:
	if filename in RUN_FILES:
		return PurePosixPath("run") / filename
	if filename in DATASET_FILES:
		return PurePosixPath("dataset") / filename
	if filename in FEATURE_FILES:
		return PurePosixPath("features") / filename
	if filename in VALIDATION_FILES:
		return PurePosixPath("validation") / filename
	if filename in COMPARISON_FILES:
		return PurePosixPath("comparison") / filename
	return PurePosixPath(filename)


def experiment_artifact_link(filename: str) -> str:
	return experiment_artifact_relpath(filename).as_posix()
