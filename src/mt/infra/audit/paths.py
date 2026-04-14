from pathlib import PurePosixPath
import re

ROOT_FILES = {
	"REPORT.md",
}

DATASET_FILES = {
	"data_audit_summary.csv",
	"data_dictionary.csv",
	"dataset_profile.csv",
}

AGGREGATION_FILES = {
	"aggregation_comparison.csv",
	"aggregation_daily_vs_weekly_total_sales.png",
	"aggregation_daily_weekly_monthly_total_sales.png",
	"weekly_total_sales.png",
}

FEATURE_FILES = {
	"feature_availability.csv",
	"feature_block_summary.csv",
	"transformation_summary.csv",
}

DIAGNOSTIC_FILES = {
	"diagnostic_summary.csv",
	"segment_summary.csv",
	"seasonality_summary.csv",
	"stationarity_summary.csv",
	"history_length_distribution.png",
	"zero_share_distribution.png",
	"missing_share_distribution.png",
	"outlier_share_distribution.png",
	"coefficient_of_variation_distribution.png",
	"trend_strength_distribution.png",
	"seasonal_autocorrelation_profile.png",
	"target_distribution_raw.png",
	"target_distribution_log1p.png",
}

OPTIONAL_DIAGNOSTIC_FILES = {
	"segment_distribution.png",
	"zero_share_vs_history.png",
}

SAMPLE_FILES = {
	"raw_sales_sample.csv",
	"calendar_sample.csv",
	"sell_prices_sample.csv",
	"weekly_panel_sample.csv",
}

VALIDATION_FILES = {
	"rolling_backtest_schematic.png",
}

CATEGORY_SCOPE_PATHS = {
	"category_summary.csv": PurePosixPath("dataset") / "category" / "category_summary.csv",
	"category_correlation_matrix.csv": PurePosixPath("dataset") / "category" / "category_correlation_matrix.csv",
	"category_growth_correlation_matrix.csv": PurePosixPath("dataset") / "category" / "category_growth_correlation_matrix.csv",
	"category_seasonal_index.csv": PurePosixPath("dataset") / "category" / "category_seasonal_index.csv",
	"total_sales_by_category.png": PurePosixPath("aggregation") / "category" / "total_sales_by_category.png",
	"normalized_weekly_sales_by_category.png": PurePosixPath("aggregation") / "category" / "normalized_weekly_sales_by_category.png",
	"series_count_by_category.png": PurePosixPath("aggregation") / "category" / "series_count_by_category.png",
	"sku_count_by_category.png": PurePosixPath("aggregation") / "category" / "sku_count_by_category.png",
	"seasonal_heatmap_by_category_week.png": PurePosixPath("diagnostics") / "category" / "seasonal_heatmap_by_category_week.png",
	"seasonal_profile_by_week_of_year.png": PurePosixPath("diagnostics") / "category" / "seasonal_profile_by_week_of_year.png",
}

SKU_SCOPE_PATHS = {
	"sku_summary.csv": PurePosixPath("dataset") / "sku" / "sku_summary.csv",
	"sku_concentration_summary.csv": PurePosixPath("dataset") / "sku" / "sku_concentration_summary.csv",
	"sku_share_stability_summary.csv": PurePosixPath("dataset") / "sku" / "sku_share_stability_summary.csv",
	"sku_top20_total_sales.png": PurePosixPath("aggregation") / "sku" / "sku_top20_total_sales.png",
	"sku_cumulative_sales_share.png": PurePosixPath("aggregation") / "sku" / "sku_cumulative_sales_share.png",
	"sku_normalized_sample.png": PurePosixPath("aggregation") / "sku" / "sku_normalized_sample.png",
}


def audit_artifact_relpath(filename: str) -> PurePosixPath:
	if filename in CATEGORY_SCOPE_PATHS:
		return CATEGORY_SCOPE_PATHS[filename]
	if filename in SKU_SCOPE_PATHS:
		return SKU_SCOPE_PATHS[filename]
	if filename in ROOT_FILES:
		return PurePosixPath(filename)
	if filename.startswith("example_"):
		return PurePosixPath("example_series") / filename
	if filename in DATASET_FILES:
		return PurePosixPath("dataset") / filename
	if filename in AGGREGATION_FILES:
		return PurePosixPath("aggregation") / filename
	if filename in FEATURE_FILES:
		return PurePosixPath("features") / filename
	if filename in DIAGNOSTIC_FILES:
		return PurePosixPath("diagnostics") / filename
	if filename in OPTIONAL_DIAGNOSTIC_FILES:
		return PurePosixPath("diagnostics") / filename
	if filename in SAMPLE_FILES:
		return PurePosixPath("samples") / filename
	if filename in VALIDATION_FILES:
		return PurePosixPath("validation") / filename
	return PurePosixPath(filename)


def audit_artifact_link(filename: str) -> str:
	return audit_artifact_relpath(filename).as_posix()


def slugify_audit_name(value: str) -> str:
	slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
	return slug.strip("_") or "unknown"


def audit_example_series_relpath(category: str, aggregation_level: str) -> PurePosixPath:
	scope = "category" if aggregation_level == "category" else "sku"
	return PurePosixPath("example_series") / scope / slugify_audit_name(category)


def audit_example_series_link(category: str, filename: str, aggregation_level: str) -> str:
	return (audit_example_series_relpath(category, aggregation_level) / filename).as_posix()
