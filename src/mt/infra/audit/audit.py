from dataclasses import dataclass

import pandas as pd

from mt.infra.audit.markdown import build_report_lines
from mt.infra.audit.profiles import (
	build_aggregation_comparison,
	build_category_summary,
	build_data_dictionary,
	build_dataset_profile,
	build_diagnostic_summary,
	build_example_feature_snapshots,
	build_feature_availability,
	build_feature_block_summary,
	build_category_correlation_matrix,
	build_category_growth_correlation_matrix,
	build_category_seasonal_index,
	build_seasonality_summary,
	build_segment_summary,
	build_stationarity_summary,
	build_summary,
	build_sku_concentration_summary,
	build_sku_share_stability_summary,
	build_sku_summary,
	build_transformation_summary,
)
from mt.domain.audit import AuditArtifacts


def build_data_audit(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
	metadata: dict[str, object],
	aggregation_level: str,
	raw_context: dict[str, object],
) -> AuditArtifacts:
	"""Создать артефакты аудита"""

	summary = build_summary(weekly, segments)
	dataset_profile = build_dataset_profile(
		weekly,
		summary,
		segments,
		metadata,
		aggregation_level,
		raw_context,
	)
	aggregation_comparison = build_aggregation_comparison(weekly, raw_context)
	segment_summary = build_segment_summary(summary)
	category_summary = build_category_summary(weekly, summary, raw_context)
	sku_summary = build_sku_summary(summary, aggregation_level)
	category_correlation_matrix = build_category_correlation_matrix(weekly)
	category_growth_correlation_matrix = build_category_growth_correlation_matrix(weekly)
	category_seasonal_index = build_category_seasonal_index(weekly)
	sku_concentration_summary = build_sku_concentration_summary(weekly, aggregation_level)
	sku_share_stability_summary = build_sku_share_stability_summary(weekly, aggregation_level)
	feature_availability = build_feature_availability(metadata, raw_context)
	feature_block_summary = build_feature_block_summary(aggregation_level)
	seasonality_summary = build_seasonality_summary(weekly)
	diagnostic_summary = build_diagnostic_summary(summary, seasonality_summary)
	stationarity_summary = build_stationarity_summary(weekly)
	data_dictionary = build_data_dictionary(weekly)
	transformation_summary = build_transformation_summary(weekly)
	example_feature_snapshots = build_example_feature_snapshots(weekly, segments)
	example_feature_snapshot = _pick_example_feature_snapshot(example_feature_snapshots)

	report_lines = build_report_lines(
		aggregation_level,
		weekly,
		summary,
		seasonality_summary,
		data_dictionary,
		transformation_summary,
		example_feature_snapshot,
		metadata,
		raw_context,
	)

	return AuditArtifacts(
		summary=summary,
		dataset_profile=dataset_profile,
		aggregation_comparison=aggregation_comparison,
		segment_summary=segment_summary,
		category_summary=category_summary,
		sku_summary=sku_summary,
		category_correlation_matrix=category_correlation_matrix,
		category_growth_correlation_matrix=category_growth_correlation_matrix,
		category_seasonal_index=category_seasonal_index,
		sku_concentration_summary=sku_concentration_summary,
		sku_share_stability_summary=sku_share_stability_summary,
		feature_availability=feature_availability,
		feature_block_summary=feature_block_summary,
		seasonality_summary=seasonality_summary,
		diagnostic_summary=diagnostic_summary,
		stationarity_summary=stationarity_summary,
		data_dictionary=data_dictionary,
		transformation_summary=transformation_summary,
		example_feature_snapshots=example_feature_snapshots,
		report_lines=report_lines,
	)


def _pick_example_feature_snapshot(
	example_feature_snapshots: dict[str, pd.DataFrame],
) -> pd.DataFrame:
	if not example_feature_snapshots:
		return pd.DataFrame()

	first_category = sorted(example_feature_snapshots)[0]

	return example_feature_snapshots[first_category]
