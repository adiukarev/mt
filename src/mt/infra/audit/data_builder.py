import pandas as pd

from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.infra.audit.dataset_builder import (
	build_category_summary,
	build_dataset_profile,
	build_sku_summary,
)
from mt.infra.audit.diagnostic_builder import (
	build_seasonality_summary,
	build_stationarity_summary,
	build_summary,
)
from mt.infra.audit.feature_builder import build_data_dictionary
from mt.infra.audit.report_builder import (
	build_report_lines,
	build_series_diagnostic_table,
)
from mt.infra.audit.series_builder import build_series_feature_snapshots


def build_data_audit(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
	metadata: dict[str, object],
	aggregation_level: str,
	raw_context: dict[str, object],
) -> AuditArtifactData:
	summary = build_summary(weekly, segments)
	dataset_profile = build_dataset_profile(weekly, summary, segments, metadata, aggregation_level, raw_context)
	category_summary = build_category_summary(weekly, summary, raw_context)
	sku_summary = build_sku_summary(summary, aggregation_level)
	seasonality_summary = build_seasonality_summary(weekly)
	stationarity_summary = build_stationarity_summary(weekly)
	series_diagnostic_table = build_series_diagnostic_table(
		summary,
		seasonality_summary,
		stationarity_summary,
	)
	data_dictionary = build_data_dictionary(weekly)
	series_feature_snapshots = build_series_feature_snapshots(weekly, segments)
	report_lines = build_report_lines(series_diagnostic_table)

	return AuditArtifactData(
		summary=summary,
		dataset_profile=dataset_profile,
		category_summary=category_summary,
		sku_summary=sku_summary,
		series_diagnostic_table=series_diagnostic_table,
		data_dictionary=data_dictionary,
		series_feature_snapshots=series_feature_snapshots,
		report_lines=report_lines,
	)
