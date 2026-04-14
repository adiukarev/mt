from mt.domain.audit import AuditPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.audit.audit import build_data_audit
from mt.infra.audit.plots import save_data_audit_plots
from mt.infra.audit.paths import audit_artifact_relpath, audit_example_series_relpath


class AuditPersistArtifactsStage(BaseStage):
	name = "audit_persist_artifacts"

	def execute(self, ctx: AuditPipelineContext) -> None:
		if ctx.dataset is None or ctx.segments is None:
			raise ValueError()

		# собирает все аналитические таблицы и markdown на основе weekly data и raw metadata
		audit_artifacts = build_data_audit(
			ctx.dataset.weekly,
			ctx.segments,
			ctx.dataset.metadata,
			ctx.dataset.aggregation_level,
			ctx.raw_context,
		)

		self._write_artifact_csv(ctx, "data_audit_summary.csv", audit_artifacts.summary)
		self._write_artifact_csv(ctx, "dataset_profile.csv", audit_artifacts.dataset_profile)
		self._write_artifact_csv(
			ctx,
			"aggregation_comparison.csv",
			audit_artifacts.aggregation_comparison
		)
		self._write_artifact_csv(ctx, "segment_summary.csv", audit_artifacts.segment_summary)
		self._write_artifact_csv(ctx, "category_summary.csv", audit_artifacts.category_summary)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"category_correlation_matrix.csv",
			audit_artifacts.category_correlation_matrix,
		)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"category_growth_correlation_matrix.csv",
			audit_artifacts.category_growth_correlation_matrix,
		)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"category_seasonal_index.csv",
			audit_artifacts.category_seasonal_index,
		)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"sku_summary.csv",
			audit_artifacts.sku_summary
		)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"sku_concentration_summary.csv",
			audit_artifacts.sku_concentration_summary,
		)
		self._write_artifact_csv_if_not_empty(
			ctx,
			"sku_share_stability_summary.csv",
			audit_artifacts.sku_share_stability_summary,
		)
		self._write_artifact_csv(ctx, "feature_availability.csv", audit_artifacts.feature_availability)
		self._write_artifact_csv(
			ctx,
			"feature_block_summary.csv",
			audit_artifacts.feature_block_summary
		)
		self._write_artifact_csv(ctx, "seasonality_summary.csv", audit_artifacts.seasonality_summary)
		self._write_artifact_csv(ctx, "diagnostic_summary.csv", audit_artifacts.diagnostic_summary)
		self._write_artifact_csv(ctx, "stationarity_summary.csv", audit_artifacts.stationarity_summary)
		self._write_artifact_csv(ctx, "data_dictionary.csv", audit_artifacts.data_dictionary)
		self._write_artifact_csv(
			ctx,
			"transformation_summary.csv",
			audit_artifacts.transformation_summary
		)

		for category, snapshot in audit_artifacts.example_feature_snapshots.items():
			write_csv(
				ctx.output_dir / audit_example_series_relpath(category,
				                                              ctx.dataset.aggregation_level) / "example_feature_snapshot.csv",
				snapshot,
			)

		self._write_artifact_csv(
			ctx,
			"raw_sales_sample.csv",
			self._get_frame(ctx.raw_context, "raw_sales_sample")
		)
		self._write_artifact_csv(
			ctx,
			"calendar_sample.csv",
			self._get_frame(ctx.raw_context, "calendar_sample")
		)
		self._write_artifact_csv(
			ctx,
			"sell_prices_sample.csv",
			self._get_frame(ctx.raw_context, "sell_prices_sample")
		)
		self._write_artifact_csv(
			ctx,
			"weekly_panel_sample.csv",
			ctx.dataset.weekly.head(10).reset_index(drop=True)
		)

		# точка входа для чтения полного аудита
		write_markdown(ctx.output_dir / "REPORT.md", audit_artifacts.report_lines)

		save_data_audit_plots(
			audit_artifacts.summary,
			ctx.segments,
			audit_artifacts.seasonality_summary,
			ctx.output_dir,
			ctx.dataset.aggregation_level,
			ctx.dataset.weekly,
			ctx.raw_context,
		)

	def _get_frame(self, raw_context: dict[str, object], key: str):
		frame = raw_context.get(key)
		if frame is None:
			raise ValueError()
		return frame

	def _write_artifact_csv(self, ctx: AuditPipelineContext, filename: str, frame) -> None:
		write_csv(ctx.output_dir / audit_artifact_relpath(filename), frame)

	def _write_artifact_csv_if_not_empty(
		self,
		ctx: AuditPipelineContext,
		filename: str,
		frame
	) -> None:
		if frame is None or getattr(frame, "empty", False):
			return

		self._write_artifact_csv(ctx, filename, frame)
