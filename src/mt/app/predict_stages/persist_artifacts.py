from __future__ import annotations

from time import perf_counter

from mt.domain.predict import PredictPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.synthetic.predict import build_report, write_diagnostic_plots, write_overlay_plot
import yaml


class PredictPersistArtifactsStage(BaseStage):
	name = "predict_persist_artifacts"

	def execute(self, ctx: PredictPipelineContext) -> None:
		if ctx.frame is None or ctx.predictions is None or ctx.metrics is None or ctx.reference_model is None:
			raise ValueError()

		write_csv(ctx.artifacts_paths_map.forecast / "predictions.csv", ctx.predictions)
		write_csv(ctx.artifacts_paths_map.forecast / "metrics_by_horizon.csv", ctx.metrics)
		write_overlay_plot(
			frame=ctx.frame,
			predictions=ctx.predictions,
			overlay_series_id=ctx.manifest.visualization.overlay_series_id,
			plot_history_weeks=ctx.manifest.visualization.plot_history_weeks,
			zoom_history_weeks=ctx.manifest.visualization.zoom_history_weeks,
			annotate_forecast_values=ctx.manifest.visualization.annotate_forecast_values,
			output_path=ctx.artifacts_paths_map.forecast / "forecast_overlay.png"
		)
		write_diagnostic_plots(
			predictions=ctx.predictions,
			metrics=ctx.metrics,
			output_dir=ctx.artifacts_paths_map.forecast,
		)
		write_markdown(
			ctx.artifacts_paths_map.forecast / "README.md",
			build_report(
				ctx.reference_model.model_name,
				ctx.frame,
				ctx.predictions,
				ctx.metrics,
				ctx.manifest.model_source.best_model_dir,
			).splitlines(),
		)
		(ctx.artifacts_paths_map.run / "predict_manifest_snapshot.yaml").write_text(
			yaml.safe_dump(ctx.manifest.as_dict(), sort_keys=False, allow_unicode=True),
			encoding="utf-8",
		)
