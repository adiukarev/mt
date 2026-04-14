import shutil
from pathlib import Path

from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.artifact.writer import write_markdown
from mt.infra.feature.registry import build_feature_registry
from mt.infra.model.best_model_training import fit_and_save_best_model_from_context, \
	select_best_model_from_metrics


class BestModelFitStage(BaseStage):
	"""Переобучить победителя эксперимента на всей доступной истории"""

	name = "experiment_best_model_fit"

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		if (
			ctx.dataset is None
			or ctx.supervised is None
			or ctx.feature_columns is None
			or ctx.overall_metrics is None
			or ctx.overall_metrics.empty
		):
			raise ValueError()

		selected_model_name = select_best_model_from_metrics(
			ctx.overall_metrics,
			ctx.manifest.enabled_model_names,
		)
		selected_model_manifest = ctx.manifest.get_enabled_model(selected_model_name)
		selected_model_metrics = ctx.overall_metrics.loc[
			ctx.overall_metrics["model_name"].astype(str) == selected_model_name
			].iloc[0].to_dict()

		best_model_dir = ctx.artifacts_paths_map.models / "best_model"

		feature_registry = build_feature_registry(
			selected_model_manifest.features,
			aggregation_level=ctx.dataset.aggregation_level,
		)

		ctx.selected_model_name = selected_model_name
		ctx.selected_model_metrics = selected_model_metrics
		ctx.best_model_artifact_path = fit_and_save_best_model_from_context(
			manifest=ctx.manifest,
			output_dir=best_model_dir,
			dataset_bundle=ctx.dataset,
			feature_registry=feature_registry,
			supervised=ctx.supervised,
			feature_columns=ctx.feature_columns,
			model_name=selected_model_name,
		)
		ctx.best_model_report_path = best_model_dir / "best_model_fit_report.md"

		# artifacts
		self._persist_artifacts(ctx, ctx.best_model_report_path)

		self._ensure_report_alias(best_model_dir)
		selected_model_final_dir = ctx.artifacts_paths_map.models / selected_model_name / "final"
		shutil.copytree(best_model_dir, selected_model_final_dir, dirs_exist_ok=True)
		self._ensure_report_alias(selected_model_final_dir)

	def _persist_artifacts(self, ctx: ExperimentPipelineContext, report_path: Path) -> None:
		metric_lines = [
			f"- {metric_name}: {metric_value}"
			for metric_name, metric_value in ctx.selected_model_metrics.items()
			if metric_name != "model_name"
		]
		write_markdown(
			report_path,
			[
				"# Best Model Fit",
				"",
				f"- Название модели: {ctx.selected_model_name}",
				f"- Уровень агрегации: {ctx.dataset.aggregation_level if ctx.dataset is not None else 'n/a'}",
				f"- Горизонты: {ctx.manifest.backtest.horizon_min}..{ctx.manifest.backtest.horizon_max}",
				"- Источник выбора: rolling backtesting текущего experiment pipeline",
				f"- Файл сравнения: {ctx.artifacts_paths_map.root / 'comparison' / 'overall_model_comparison.csv'}",
				f"- Артефакты лучшей модели модели: {ctx.best_model_artifact_path}",
				"",
				"## Метрики выбранной модели",
				"",
				*(metric_lines or ["- Метрики недоступны"]),
			],
		)

	def _ensure_report_alias(self, model_dir: Path) -> None:
		legacy_report = model_dir / "best_model_fit_report.md"
		final_report = model_dir / "final_fit_report.md"
		if legacy_report.exists():
			shutil.copyfile(legacy_report, final_report)
