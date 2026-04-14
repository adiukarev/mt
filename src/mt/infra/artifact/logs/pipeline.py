import logging

from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.manifest import ExperimentManifest


def log_experiment_start(manifest: ExperimentManifest) -> None:
	feature_manifest = manifest.build_combined_feature_manifest()

	logging.info(
		f"Старт эксперимента | aggregation={manifest.dataset.aggregation_level} | "
		f"feature_set={feature_manifest.feature_set} | "
		f"models={','.join(manifest.enabled_model_names)} | "
		f"output={manifest.runtime.artifacts_dir}"
	)


def log_experiment_end(ctx: ExperimentPipelineContext) -> None:
	wall_time = (
		f"{ctx.pipeline_wall_time_seconds:.3f}s"
		if ctx.pipeline_wall_time_seconds is not None
		else "n/a"
	)

	logging.info(
		f"Запуск завершен | wall_time={wall_time} | "
		f"артефакты сохранены в {ctx.artifacts_paths_map.root}"
	)


def log_experiment_metrics(ctx: ExperimentPipelineContext) -> None:
	"""Вывести итоговую таблицу метрик в консоль в конце пайплайна"""

	if ctx.overall_metrics.empty:
		logging.info("Итоговые метрики отсутствуют")
		return

	display_columns = [
		column
		for column in ("model_name", "WAPE", "sMAPE", "MAE", "RMSE", "Bias", "MedianAE")
		if column in ctx.overall_metrics.columns
	]
	metrics_table = ctx.overall_metrics.loc[:, display_columns].copy()
	float_columns = [column for column in display_columns if column != "model_name"]
	for column in float_columns:
		metrics_table[column] = metrics_table[column].map(lambda value: f"{value:.6f}")

	logging.info(f"Итоговые метрики по моделям:\n{metrics_table.to_string(index=False)}")
