from pathlib import Path

import pandas as pd

from mt.domain.experiment import ExperimentPipelineContext
from mt.domain.manifest import ModelManifest
from mt.domain.model import ModelResult
from mt.domain.stage import BaseStage
from mt.infra.artifact.experiment_paths import experiment_artifact_relpath
from mt.infra.artifact.logs.model import log_model_runner_start, log_model_runner_end
from mt.infra.artifact.serialization import dump_yaml
from mt.infra.artifact.writer import write_csv, write_markdown
from mt.infra.metric.common import aggregate_metrics
from mt.infra.model.feature_resolution import resolve_model_feature_columns
from mt.infra.model.runner import run_model


class ModelExecutionStage(BaseStage):
	"""Этап запуска моделей и сохранения модельных артефактов"""

	name = "experiment_model_execution"

	def execute(self, ctx: ExperimentPipelineContext) -> None:
		"""Прогнать все модели из манифеста и собрать общие прогнозы"""

		if ctx.dataset is None or ctx.supervised is None or ctx.windows is None:
			raise ValueError()

		all_predictions: list[pd.DataFrame] = []
		model_catalog_rows: list[dict[str, object]] = []
		model_feature_usage_rows: list[dict[str, object]] = []

		for model_manifest in ctx.model_manifests:
			model_name = model_manifest.name

			log_model_runner_start(model_name, ctx.windows)

			model_feature_columns = self._resolve_model_feature_columns(ctx, model_manifest)

			# supervised общий, окна тоже общие
			# отличие между моделями тут только в model_feature_columns и dl
			result = run_model(
				model_name,
				ctx.supervised,
				model_feature_columns,
				ctx.windows,
				ctx.manifest.runtime.seed,
				model_params=model_manifest.adapter_params,
				dl_manifest=model_manifest.dl_manifest,
			)

			model_dir = ctx.artifacts_paths_map.models / model_name
			model_dir.mkdir(parents=True, exist_ok=True)

			log_model_runner_end(result, model_dir)

			model_catalog_rows.append(
				{
					"record_type": "model",
					"name": result.info.model_name,
					"status": "completed",
					"wall_time_seconds": result.wall_time_seconds,
				}
			)
			model_feature_usage_rows.append(
				{
					"model_name": result.info.model_name,
					"model_family": result.info.model_family,
					"feature_count": len(result.used_feature_columns),
					"used_feature_columns": " | ".join(result.used_feature_columns),
				}
			)

			self.persist_artifacts(ctx, model_dir, result, model_manifest)

			if not result.predictions.empty:
				all_predictions.append(result.predictions)

		ctx.predictions = pd.concat(
			all_predictions,
			ignore_index=True) if all_predictions else pd.DataFrame()
		ctx.model_catalog_rows = model_catalog_rows
		ctx.model_feature_usage_rows = model_feature_usage_rows
		ctx.run_catalog_rows.extend(model_catalog_rows)
		write_csv(
			ctx.artifacts_paths_map.root / experiment_artifact_relpath("model_feature_usage.csv"),
			pd.DataFrame(model_feature_usage_rows)
		)

	def _resolve_model_feature_columns(
		self,
		ctx: ExperimentPipelineContext,
		model_manifest: ModelManifest,
	) -> list[str]:
		return resolve_model_feature_columns(
			model_manifest,
			ctx.feature_columns,
			ctx.dataset.aggregation_level
		)

	def persist_artifacts(
		self,
		ctx: ExperimentPipelineContext,
		model_dir: Path,
		result: ModelResult,
		model_manifest: ModelManifest,
	):
		"""Сохранить артефакты модели и вернуть признак сопоставимых прогнозов"""

		# сырые прогнозы оставляем как есть, потом из них можно все пересобрать
		result.predictions.to_csv(model_dir / "raw_predictions.csv", index=False)

		# тут сразу и overall и по горизонтам
		metrics_overall, metrics_by_horizon = aggregate_metrics(result.predictions)
		metrics_overall.to_csv(model_dir / "metrics_overall.csv", index=False)
		metrics_by_horizon.to_csv(model_dir / "metrics_by_horizon.csv", index=False)

		dump_yaml(
			model_dir / "run_manifest.yaml",
			{
				"aggregation_level": ctx.dataset.aggregation_level,
				"model_name": result.info.model_name,
				"model_family": result.info.model_family,
				"features": _serialize_feature_manifest(model_manifest),
				"config": model_manifest.config,
				"used_feature_columns": result.used_feature_columns,
				"seed": ctx.manifest.runtime.seed,
			},
		)

		write_markdown(
			model_dir / "run_summary.md",
			[
				f"# {result.info.model_name}",
				"",
				f"- семейство модели: {result.info.model_family}",
				f"- признаки: {'enabled' if model_manifest.features.enabled else 'disabled'}",
				f"- feature_set: {model_manifest.features.feature_set if model_manifest.features.enabled else 'none'}",
				f"- число использованных признаков: {len(result.used_feature_columns)}",
				(
					f"- время обучения, сек.: {result.train_time_seconds:.3f}"
					if result.train_time_seconds is not None
					else "- время обучения, сек.: n/a"
				),
				(
					f"- время инференса, сек.: {result.inference_time_seconds:.3f}"
					if result.inference_time_seconds is not None
					else "- время инференса, сек.: n/a"
				),
				(
					f"- полное время модели, сек.: {result.wall_time_seconds:.3f}"
					if result.wall_time_seconds is not None
					else "- полное время модели, сек.: n/a"
				),
				"",
				"## Метрики",
				f"- WAPE: {metrics_overall.iloc[0]['WAPE']:.4f}",
				f"- sMAPE: {metrics_overall.iloc[0]['sMAPE']:.4f}",
				f"- MAE: {metrics_overall.iloc[0]['MAE']:.4f}",
				f"- Bias: {metrics_overall.iloc[0]['Bias']:.4f}",
			]
		)


def _serialize_feature_manifest(model_manifest: ModelManifest) -> dict[str, object]:
	features = model_manifest.features
	if not features.enabled:
		return {"enabled": False}

	payload: dict[str, object] = {"enabled": True}
	defaults = ModelManifest(name=model_manifest.name, config=model_manifest.config).features

	for field_name in (
			"feature_set",
			"lags",
			"rolling_windows",
			"use_calendar",
			"use_category_encodings",
			"use_price",
			"use_promo",
			"use_external",
	):
		value = getattr(features, field_name)
		default_value = getattr(defaults, field_name)
		if value != default_value:
			payload[field_name] = value

	return payload
