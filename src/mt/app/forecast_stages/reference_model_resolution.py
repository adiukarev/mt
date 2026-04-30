from pathlib import Path

from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.forecast.forecast_pipeline_context import ForecastPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.forecast.reference_model import ReferenceModelConfig, load_reference_model_config


class ForecastReferenceModelResolutionPipelineStage(BasePipelineStage):
	def execute(self, ctx: ForecastPipelineContext) -> None:
		execution_mode = ctx.observability.execution_mode if ctx.observability else None
		ctx.reference_model = load_reference_model_config(
			model_manifest=ctx.manifest.model,
			dataset_manifest=ctx.manifest.dataset,
			execution_mode=execution_mode,
		)
		ctx.resolved_dataset_manifest = self._resolve_dataset_manifest(
			ctx,
			ctx.reference_model,
		)

	def _resolve_dataset_manifest(
		self,
		ctx: ForecastPipelineContext,
		reference_model: ReferenceModelConfig,
	) -> DatasetManifest:
		if ctx.manifest.dataset is not None:
			return self._with_reference_model_scope(
				ctx.manifest.dataset,
				reference_model,
			)

		if ctx.manifest.dataset_path is None:
			raise ValueError()

		source = self._require_training_dataset_manifest(reference_model)
		path = ctx.manifest.dataset_path
		kind = "synthetic" if Path(path).suffix.lower() == ".csv" else source.kind

		return DatasetManifest(
			kind=kind,
			path=path,
			aggregation_level=source.aggregation_level,
			target_name=source.target_name,
			week_anchor=source.week_anchor,
			series_limit=source.series_limit,
			series_allowlist=self._copy_series_allowlist(source),
		)

	def _with_reference_model_scope(
		self,
		requested: DatasetManifest,
		reference_model: ReferenceModelConfig,
	) -> DatasetManifest:
		source = self._require_training_dataset_manifest(reference_model)
		if requested.kind != source.kind:
			raise ValueError("Forecast dataset kind does not match reference model")
		if requested.aggregation_level != source.aggregation_level:
			raise ValueError(
				"Forecast aggregation level does not match reference model"
			)
		if requested.target_name != source.target_name:
			raise ValueError("Forecast target name does not match reference model")
		if requested.week_anchor != source.week_anchor:
			raise ValueError("Forecast week anchor does not match reference model")

		return DatasetManifest(
			kind=requested.kind,
			path=requested.path,
			aggregation_level=requested.aggregation_level,
			target_name=requested.target_name,
			week_anchor=requested.week_anchor,
			series_limit=source.series_limit,
			series_allowlist=self._copy_series_allowlist(source),
		)

	def _require_training_dataset_manifest(
		self,
		reference_model: ReferenceModelConfig,
	) -> DatasetManifest:
		if reference_model.training_dataset_manifest is not None:
			return reference_model.training_dataset_manifest
		if reference_model.artifact is not None:
			return reference_model.artifact.dataset_manifest

		raise ValueError("Reference model bundle does not include dataset manifest")

	def _copy_series_allowlist(self, source: DatasetManifest) -> list[str] | None:
		if source.series_allowlist is None:
			return None
		return list(source.series_allowlist)
