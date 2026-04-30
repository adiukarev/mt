from mt.domain.dataset.dataset_source_type import DatasetSourceType
from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.domain.pipeline.pipeline_stage import BasePipelineStage
from mt.infra.dataset.sources.source_resolver import build_dataset_source_service
from mt.infra.dataset.sources.synthetic_refresh import build_synthetic_dataset_load_data


class ExperimentDataSourceSyncPipelineStage(BasePipelineStage):
	def execute(self, ctx: ExperimentPipelineContext) -> None:
		source_manifest = ctx.manifest.source
		if source_manifest is None:
			return

		ctx.source_descriptor = {
			"source_type": source_manifest.source_type.value,
			"dataset_path": ctx.manifest.dataset.path,
		}
		result = build_dataset_source_service(source_manifest).refresh_dataset(ctx.manifest.dataset)
		ctx.source_descriptor.update(result.source_descriptor)
		ctx.source_descriptor["materialized_paths"] = result.materialized_paths
		if source_manifest.source_type == DatasetSourceType.SYNTHETIC:
			ctx.raw_dataset = build_synthetic_dataset_load_data(ctx.manifest.dataset, result)
