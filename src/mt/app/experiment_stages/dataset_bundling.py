from mt.domain.experiment import ExperimentPipelineContext
from mt.app.base_stages.dataset_bundling import BaseDatasetBundlingStage


class DatasetBundlingStage(BaseDatasetBundlingStage):
	name = "experiment_dataset_bundling"

	def get_dataset_manifest(self, ctx: ExperimentPipelineContext):
		return ctx.manifest.dataset
