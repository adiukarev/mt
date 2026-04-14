from __future__ import annotations

from time import perf_counter

from mt.domain.predict import PredictPipelineContext
from mt.domain.stage import BaseStage
from mt.infra.synthetic.predict import load_reference_model_config


class PredictReferenceModelResolutionStage(BaseStage):
	name = "predict_reference_model_resolution"

	def execute(self, ctx: PredictPipelineContext) -> None:
		ctx.reference_model = load_reference_model_config(ctx.manifest.model_source.best_model_dir)
