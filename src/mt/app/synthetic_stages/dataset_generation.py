from __future__ import annotations

from time import perf_counter

from mt.domain.stage import BaseStage
from mt.domain.synthetic_generation import SyntheticGenerationPipelineContext
from mt.infra.synthetic.generator import build_series_metadata, generate_dataset_frame


class SyntheticDatasetGenerationStage(BaseStage):
	name = "synthetic_dataset_generation"

	def execute(self, ctx: SyntheticGenerationPipelineContext) -> None:
		ctx.dataset = generate_dataset_frame(ctx.manifest)
		ctx.metadata = build_series_metadata(ctx.dataset)
