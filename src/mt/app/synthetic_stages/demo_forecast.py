from __future__ import annotations

from time import perf_counter

import pandas as pd

from mt.domain.stage import BaseStage
from mt.domain.synthetic_generation import SyntheticGenerationPipelineContext
from mt.infra.synthetic.generator import build_demo_forecast_frame


class SyntheticDemoForecastStage(BaseStage):
	name = "synthetic_demo_forecast"

	def execute(self, ctx: SyntheticGenerationPipelineContext) -> None:
		if ctx.dataset is None:
			raise ValueError()

		ctx.demo_forecast = (
			build_demo_forecast_frame(ctx.dataset, ctx.manifest)
			if ctx.manifest.demo_forecast.enabled
			else pd.DataFrame()
		)
