from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mt.domain.pipeline_context import BasePipelineContext
from mt.domain.predict_manifest import SyntheticPredictManifest
from mt.infra.synthetic.predict import ReferenceModelConfig


@dataclass(slots=True)
class PredictArtifactPathsMap:
	"""Разметка каталогов predict pipeline."""

	root: Path
	dataset: Path
	forecast: Path
	run: Path


@dataclass(slots=True)
class PredictPipelineContext(BasePipelineContext):
	"""Состояние synthetic predict pipeline."""

	manifest: SyntheticPredictManifest
	artifacts_paths_map: PredictArtifactPathsMap
	frame: pd.DataFrame | None = None
	reference_model: ReferenceModelConfig | None = None
	resolved_horizon: int | None = None
	predictions: pd.DataFrame | None = None
	metrics: pd.DataFrame | None = None
	pipeline_steps: list[dict[str, object]] = field(default_factory=list)
