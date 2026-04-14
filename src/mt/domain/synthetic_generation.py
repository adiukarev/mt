from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mt.domain.pipeline_context import BasePipelineContext
from mt.domain.synthetic import SyntheticManifest


@dataclass(slots=True)
class SyntheticArtifactPathsMap:
	"""Разметка каталогов synthetic generation pipeline"""

	root: Path
	dataset: Path
	preview: Path
	run: Path


@dataclass(slots=True)
class SyntheticGenerationPipelineContext(BasePipelineContext):
	"""Состояние synthetic generation pipeline"""

	manifest: SyntheticManifest
	artifacts_paths_map: SyntheticArtifactPathsMap
	dataset: pd.DataFrame | None = None
	metadata: pd.DataFrame | None = None
	demo_forecast: pd.DataFrame | None = None
	pipeline_steps: list[dict[str, object]] = field(default_factory=list)
