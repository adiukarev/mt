from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mt.domain.dataset import DatasetBundle, DatasetLoadData
from mt.domain.manifest import ExperimentManifest, FeatureManifest, ModelManifest
from mt.domain.pipeline_context import BasePipelineContext
from mt.infra.analysis.comparison import ComparisonArtifacts


@dataclass(slots=True)
class ExperimentArtifactPathsMap:
	"""Артефакты эксперимента"""

	root: Path
	models: Path
	reports: Path
	tables: Path
	plots: Path


@dataclass(slots=True)
class ExperimentPipelineContext(BasePipelineContext):
	"""Общий стейт контекст между этапами experiment pipeline"""

	# манифест запуска
	manifest: ExperimentManifest
	artifacts_paths_map: ExperimentArtifactPathsMap
	# сырые таблицы
	raw_dataset: DatasetLoadData | None = None
	# подготовленный датасет
	dataset: DatasetBundle | None = None
	# сегменты рядов
	segments: pd.DataFrame | None = None
	# реестр всех фич которые вообще в этом прогоне есть
	feature_registry: pd.DataFrame | None = None
	# колонки из supervised которые можно отдать модели
	feature_columns: list[str] = field(default_factory=list)
	# общий набор фич для всех enabled models
	feature_manifest: FeatureManifest = field(default_factory=FeatureManifest)
	# общая supervised таблица
	supervised: pd.DataFrame | None = None
	# rolling окна
	windows: pd.DataFrame | None = None
	# прогнозы вместе
	predictions: pd.DataFrame | None = None
	# строки для run catalog
	model_catalog_rows: list[dict[str, object]] = field(default_factory=list)
	# кто какие фичи использовал
	model_feature_usage_rows: list[dict[str, object]] = field(default_factory=list)
	# итоговые метрики
	overall_metrics: pd.DataFrame | None = None
	# метрики по горизонтам
	by_horizon_metrics: pd.DataFrame | None = None
	# артефакты сравнения
	comparison: ComparisonArtifacts | None = None
	# лучшая модель и финальный артефакт, переобученный на полной истории
	selected_model_name: str | None = None
	selected_model_metrics: dict[str, Any] = field(default_factory=dict)
	best_model_artifact_path: Path | None = None
	best_model_report_path: Path | None = None
	# общий run catalog
	run_catalog_rows: list[dict[str, object]] = field(default_factory=list)
	# модели которые реально включены
	model_manifests: list[ModelManifest] = field(default_factory=list)
