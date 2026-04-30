import os
from pathlib import Path

_LOCAL_AIRFLOW_HOME = Path(__file__).resolve().parents[1] / ".airflow"
_LOCAL_AIRFLOW_LOGS = _LOCAL_AIRFLOW_HOME / "logs"
_LOCAL_MPLCONFIGDIR = Path(__file__).resolve().parents[1] / ".matplotlib"
_LOCAL_AIRFLOW_HOME.mkdir(parents=True, exist_ok=True)
_LOCAL_AIRFLOW_LOGS.mkdir(parents=True, exist_ok=True)
_LOCAL_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("AIRFLOW_HOME", str(_LOCAL_AIRFLOW_HOME))
os.environ.setdefault("AIRFLOW__LOGGING__BASE_LOG_FOLDER", str(_LOCAL_AIRFLOW_LOGS))
os.environ.setdefault("MPLCONFIGDIR", str(_LOCAL_MPLCONFIGDIR))

import airflow

from mt.orchestration.dag_builder import PipelineDagSettings, build_pipeline_dag
from mt.orchestration.monitoring_experiment_trigger import (
	build_monitoring_experiment_trigger_conf,
)

# m5

## m5 audit

audit_m5_category_pipeline = build_pipeline_dag(
	dag_id="audit_m5_category_pipeline",
	pipeline_type="audit",
	default_manifest_path="manifests/m5/audit_category.yaml",
	description="Пайплайн аудита retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "audit", "curated")),
)

audit_m5_sku_pipeline = build_pipeline_dag(
	dag_id="audit_m5_sku_pipeline",
	pipeline_type="audit",
	default_manifest_path="manifests/m5/audit_sku.yaml",
	description="Пайплайн аудита retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "audit", "curated")),
)

## m5 experiment

experiment_m5_category_single_pipeline = build_pipeline_dag(
	dag_id="experiment_m5_category_single_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/m5/experiment_category_single.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("m5", "experiment", "single")),
)

experiment_m5_category_curated_pipeline = build_pipeline_dag(
	dag_id="experiment_m5_category_curated_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/m5/experiment_category_curated.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("m5", "experiment", "curated")),
)

experiment_m5_sku_single_pipeline = build_pipeline_dag(
	dag_id="experiment_m5_sku_single_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/m5/experiment_sku_single.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("m5", "experiment", "single")),
)

experiment_m5_sku_curated_pipeline = build_pipeline_dag(
	dag_id="experiment_m5_sku_curated_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/m5/experiment_sku_curated.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("m5", "experiment", "curated")),
)

## m5 forecast

forecast_m5_category_single_pipeline = build_pipeline_dag(
	dag_id="forecast_m5_category_single_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/m5/forecast_category_single.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "forecast", "single")),
)

forecast_m5_category_curated_pipeline = build_pipeline_dag(
	dag_id="forecast_m5_category_curated_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/m5/forecast_category_curated.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "forecast", "curated")),
)

forecast_m5_sku_single_pipeline = build_pipeline_dag(
	dag_id="forecast_m5_sku_single_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/m5/forecast_sku_single.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "forecast", "single")),
)

forecast_m5_sku_curated_pipeline = build_pipeline_dag(
	dag_id="forecast_m5_sku_curated_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/m5/forecast_sku_curated.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("m5", "forecast", "curated")),
)


# favorita

## favorita audit

audit_favorita_category_pipeline = build_pipeline_dag(
	dag_id="audit_favorita_category_pipeline",
	pipeline_type="audit",
	default_manifest_path="manifests/favorita/audit_category.yaml",
	description="Пайплайн аудита retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "audit")),
)

audit_favorita_sku_pipeline = build_pipeline_dag(
	dag_id="audit_favorita_sku_pipeline",
	pipeline_type="audit",
	default_manifest_path="manifests/favorita/audit_sku.yaml",
	description="Пайплайн аудита retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "audit")),
)

## favorita experiment

experiment_favorita_category_single_pipeline = build_pipeline_dag(
	dag_id="experiment_favorita_category_single_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/favorita/experiment_category_single.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("favorita", "experiment", "single")),
)

experiment_favorita_category_curated_pipeline = build_pipeline_dag(
	dag_id="experiment_favorita_category_curated_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/favorita/experiment_category_curated.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("favorita", "experiment", "curated")),
)

experiment_favorita_sku_single_pipeline = build_pipeline_dag(
	dag_id="experiment_favorita_sku_single_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/favorita/experiment_sku_single.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("favorita", "experiment", "single")),
)

experiment_favorita_sku_curated_pipeline = build_pipeline_dag(
	dag_id="experiment_favorita_sku_curated_pipeline",
	pipeline_type="experiment",
	default_manifest_path="manifests/favorita/experiment_sku_curated.yaml",
	description="Пайплайн эксперимента c retail-датасетом",
	settings=PipelineDagSettings(dag_tags=("favorita", "experiment", "curated")),
)

## favorita forecast

forecast_favorita_category_single_pipeline = build_pipeline_dag(
	dag_id="forecast_favorita_category_single_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/favorita/forecast_category_single.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "forecast", "single")),
)

forecast_favorita_category_curated_pipeline = build_pipeline_dag(
	dag_id="forecast_favorita_category_curated_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/favorita/forecast_category_curated.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "forecast", "curated")),
)

forecast_favorita_sku_single_pipeline = build_pipeline_dag(
	dag_id="forecast_favorita_sku_single_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/favorita/forecast_sku_single.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "forecast", "single")),
)

forecast_favorita_sku_curated_pipeline = build_pipeline_dag(
	dag_id="forecast_favorita_sku_curated_pipeline",
	pipeline_type="forecast",
	default_manifest_path="manifests/favorita/forecast_sku_curated.yaml",
	description="Пайплайн предсказания на основе сохранённых артефактов лучшей модели и retail-датасета",
	settings=PipelineDagSettings(dag_tags=("favorita", "forecast", "curated")),
)

# synthetic

synthetic_generation_pipeline = build_pipeline_dag(
	dag_id="synthetic_generation_pipeline",
	pipeline_type="synthetic_generation",
	default_manifest_path="manifests/synthetic_generation.yaml",
	description="Пайплайн генерации синтетического retail-датасета",
	settings=PipelineDagSettings(
		schedule="0 5 * * 1",
		dag_tags=("scheduled", "synthetic"),
		max_active_runs=1,
	),
)

# monitoring

monitoring_pipeline = build_pipeline_dag(
	dag_id="monitoring_pipeline",
	pipeline_type="monitoring",
	default_manifest_path="manifests/synthetic_monitoring.yaml",
	description="Synthetic-first monitoring pipeline with drift checks and decision artifact publishing",
	settings=PipelineDagSettings(
		schedule="0 6 * * 1",
		dag_tags=("scheduled", "monitoring"),
		max_active_runs=1,
	),
	trigger_dag_id="experiment_pipeline",
	trigger_conf_builder=build_monitoring_experiment_trigger_conf,
)
