"""Compatibility facade for forecast inference.

New code should import from the role-specific modules in this package:
`reference_model`, `prediction_builder`, `metrics_builder`, `inference_policy`, and `frame`.
"""

from mt.infra.forecast.frame import infer_horizon
from mt.infra.forecast.inference_policy import resolve_forecast_inference_policy
from mt.infra.forecast.metrics_builder import build_metrics, build_probabilistic_metrics
from mt.infra.forecast.prediction_builder import (
	build_saved_model_predictions,
	prepare_full_weekly,
	prepare_history_weekly,
	run_saved_model_forecast_window,
	validate_saved_model_predictions as _validate_saved_model_predictions,
)
from mt.infra.forecast.reference_model import ReferenceModelConfig, load_reference_model_config
