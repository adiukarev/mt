from collections.abc import Callable

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_config_dl_manifest import (
	ModelConfigMlpManifest,
	ModelConfigNBeatsManifest,
)
from mt.domain.model.model_config_ml_manifest import (
	ModelConfigCatBoostManifest,
	ModelConfigLightGBMManifest,
	ModelConfigRidgeManifest,
)
from mt.domain.model.model_config_manifest import (
	ModelConfigManifest,
	build_model_config,
	serialize_model_config
)
from mt.domain.model.model_name import ModelName, normalize_model_name
from mt.domain.model.model_config_statistical_manifest import ModelConfigEtsManifest

AdapterBuilder = Callable[[ModelConfigManifest | None], ForecastModelAdapter]


def _build_naive_adapter(_: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.naive import NaiveAdapter

	return NaiveAdapter()


def _build_seasonal_naive_adapter(_: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.seasonal_naive import SeasonalNaiveAdapter

	return SeasonalNaiveAdapter()


def _build_ets_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.ets import ETSAdapter

	if not isinstance(model_config, ModelConfigEtsManifest):
		raise TypeError()
	return ETSAdapter(model_config)


def _build_lightgbm_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.lightgbm import LightGBMAdapter

	if not isinstance(model_config, ModelConfigLightGBMManifest):
		raise TypeError()
	return LightGBMAdapter(model_config)


def _build_ridge_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.ridge import RidgeAdapter

	if not isinstance(model_config, ModelConfigRidgeManifest):
		raise TypeError()
	return RidgeAdapter(model_config)


def _build_catboost_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.catboost import CatBoostAdapter

	if not isinstance(model_config, ModelConfigCatBoostManifest):
		raise TypeError()
	return CatBoostAdapter(model_config)


def _build_mlp_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.mlp import MLPAdapter

	if not isinstance(model_config, ModelConfigMlpManifest):
		raise TypeError()
	return MLPAdapter(model_config)


def _build_nbeats_adapter(model_config: ModelConfigManifest | None) -> ForecastModelAdapter:
	from mt.infra.model.adapters.nbeats import NBeatsAdapter

	if not isinstance(model_config, ModelConfigNBeatsManifest):
		raise TypeError()
	return NBeatsAdapter(model_config)


MODEL_ADAPTER_BUILDERS: dict[ModelName, AdapterBuilder] = {
	ModelName.NAIVE: _build_naive_adapter,
	ModelName.SEASONAL_NAIVE: _build_seasonal_naive_adapter,
	ModelName.ETS: _build_ets_adapter,
	ModelName.LIGHTGBM: _build_lightgbm_adapter,
	ModelName.RIDGE: _build_ridge_adapter,
	ModelName.CATBOOST: _build_catboost_adapter,
	ModelName.MLP: _build_mlp_adapter,
	ModelName.NBEATS: _build_nbeats_adapter,
}


def build_model_adapter(
	model_name: str | ModelName,
	config: ModelConfigManifest | dict[str, object] | None = None,
) -> ForecastModelAdapter:
	"""Создать сопоставимый адаптер модели по конфигурации"""

	resolved_model_name = normalize_model_name(model_name)
	config_payload = (
		serialize_model_config(config)
		if config is not None and not isinstance(config, dict)
		else config
	)
	resolved_model_config = build_model_config(resolved_model_name, config_payload)

	try:
		return MODEL_ADAPTER_BUILDERS[resolved_model_name](resolved_model_config)
	except KeyError as error:
		raise ValueError() from error
