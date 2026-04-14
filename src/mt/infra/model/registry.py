from mt.domain.manifest import DLManifest
from mt.domain.model import ForecastModelAdapter


def build_model_adapter(
	model_name: str,
	model_params: dict[str, object] | None = None,
	dl_manifest: DLManifest | None = None
) -> ForecastModelAdapter:
	"""Создать сопоставимый адаптер модели по конфигурации"""

	name = model_name.lower()
	model_params = {} if model_params is None else dict(model_params)

	if name == "naive":
		from mt.infra.model.adapters.naive import NaiveAdapter

		return NaiveAdapter()
	if name == "seasonal_naive":
		from mt.infra.model.adapters.seasonal_naive import SeasonalNaiveAdapter

		return SeasonalNaiveAdapter()
	if name == "ets":
		from mt.infra.model.adapters.ets import ETSAdapter

		return ETSAdapter(model_params)
	if name == "lightgbm":
		from mt.infra.model.adapters.lightgbm import LightGBMAdapter

		return LightGBMAdapter(model_params)
	if name == "ridge":
		from mt.infra.model.adapters.ridge import RidgeAdapter

		return RidgeAdapter(model_params)
	if name == "catboost":
		from mt.infra.model.adapters.catboost import CatBoostAdapter

		return CatBoostAdapter(model_params)
	if name == "mlp":
		from mt.infra.model.adapters.mlp import MLPAdapter

		return MLPAdapter(dl_manifest, model_params)
	if name == "nbeats":
		from mt.infra.model.adapters.nbeats import NBeatsAdapter

		return NBeatsAdapter(dl_manifest, model_params)

	raise ValueError()
