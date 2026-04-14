from __future__ import annotations

from neuralforecast.common._base_model import BaseModel
from neuralforecast.models import MLP

from mt.domain.manifest import DLManifest
from mt.infra.model.adapters.base_neuralforecast import BaseNeuralForecastAdapter


class MLPAdapter(BaseNeuralForecastAdapter):
	"""Библиотечный neuralforecast.MLP"""

	def __init__(self, manifest: DLManifest, params: dict[str, object] | None = None) -> None:
		super().__init__(
			model_name="mlp",
			manifest=manifest,
			params=params,
		)

	def build_library_model(self, horizon: int, seed: int) -> BaseModel:
		return MLP(
			num_layers=int(self.manifest.n_layers),
			hidden_size=int(self.manifest.hidden_size),
			**self._common_model_kwargs(horizon=horizon, seed=seed),
		)
