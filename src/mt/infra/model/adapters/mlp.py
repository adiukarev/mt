from typing import Any

from mt.domain.model.model_config_dl_manifest import ModelConfigMlpManifest
from mt.domain.model.model_name import ModelName
from mt.infra.model.adapters.base_neuralforecast import BaseNeuralForecastAdapter


class MLPAdapter(BaseNeuralForecastAdapter):
	"""neuralforecast.MLP"""

	def __init__(self, model_config: ModelConfigMlpManifest) -> None:
		super().__init__(model_name=ModelName.MLP, model_config=model_config)

	def build_library_model(self, horizon: int, seed: int, loss: Any) -> Any:
		from neuralforecast.models import MLP

		return MLP(
			num_layers=self.model_config.n_layers,
			hidden_size=self.model_config.hidden_size,
			loss=loss,
			**self._common_model_kwargs(horizon=horizon, seed=seed),
		)
