from typing import Any

from mt.domain.model.model_config_dl_manifest import ModelConfigNBeatsManifest
from mt.domain.model.model_name import ModelName
from mt.infra.model.adapters.base_neuralforecast import BaseNeuralForecastAdapter


class NBeatsAdapter(BaseNeuralForecastAdapter):
	"""neuralforecast.NBEATS"""

	def __init__(self, model_config: ModelConfigNBeatsManifest) -> None:
		super().__init__(model_name=ModelName.NBEATS, model_config=model_config)

	def build_library_model(self, horizon: int, seed: int, loss: Any) -> Any:
		from neuralforecast.models import NBEATS

		hidden_layers = [self.model_config.hidden_size] * max(2, self.model_config.n_layers)
		n_blocks = max(1, self.model_config.n_blocks)
		stack_types = ["identity"] if horizon == 1 else ["identity", "trend", "seasonality"]
		stack_blocks = [n_blocks] if horizon == 1 else [1, 1, n_blocks]
		stack_mlp_units = [hidden_layers] if horizon == 1 else [hidden_layers, hidden_layers,
		                                                        hidden_layers]

		return NBEATS(
			stack_types=stack_types,
			n_blocks=stack_blocks,
			mlp_units=stack_mlp_units,
			dropout_prob_theta=self.model_config.dropout,
			activation=self.model_config.activation,
			loss=loss,
			**self._common_model_kwargs(horizon=horizon, seed=seed),
		)
