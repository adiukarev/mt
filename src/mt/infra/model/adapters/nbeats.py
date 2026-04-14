from __future__ import annotations

from neuralforecast.common._base_model import BaseModel
from neuralforecast.models import NBEATS

from mt.domain.manifest import DLManifest
from mt.infra.model.adapters.base_neuralforecast import BaseNeuralForecastAdapter


class NBeatsAdapter(BaseNeuralForecastAdapter):
	"""Библиотечный neuralforecast.NBEATS"""

	def __init__(self, manifest: DLManifest, params: dict[str, object] | None = None) -> None:
		super().__init__(
			model_name="nbeats",
			manifest=manifest,
			params=params,
		)

	def build_library_model(self, horizon: int, seed: int) -> BaseModel:
		hidden_layers = [int(self.manifest.hidden_size)] * max(2, int(self.manifest.n_layers))
		n_blocks = max(1, int(self.manifest.n_blocks))
		stack_types = ["identity"] if horizon == 1 else ["identity", "trend", "seasonality"]
		stack_blocks = [n_blocks] if horizon == 1 else [1, 1, n_blocks]
		stack_mlp_units = [hidden_layers] if horizon == 1 else [hidden_layers, hidden_layers,
		                                                        hidden_layers]

		return NBEATS(
			stack_types=stack_types,
			n_blocks=stack_blocks,
			mlp_units=stack_mlp_units,
			dropout_prob_theta=float(self.params.get("dropout", 0.0)),
			activation=str(self.params.get("activation", "ReLU")),
			**self._common_model_kwargs(horizon=horizon, seed=seed),
		)
