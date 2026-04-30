from dataclasses import asdict, dataclass, fields
from typing import Any, TypeAlias


@dataclass(slots=True)
class ModelConfigMlpManifest:
	history_length: int = 52
	hidden_size: int = 128
	n_layers: int = 4
	learning_rate: float = 0.05
	loss_function: str = "MQLoss"
	batch_size: int = 32
	weight_decay: float = 1e-5
	device: str = "cpu"
	scaler_type: str = "robust"
	max_steps: int = 80
	early_stop_patience_steps: int = -1
	val_check_steps: int = 100

	def __post_init__(self) -> None:
		if self.history_length < 8 or self.hidden_size < 8:
			raise ValueError()
		if self.n_layers < 1 or self.batch_size < 1:
			raise ValueError()
		if self.learning_rate <= 0.0:
			raise ValueError()
		if self.weight_decay < 0.0 or self.max_steps < 1 or self.val_check_steps < 1:
			raise ValueError()
		if not self.device or not self.scaler_type:
			raise ValueError()
		if self.loss_function not in {"MQLoss", "Quantile", "MAE", "MSE", "RMSE"}:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigMlpManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


@dataclass(slots=True)
class ModelConfigNBeatsManifest:
	history_length: int = 52
	hidden_size: int = 128
	n_blocks: int = 3
	n_layers: int = 4
	learning_rate: float = 0.05
	loss_function: str = "MQLoss"
	batch_size: int = 32
	weight_decay: float = 1e-5
	device: str = "cpu"
	scaler_type: str = "robust"
	max_steps: int = 80
	early_stop_patience_steps: int = -1
	val_check_steps: int = 100
	dropout: float = 0.0
	activation: str = "ReLU"

	def __post_init__(self) -> None:
		if self.history_length < 8 or self.hidden_size < 8:
			raise ValueError()
		if self.n_blocks < 1 or self.n_layers < 1 or self.batch_size < 1:
			raise ValueError()
		if self.learning_rate <= 0.0:
			raise ValueError()
		if self.weight_decay < 0.0 or self.max_steps < 1 or self.val_check_steps < 1:
			raise ValueError()
		# if self.dropout < 0.0 or self.dropout >= 1.0:
		# 	raise ValueError()
		if self.dropout > 0.0:
			raise ValueError(
				"N-BEATS dropout must be 0.0; current neuralforecast NBEATS does not support dropout"
			)
		if not self.device or not self.scaler_type or not self.activation:
			raise ValueError()
		if self.loss_function not in {"MQLoss", "Quantile", "MAE", "MSE", "RMSE"}:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigNBeatsManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return asdict(self)


ModelConfigDlManifest: TypeAlias = ModelConfigMlpManifest | ModelConfigNBeatsManifest


def clamp_val_check_steps(config: ModelConfigDlManifest) -> ModelConfigDlManifest:
	resolved_val_check_steps = min(config.val_check_steps, config.max_steps)
	if isinstance(config, ModelConfigMlpManifest):
		return ModelConfigMlpManifest(
			history_length=config.history_length,
			hidden_size=config.hidden_size,
			n_layers=config.n_layers,
			learning_rate=config.learning_rate,
			loss_function=config.loss_function,
			batch_size=config.batch_size,
			weight_decay=config.weight_decay,
			device=config.device,
			scaler_type=config.scaler_type,
			max_steps=config.max_steps,
			early_stop_patience_steps=config.early_stop_patience_steps,
			val_check_steps=resolved_val_check_steps,
		)

	return ModelConfigNBeatsManifest(
		history_length=config.history_length,
		hidden_size=config.hidden_size,
		n_blocks=config.n_blocks,
		n_layers=config.n_layers,
		learning_rate=config.learning_rate,
		loss_function=config.loss_function,
		batch_size=config.batch_size,
		weight_decay=config.weight_decay,
		device=config.device,
		scaler_type=config.scaler_type,
		max_steps=config.max_steps,
		early_stop_patience_steps=config.early_stop_patience_steps,
		val_check_steps=resolved_val_check_steps,
		dropout=config.dropout,
		activation=config.activation,
	)
