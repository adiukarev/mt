from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from mt.domain.forecast.forecast_model_adapter import ForecastModelAdapter
from mt.domain.model.model_info import ModelInfo
from mt.domain.probabilistic.probabilistic_settings import DEFAULT_PROBABILISTIC_QUANTILES
from mt.domain.model.model_family import ModelFamily
from mt.domain.model.model_name import ModelName
from mt.domain.model.model_config_dl_manifest import ModelConfigDlManifest, clamp_val_check_steps
from mt.infra.probabilistic.schema import finalize_prediction_frame, has_complete_probabilistic_output


class BaseNeuralForecastAdapter(ForecastModelAdapter):
	"""Общая обертка над neuralforecast с безопасным fallback без hard import-time зависимости"""

	def __init__(
		self,
		model_name: ModelName,
		model_config: ModelConfigDlManifest,
	) -> None:
		super().__init__(ModelInfo(model_name=model_name, model_family=ModelFamily.DL))

		self.model_config = model_config
		self.forecaster: Any | None = None
		self.prediction_column: str | None = None
		self._native_quantiles = DEFAULT_PROBABILISTIC_QUANTILES

	def resolve_feature_columns(
		self,
		prepared_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> list[str]:
		return []

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int,
	) -> None:
		training_df = train_frame.dropna(subset=["sales_units"]).loc[
			:, ["series_id", "week_start", "sales_units"]
		].copy()

		if training_df.empty:
			self.forecaster = None
			self.prediction_column = None
			return

		neuralforecast, model = self._build_runtime_model(horizon=horizon, seed=seed)

		training_df = training_df.rename(
			columns={"series_id": "unique_id", "week_start": "ds", "sales_units": "y"}
		)
		self.forecaster = neuralforecast.NeuralForecast(models=[model], freq="W-MON")
		self.forecaster.fit(df=training_df, val_size=0, verbose=False)
		self.prediction_column = self._resolve_prediction_column_name(model)

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
	) -> pd.DataFrame:
		return predict_frame.dropna(subset=[target_column]).copy()

	def select_inference_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> pd.DataFrame:
		return predict_frame.copy()

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		if predict_frame.empty:
			return np.asarray([], dtype=float)

		if self.prediction_column is None or self.forecaster is None:
			raise ValueError(
				"NeuralForecast adapter is not fitted; saved artifact cannot produce "
				f"native predictions for horizon={horizon}"
			)

		return self._predict_point_from_forecaster(predict_frame, horizon)

	def supports_native_probabilistic(self) -> bool:
		return True

	def predict_quantiles(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		quantiles: tuple[float, ...] = DEFAULT_PROBABILISTIC_QUANTILES,
	) -> pd.DataFrame | None:
		if predict_frame.empty:
			return pd.DataFrame()
		if self.forecaster is None or self.prediction_column is None:
			return None

		target_timestamp = self._resolve_target_timestamp(predict_frame, horizon)
		predictions = self.forecaster.predict(verbose=False, quantiles=list(quantiles), h=horizon)
		target_rows = predictions.loc[predictions["ds"] == target_timestamp].copy()
		if target_rows.empty:
			return pd.DataFrame()
		target_rows = target_rows.drop_duplicates(subset=["unique_id"], keep="first")
		target_rows["unique_id"] = target_rows["unique_id"].astype(str)
		ordered = predict_frame.loc[:, ["series_id"]].copy()
		ordered["series_id"] = ordered["series_id"].astype(str)

		column_map = self._resolve_native_quantile_columns(target_rows.columns.tolist())
		if column_map["q50"] is None:
			return None

		keep_columns = ["unique_id", *[column for column in column_map.values() if column is not None]]
		merged = ordered.merge(
			target_rows.loc[:, keep_columns].copy(),
			left_on="series_id",
			right_on="unique_id",
			how="left",
			sort=False,
		)
		result = pd.DataFrame(
			{
				"q10": _safe_series(merged, column_map["lo_80"]),
				"q50": _safe_series(merged, column_map["q50"]),
				"q90": _safe_series(merged, column_map["hi_80"]),
				"lo_80": _safe_series(merged, column_map["lo_80"]),
				"hi_80": _safe_series(merged, column_map["hi_80"]),
				"lo_95": _safe_series(merged, column_map["lo_95"]),
				"hi_95": _safe_series(merged, column_map["hi_95"]),
			}
		)
		result, _, _ = finalize_prediction_frame(result)
		if not has_complete_probabilistic_output(result):
			return None
		return result

	def _resolve_prediction_column_name(self, model: Any) -> str:
		alias = getattr(model, "alias", None)
		if alias is not None:
			return f"{str(alias)}-median"
		return type(model).__name__

	def _common_model_kwargs(self, horizon: int, seed: int) -> dict[str, object]:
		runtime_config = self._runtime_model_config()
		device = self.model_config.device
		accelerator = "auto"
		if device == "cpu":
			accelerator = "cpu"
		elif device == "mps":
			accelerator = "mps"
		elif device == "cuda":
			accelerator = "gpu"

		weight_decay = runtime_config.weight_decay
		optimizer_kwargs: dict[str, object] | None = None
		optimizer: object | None = None
		if weight_decay > 0.0:
			optimizer = self._resolve_weight_decay_optimizer()
			optimizer_kwargs = {"weight_decay": weight_decay}

		common_kwargs: dict[str, object] = {
			"h": horizon,
			"input_size": runtime_config.history_length,
			"learning_rate": runtime_config.learning_rate,
			"batch_size": runtime_config.batch_size,
			"random_seed": seed,
			"max_steps": runtime_config.max_steps,
			"scaler_type": runtime_config.scaler_type,
			"early_stop_patience_steps": runtime_config.early_stop_patience_steps,
			"val_check_steps": runtime_config.val_check_steps,
			"enable_progress_bar": False,
			"logger": False,
			"enable_checkpointing": False,
			"enable_model_summary": False,
			"num_sanity_val_steps": 0,
			"accelerator": accelerator,
			"devices": 1,
			"alias": self.get_model_info().model_name,
		}
		if optimizer is not None and optimizer_kwargs is not None:
			common_kwargs["optimizer"] = optimizer
			common_kwargs["optimizer_kwargs"] = optimizer_kwargs
		return common_kwargs

	def _resolve_weight_decay_optimizer(self) -> object | None:
		from torch.optim import AdamW

		return AdamW

	def _runtime_model_config(self) -> ModelConfigDlManifest:
		return clamp_val_check_steps(self.model_config)

	def _build_runtime_model(self, horizon: int, seed: int) -> tuple[Any, Any]:
		import neuralforecast

		model = self.build_library_model(
			horizon=horizon,
			seed=seed,
			loss=self._build_loss(),
		)
		return neuralforecast, model

	def _build_loss(self) -> Any:
		from neuralforecast.losses.pytorch import MAE, MQLoss, MSE, RMSE

		loss_name = getattr(self.model_config, "loss_function", "MQLoss")
		if loss_name in {"MQLoss", "Quantile"}:
			return MQLoss(quantiles=list(self._native_quantiles))
		if loss_name == "MAE":
			return MAE()
		if loss_name == "MSE":
			return MSE()
		if loss_name == "RMSE":
			return RMSE()
		raise ValueError(f"Unsupported neuralforecast loss: {loss_name}")

	@abstractmethod
	def build_library_model(self, horizon: int, seed: int, loss: Any) -> Any:
		"""Создать библиотечную модель neuralforecast."""

	def _resolve_native_quantile_columns(self, columns: list[str]) -> dict[str, str | None]:
		result: dict[str, str | None] = {
			"lo_80": None,
			"q50": None,
			"hi_80": None,
			"lo_95": None,
			"hi_95": None,
		}
		for column in columns:
			lower_column = column.lower()
			if lower_column.endswith("-lo-80") or lower_column.endswith("-lo-80.0"):
				result["lo_80"] = column
			elif lower_column.endswith("-median") or lower_column.endswith("-median.0"):
				result["q50"] = column
			elif lower_column.endswith("-hi-80") or lower_column.endswith("-hi-80.0"):
				result["hi_80"] = column
			elif lower_column.endswith("-lo-95") or lower_column.endswith("-lo-95.0"):
				result["lo_95"] = column
			elif lower_column.endswith("-hi-95") or lower_column.endswith("-hi-95.0"):
				result["hi_95"] = column
			elif "_ql0.1" in lower_column or lower_column.endswith("_ql0.10"):
				result["lo_80"] = column
			elif "_ql0.5" in lower_column or lower_column.endswith("_ql0.50"):
				result["q50"] = column
			elif "_ql0.9" in lower_column or lower_column.endswith("_ql0.90"):
				result["hi_80"] = column
			elif "_ql0.025" in lower_column:
				result["lo_95"] = column
			elif "_ql0.975" in lower_column:
				result["hi_95"] = column
		return result

	def _predict_point_from_forecaster(
		self,
		predict_frame: pd.DataFrame,
		horizon: int,
	) -> np.ndarray:
		target_timestamp = self._resolve_target_timestamp(predict_frame, horizon)
		predictions = self.forecaster.predict(verbose=False)
		target_rows = predictions.loc[
			predictions["ds"] == target_timestamp, ["unique_id", self.prediction_column]
		].copy()
		if target_rows.empty:
			available_dates = sorted(pd.to_datetime(predictions["ds"]).dt.strftime("%Y-%m-%d").unique())
			raise ValueError(
				"NeuralForecast saved artifact cannot predict requested target_date: "
				f"requested={target_timestamp.date()}, available={available_dates[:5]}"
			)
		target_rows = target_rows.drop_duplicates(subset=["unique_id"], keep="first")
		target_rows["unique_id"] = target_rows["unique_id"].astype(str)

		ordered = predict_frame.loc[:, ["series_id"]].copy()
		ordered["series_id"] = ordered["series_id"].astype(str)
		merged = ordered.merge(
			target_rows,
			left_on="series_id",
			right_on="unique_id",
			how="left",
			sort=False,
		)
		if merged[self.prediction_column].isna().any():
			missing_series = merged.loc[merged[self.prediction_column].isna(), "series_id"].tolist()
			raise ValueError(
				"NeuralForecast saved artifact did not return predictions for all series: "
				f"missing={missing_series[:5]}"
			)
		return merged[self.prediction_column].to_numpy(dtype=float)

	def _resolve_target_timestamp(
		self,
		predict_frame: pd.DataFrame,
		horizon: int,
	) -> pd.Timestamp:
		forecast_origin = pd.Timestamp(predict_frame["week_start"].iloc[0])
		return forecast_origin + pd.Timedelta(weeks=int(horizon))


def _safe_series(frame: pd.DataFrame, column_name: str | None) -> pd.Series:
	if column_name is None or column_name not in frame.columns:
		return pd.Series(np.nan, index=frame.index, dtype=float)
	return frame[column_name].astype(float)
