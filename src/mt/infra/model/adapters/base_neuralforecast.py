from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.common._base_model import BaseModel

from mt.domain.manifest import DLManifest
from mt.domain.model import ForecastModelAdapter, ModelInfo


class BaseNeuralForecastAdapter(ForecastModelAdapter):
	"""Общая обертка над библиотечными моделями neuralforecast"""

	def __init__(
		self,
		model_name: str,
		manifest: DLManifest,
		params: dict[str, object] | None = None
	) -> None:
		super().__init__(ModelInfo(model_name=model_name, model_family="dl"))

		self.manifest = manifest
		self.params = {} if params is None else dict(params)
		self.forecaster: NeuralForecast | None = None
		self.prediction_column: str | None = None

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
		training_df = train_frame.dropna(subset=[target_column]).loc[
			:, ["series_id", "week_start", target_column]].copy()

		if training_df.empty:
			self.forecaster = None
			self.prediction_column = None
			return

		training_df = training_df.rename(
			columns={
				"series_id": "unique_id",
				"week_start": "ds",
				target_column: "y",
			}
		)

		model = self.build_library_model(horizon=horizon, seed=seed)

		self.forecaster = NeuralForecast(models=[model], freq="W-MON")
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
		if self.forecaster is None or self.prediction_column is None or predict_frame.empty:
			return np.asarray([], dtype=float)

		forecast_origin = pd.Timestamp(predict_frame["week_start"].iloc[0])
		predictions = self.forecaster.predict(verbose=False)

		target_rows = predictions.loc[
			predictions["ds"] == forecast_origin, ["unique_id", self.prediction_column]].copy()
		if target_rows.empty:
			return np.asarray([], dtype=float)
		target_rows = target_rows.drop_duplicates(subset=["unique_id"], keep="first")

		ordered = predict_frame.loc[:, ["series_id"]].copy()
		ordered["series_id"] = ordered["series_id"].astype(str)
		target_rows["unique_id"] = target_rows["unique_id"].astype(str)

		merged = ordered.merge(
			target_rows,
			left_on="series_id",
			right_on="unique_id",
			how="left",
			sort=False
		)

		return merged[self.prediction_column].to_numpy(dtype=float)

	def _resolve_prediction_column_name(self, model: BaseModel) -> str:
		alias = getattr(model, "alias", None)

		if alias is not None:
			return str(alias)

		return type(model).__name__

	def _common_model_kwargs(self, horizon: int, seed: int) -> dict[str, object]:
		device = str(self.params.get("device", self.manifest.device))
		accelerator = "auto"
		if device == "cpu":
			accelerator = "cpu"
		elif device == "mps":
			accelerator = "mps"
		elif device == "cuda":
			accelerator = "gpu"

		weight_decay = float(self.params.get("weight_decay", self.manifest.weight_decay))
		optimizer_kwargs: dict[str, object] | None = None
		if weight_decay > 0.0:
			optimizer_kwargs = {"weight_decay": weight_decay}

		return {
			"h": horizon,
			"input_size": int(self.manifest.history_length),
			"learning_rate": float(self.params.get("learning_rate", self.manifest.learning_rate)),
			"batch_size": int(self.params.get("batch_size", self.manifest.batch_size)),
			"random_seed": seed,
			"max_steps": int(self.params.get("max_steps", self.manifest.epochs)),
			"scaler_type": str(self.params.get("scaler_type", "robust")),
			"early_stop_patience_steps": int(self.params.get("early_stop_patience_steps", -1)),
			"val_check_steps": int(self.params.get("val_check_steps", 100)),
			"enable_progress_bar": False,
			"logger": False,
			"enable_checkpointing": False,
			"enable_model_summary": False,
			"num_sanity_val_steps": 0,
			"accelerator": accelerator,
			"devices": 1,
			"alias": self.get_model_info().model_name,
			"optimizer_kwargs": optimizer_kwargs,
		}

	@abstractmethod
	def build_library_model(self, horizon: int, seed: int) -> BaseModel:
		"""Создать библиотечную модель neuralforecast"""
