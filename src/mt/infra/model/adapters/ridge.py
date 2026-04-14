import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from mt.domain.model import ModelInfo, ForecastModelAdapter


class RidgeAdapter(ForecastModelAdapter):
	"""Простой глобальный линейный baseline на Ridge regression"""

	def __init__(self, params: dict[str, object] | None = None) -> None:
		super().__init__(ModelInfo(model_name="ridge", model_family="ml"))

		self.params = {} if params is None else dict(params)
		self.model: Ridge | None = None
		self.scaler: StandardScaler | None = None

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int,
	) -> None:
		valid_train = train_frame.dropna(subset=feature_columns + [target_column])
		if valid_train.empty:
			self.model = None
			self.scaler = None
			return

		self.scaler = StandardScaler()
		train_features = self.scaler.fit_transform(valid_train[feature_columns])
		self.model = Ridge(alpha=float(self.params.get("alpha", 1.0)))
		self.model.fit(train_features, valid_train[target_column])

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
	) -> pd.DataFrame:
		return predict_frame.dropna(subset=feature_columns + [target_column]).copy()

	def select_inference_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
	) -> pd.DataFrame:
		return predict_frame.dropna(subset=feature_columns).copy()

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		if self.model is None or predict_frame.empty:
			return np.asarray([], dtype=float)

		if self.scaler is None:
			raise ValueError()

		predict_features = self.scaler.transform(predict_frame[feature_columns])
		return np.asarray(self.model.predict(predict_features), dtype=float)
