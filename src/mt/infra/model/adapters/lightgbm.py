import numpy as np
import pandas as pd

from mt.domain.model import ModelInfo, ForecastModelAdapter
from lightgbm import LGBMRegressor
from sklearn.base import RegressorMixin


class LightGBMAdapter(ForecastModelAdapter):
	"""Глобальный адаптер регрессора LightGBM"""

	def __init__(self, params: dict[str, object] | None = None) -> None:
		super().__init__(ModelInfo(model_name="lightgbm", model_family="ml"))

		self.params = {} if params is None else dict(params)
		self.model: RegressorMixin | None = None

	def fit(
		self,
		train_frame:
		pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int
	) -> None:
		valid_train = train_frame.dropna(subset=feature_columns + [target_column])

		if valid_train.empty:
			self.model = None
			return

		self.model = LGBMRegressor(
			n_estimators=int(self.params.get("n_estimators", 300)),
			learning_rate=float(self.params.get("learning_rate", 0.05)),
			max_depth=int(self.params.get("max_depth", 6)),
			num_leaves=int(self.params.get("num_leaves", 31)),
			min_child_samples=int(self.params.get("min_child_samples", 20)),
			subsample=float(self.params.get("subsample", 0.9)),
			colsample_bytree=float(self.params.get("colsample_bytree", 0.9)),
			random_state=seed,
			n_jobs=1,
			force_row_wise=True,
			verbosity=-1,
		)

		self.model.fit(valid_train[feature_columns], valid_train[target_column])

	def select_predict_frame(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str
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
		horizon: int
	) -> np.ndarray:
		if self.model is None or predict_frame.empty:
			return np.asarray([], dtype=float)

		return np.asarray(self.model.predict(predict_frame[feature_columns]), dtype=float)
