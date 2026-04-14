import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.base import RegressorMixin

from mt.domain.model import ModelInfo, ForecastModelAdapter


class CatBoostAdapter(ForecastModelAdapter):
	"""Глобальный адаптер регрессора CatBoost"""

	def __init__(self, params: dict[str, object] | None = None) -> None:
		super().__init__(ModelInfo(model_name="catboost", model_family="ml"))

		self.params = {} if params is None else dict(params)
		self.model: RegressorMixin | None = None

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int, seed: int) -> None:
		"""Обучить CatBoost на валидных строках окна"""

		valid_train = train_frame.dropna(subset=feature_columns + [target_column])

		if valid_train.empty:
			self.model = None
			return

		self.model = CatBoostRegressor(
			depth=int(self.params.get("depth", 6)),
			iterations=int(self.params.get("iterations", 300)),
			learning_rate=float(self.params.get("learning_rate", 0.05)),
			l2_leaf_reg=float(self.params.get("l2_leaf_reg", 3.0)),
			loss_function=str(self.params.get("loss_function", "MAE")),
			random_seed=seed,
			thread_count=1,
			allow_writing_files=False,
			verbose=False,
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
