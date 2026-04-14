import numpy as np
import pandas as pd

from mt.infra.model.adapters.naive import NaiveAdapter


class SeasonalNaiveAdapter(NaiveAdapter):
	"""Сезонный baseline: прогноз равен значению ряда год назад."""

	model_name = "seasonal_naive"
	seasonal_lag = 52

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		"""Вернуть сезонный лаг, а при его отсутствии откатиться к `lag_1`."""

		fallback = predict_frame["lag_1"].astype(float)
		seasonal_column = f"lag_{self.seasonal_lag}"

		if seasonal_column not in predict_frame.columns:
			return fallback.to_numpy()

		return predict_frame[seasonal_column].astype(float).fillna(fallback).to_numpy()
