import numpy as np
import pandas as pd

from mt.infra.model.adapters.naive import NaiveAdapter
from mt.domain.model.model_name import ModelName


class SeasonalNaiveAdapter(NaiveAdapter):
	"""Сезонный baseline: прогноз равен значению ряда год назад"""

	model_name = ModelName.SEASONAL_NAIVE
	seasonal_lag = 52
	history: pd.DataFrame | None = None

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int
	) -> None:
		self.history = train_frame[["series_id", "week_start", "sales_units"]].copy()

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		"""Вернуть значение за сезон до target-date, а при его отсутствии откатиться к `lag_1`"""

		fallback = predict_frame["lag_1"].astype(float)
		if self.history is None:
			return fallback.to_numpy()

		history = self.history.copy()
		history["week_start"] = pd.to_datetime(history["week_start"])
		lookup = {
			(row.series_id, pd.Timestamp(row.week_start)): float(row.sales_units)
			for row in history.itertuples(index=False)
		}

		outputs: list[float] = []
		for row_idx, row in enumerate(predict_frame.itertuples(index=False)):
			target_date = pd.Timestamp(row.week_start) + pd.Timedelta(weeks=horizon)
			seasonal_date = target_date - pd.Timedelta(weeks=self.seasonal_lag)
			outputs.append(
				lookup.get((row.series_id, seasonal_date), float(fallback.iloc[row_idx]))
			)

		return np.asarray(outputs, dtype=float)
