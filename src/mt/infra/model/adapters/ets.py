import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from mt.domain.model import ModelInfo, ForecastModelAdapter


class ETSAdapter(ForecastModelAdapter):
	"""Поканальная эталонная ETS-модель"""

	def __init__(self, params: dict[str, object] | None = None) -> None:
		super().__init__(ModelInfo(model_name="ets", model_family="statistical"))

		self.params = {} if params is None else dict(params)
		self.history: pd.DataFrame | None = None

	def fit(
		self,
		train_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
		seed: int,
	) -> None:
		self.history = train_frame[["series_id", "week_start", "sales_units"]].copy()

	def predict(
		self,
		predict_frame: pd.DataFrame,
		feature_columns: list[str],
		target_column: str,
		horizon: int,
	) -> np.ndarray:
		assert self.history is not None

		outputs: list[float] = []
		history = self.history

		for row in predict_frame.itertuples(index=False):
			series_history = (
				history[history["series_id"] == row.series_id]
				.sort_values("week_start")
				.set_index("week_start")["sales_units"]
				.astype(float)
			)
			series_history.index = pd.DatetimeIndex(series_history.index, freq="W-MON")

			if len(series_history) < 8:
				# Для коротких рядов ETS нестабилен, поэтому используем честный last-value fallback
				# из самой истории ряда, а не зависим от внешних табличных признаков.
				outputs.append(float(series_history.iloc[-1]))
				continue

			model = ExponentialSmoothing(
				series_history,
				trend=self.params.get("trend", "add"),
				seasonal=self.params.get("seasonal"),
				seasonal_periods=self.params.get("seasonal_periods"),
				initialization_method=str(self.params.get("initialization_method", "estimated")),
			).fit()

			outputs.append(float(model.forecast(horizon).iloc[-1]))

		return np.asarray(outputs, dtype=float)
