import numpy as np
import pandas as pd


def calculate_metrics(frame: pd.DataFrame) -> dict[str, float]:
	"""Посчитать полный набор метрик по таблице прогнозов"""

	actual = frame["actual"].astype(float).to_numpy()
	predicted = frame["prediction"].astype(float).to_numpy()

	return {
		"WAPE": calculate_wape(actual, predicted),
		"sMAPE": calculate_smape(actual, predicted),
		"MAE": calculate_mae(actual, predicted),
		"RMSE": calculate_rmse(actual, predicted),
		"Bias": calculate_bias(actual, predicted),
		"MedianAE": calculate_median_ae(actual, predicted),
	}


def calculate_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
	"""Посчитать вектор ошибок прогноза"""

	return predicted - actual


def calculate_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать WAPE"""

	error = calculate_error(actual, predicted)
	denominator = float(np.abs(actual).sum())
	if denominator == 0.0:
		return float("nan")
	return float(np.abs(error).sum() / denominator)


def calculate_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать symmetric MAPE"""

	error = calculate_error(actual, predicted)
	denominator = np.abs(actual) + np.abs(predicted)
	if len(error) == 0:
		return float("nan")
	return float(np.mean(200.0 * np.abs(error) / np.where(denominator == 0.0, 1.0, denominator)))


def calculate_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать MAE"""

	error = calculate_error(actual, predicted)
	if len(error) == 0:
		return float("nan")
	return float(np.mean(np.abs(error)))


def calculate_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать RMSE"""

	error = calculate_error(actual, predicted)
	if len(error) == 0:
		return float("nan")
	return float(np.sqrt(np.mean(np.square(error))))


def calculate_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать bias прогноза"""

	error = calculate_error(actual, predicted)
	if len(error) == 0:
		return float("nan")
	return float(np.mean(error))


def calculate_median_ae(actual: np.ndarray, predicted: np.ndarray) -> float:
	"""Посчитать медиану абсолютной ошибки"""

	error = calculate_error(actual, predicted)
	if len(error) == 0:
		return float("nan")
	return float(np.median(np.abs(error)))
