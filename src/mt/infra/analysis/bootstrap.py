import numpy as np
import pandas as pd

from mt.domain.metric.metric_point import calculate_point_metrics
from mt.infra.observability.runtime.stage_events import log_bootstrap_ci


def build_bootstrap_ci(
	predictions: pd.DataFrame,
	metric_name: str = "WAPE",
	n_bootstrap: int = 300,
	seed: int = 42,
) -> pd.DataFrame:
	"""Оценить bootstrap-доверительный интервал для разности метрик моделей.

	Ресэмплинг идет блоками `series_id x forecast_origin x horizon`, чтобы не
	разрывать связь между наблюдениями внутри одного оценочного блока rolling
	backtesting. Это дает более устойчивую оценку неопределенности, чем сравнение
	по одной агрегированной точке качества.
	"""

	if predictions.empty:
		return pd.DataFrame()

	block_columns = ["series_id", "forecast_origin", "horizon"]
	available_columns = [column for column in block_columns if column in predictions.columns]
	if not available_columns:
		return pd.DataFrame()

	if metric_name == "WAPE":
		return _build_wape_bootstrap_ci_fast(
			predictions=predictions,
			available_columns=available_columns,
			n_bootstrap=n_bootstrap,
			seed=seed,
		)

	models = sorted(predictions["model_name"].unique().tolist())
	blocks = predictions[available_columns].drop_duplicates().reset_index(drop=True)
	rng = np.random.default_rng(seed)
	rows: list[dict[str, object]] = []
	for idx, model_a in enumerate(models):
		for model_b in models[idx + 1:]:
			log_bootstrap_ci(
				model_a,
				model_b,
				blocks,
				n_bootstrap,
			)
			diffs: list[float] = []
			for _ in range(n_bootstrap):
				sampled_blocks = blocks.sample(n=len(blocks), replace=True,
				                               random_state=int(rng.integers(0, 2 ** 31 - 1)))
				sampled = sampled_blocks.merge(predictions, on=available_columns, how="left")
				metrics_a = calculate_point_metrics(sampled[sampled["model_name"] == model_a])
				metrics_b = calculate_point_metrics(sampled[sampled["model_name"] == model_b])
				diffs.append(metrics_a[metric_name] - metrics_b[metric_name])
			rows.append(
				{
					"model_a": model_a,
					"model_b": model_b,
					"metric": metric_name,
					"mean_difference": float(np.mean(diffs)),
					"median_difference": float(np.median(diffs)),
					"ci95_low": float(np.quantile(diffs, 0.025)),
					"ci95_high": float(np.quantile(diffs, 0.975)),
					"n_bootstrap": n_bootstrap,
					"bootstrap_unit": "series_id x forecast_origin x horizon",
				}
			)

	return pd.DataFrame(rows).sort_values(["model_a", "model_b"]).reset_index(drop=True)


def _build_wape_bootstrap_ci_fast(
	predictions: pd.DataFrame,
	available_columns: list[str],
	n_bootstrap: int,
	seed: int,
) -> pd.DataFrame:
	"""Быстрый bootstrap для WAPE через предагрегированные блоки."""

	models = sorted(predictions["model_name"].unique().tolist())
	working = predictions.loc[:, [*available_columns, "model_name", "actual", "prediction"]].copy()
	working["abs_error"] = (
		working["prediction"].astype(float) - working["actual"].astype(float)).abs()
	working["abs_actual"] = working["actual"].astype(float).abs()
	aggregated = (
		working.groupby([*available_columns, "model_name"], as_index=False)
		.agg(
			abs_error_sum=("abs_error", "sum"),
			abs_actual_sum=("abs_actual", "sum"),
		)
	)
	block_keys = aggregated[available_columns].drop_duplicates().reset_index(drop=True)
	block_keys["block_id"] = np.arange(len(block_keys), dtype=int)
	aggregated = aggregated.merge(block_keys, on=available_columns, how="left")

	error_matrix = np.zeros((len(models), len(block_keys)), dtype=float)
	actual_matrix = np.zeros((len(models), len(block_keys)), dtype=float)
	model_to_idx = {model_name: idx for idx, model_name in enumerate(models)}
	for row in aggregated.itertuples(index=False):
		model_idx = model_to_idx[str(row.model_name)]
		block_idx = int(row.block_id)
		error_matrix[model_idx, block_idx] = float(row.abs_error_sum)
		actual_matrix[model_idx, block_idx] = float(row.abs_actual_sum)

	rng = np.random.default_rng(seed)
	sampled_block_ids = rng.integers(0, len(block_keys), size=(n_bootstrap, len(block_keys)))
	rows: list[dict[str, object]] = []
	for idx, model_a in enumerate(models):
		for model_b in models[idx + 1:]:
			log_bootstrap_ci(
				model_a,
				model_b,
				block_keys,
				n_bootstrap,
			)
			model_a_idx = model_to_idx[model_a]
			model_b_idx = model_to_idx[model_b]
			error_a = error_matrix[model_a_idx][sampled_block_ids].sum(axis=1)
			error_b = error_matrix[model_b_idx][sampled_block_ids].sum(axis=1)
			denom_a = actual_matrix[model_a_idx][sampled_block_ids].sum(axis=1)
			denom_b = actual_matrix[model_b_idx][sampled_block_ids].sum(axis=1)
			wape_a = np.divide(error_a, denom_a, out=np.full_like(error_a, np.nan), where=denom_a != 0.0)
			wape_b = np.divide(error_b, denom_b, out=np.full_like(error_b, np.nan), where=denom_b != 0.0)
			diffs = wape_a - wape_b

			rows.append(
				{
					"model_a": model_a,
					"model_b": model_b,
					"metric": "WAPE",
					"mean_difference": float(np.nanmean(diffs)),
					"median_difference": float(np.nanmedian(diffs)),
					"ci95_low": float(np.nanquantile(diffs, 0.025)),
					"ci95_high": float(np.nanquantile(diffs, 0.975)),
					"n_bootstrap": n_bootstrap,
					"bootstrap_unit": "series_id x forecast_origin x horizon",
				}
			)

	return pd.DataFrame(rows).sort_values(["model_a", "model_b"]).reset_index(drop=True)
