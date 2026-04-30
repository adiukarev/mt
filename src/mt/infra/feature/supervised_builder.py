import numpy as np
import pandas as pd

from mt.domain.backtest.backtest_manifest import BacktestManifest
from mt.domain.feature.feature_manifest import FeatureManifest
from mt.domain.feature.feature_set import FeatureGroup, get_feature_groups
from mt.infra.feature.resolvers.calendar import resolve_calendar_features
from mt.infra.feature.resolvers.categorical import resolve_categorical_features
from mt.infra.feature.resolvers.lags import resolve_lag_features
from mt.infra.feature.resolvers.rolling import resolve_rolling_features


def build_supervised_frame(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
	feature_manifest: FeatureManifest,
	backtest_manifest: BacktestManifest | None = None,
) -> tuple[pd.DataFrame, list[str]]:
	"""
	Построить supervised-таблицу для прямого прогноза.

	Идея supervised-таблицы:
	- одна строка описывает один ряд `series_id` в одну неделю `week_start`;
	- в строке лежат только признаки, которые известны на эту дату;
	- target_h1, target_h2, ... добавляются отдельно и означают продажи через 1, 2, ... недель.

	Именно такую таблицу удобно отдавать ML/DL-моделям,
	которые учатся отображению `features_t -> y_{t+h}`.

	В experiment pipeline эта таблица строится как общий leakage-safe superset
	для всех включенных моделей. Дальше каждая модель получает только свое
	разрешенное подмножество колонок, а не отдельную физическую таблицу.
	"""

	manifest = backtest_manifest or BacktestManifest()
	enabled_groups = (
		get_feature_groups(feature_manifest.feature_set)
		if feature_manifest.enabled
		else frozenset()
	)

	data = weekly.merge(segments[["series_id", "segment_label"]], on="series_id", how="left").copy()
	data = data.sort_values(["series_id", "week_start"])
	feature_columns: list[str] = []

	if FeatureGroup.LAGS in enabled_groups:
		data, lag_columns = resolve_lag_features(data, feature_manifest.lags)
		feature_columns.extend(lag_columns)

	if FeatureGroup.ROLLING in enabled_groups:
		# rolling тут собирает локальный уровень/разброс/чуть тренд
		data, rolling_columns = resolve_rolling_features(
			data,
			feature_manifest.rolling_windows,
		)
		feature_columns.extend(rolling_columns)

	if FeatureGroup.CALENDAR in enabled_groups and feature_manifest.use_calendar:
		# календарь ок, он известен заранее
		data, calendar_columns = resolve_calendar_features(data)
		feature_columns.extend(calendar_columns)

	if FeatureGroup.CATEGORICAL in enabled_groups and feature_manifest.use_category_encodings:
		# просто статика ряда
		data, categorical_columns = resolve_categorical_features(data)
		feature_columns.extend(categorical_columns)

	# после rolling/ratio иногда может приехать мусор типа inf
	numeric_columns = data.select_dtypes(include=[np.number]).columns
	data.loc[:, numeric_columns] = data.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
	data = data.sort_values(["series_id", "week_start"]).reset_index(drop=True)

	for horizon in range(manifest.horizon_start, manifest.horizon_end + 1):
		data[f"target_h{horizon}"] = (
			data.groupby("series_id")["sales_units"].shift(-horizon)
		)

	return data, feature_columns
