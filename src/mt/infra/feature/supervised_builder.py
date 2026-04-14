import numpy as np
import pandas as pd

from mt.domain.manifest import FeatureManifest
from .builders.calendar import add_calendar_features
from .builders.categorical import add_categorical_features
from .builders.lags import add_lag_features
from .builders.rolling import add_rolling_features

from .registry import FEATURE_SET_GROUPS, resolve_feature_set_groups_key


def make_supervised_frame(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
	manifest: FeatureManifest,
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

	enabled_groups = (
		FEATURE_SET_GROUPS[resolve_feature_set_groups_key(manifest.feature_set)]
		if manifest.enabled
		else set()
	)

	panel = weekly.merge(segments[["series_id", "segment_label"]], on="series_id", how="left").copy()
	panel = panel.sort_values(["series_id", "week_start"])
	feature_columns: list[str] = []

	if "lags" in enabled_groups:
		panel, lag_columns = add_lag_features(panel, manifest.lags)
		feature_columns.extend(lag_columns)

	if "rolling" in enabled_groups:
		# rolling тут собирает локальный уровень/разброс/чуть тренд
		panel, rolling_columns = add_rolling_features(panel, manifest.rolling_windows, manifest.lags)
		feature_columns.extend(rolling_columns)

	if "calendar" in enabled_groups and manifest.use_calendar:
		# календарь ок, он известен заранее
		panel, calendar_columns = add_calendar_features(panel)
		feature_columns.extend(calendar_columns)

	if "categorical" in enabled_groups and manifest.use_category_encodings:
		# просто статика ряда
		panel, categorical_columns = add_categorical_features(panel)
		feature_columns.extend(categorical_columns)

	if "price_promo" in enabled_groups:
		if manifest.use_price:
			panel["price"] = panel["price"].astype(float)
			feature_columns.append("price")
		if manifest.use_promo:
			panel["promo"] = panel["promo"].fillna(0).astype(float)
			feature_columns.append("promo")

	# после rolling/ratio иногда может приехать мусор типа inf
	numeric_columns = panel.select_dtypes(include=[np.number]).columns
	panel.loc[:, numeric_columns] = panel.loc[:, numeric_columns].replace([np.inf, -np.inf], np.nan)
	panel = panel.sort_values(["series_id", "week_start"]).reset_index(drop=True)
	return panel, feature_columns
