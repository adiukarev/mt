import pandas as pd


def add_categorical_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить категориальные кодировки для стабильных метаданных"""

	panel["category_code"] = panel["category"].astype("category").cat.codes
	panel["segment_code"] = panel["segment_label"].fillna("unknown").astype("category").cat.codes
	feature_columns = ["category_code", "segment_code"]

	if "sku" in panel.columns:
		panel["sku_code"] = panel["sku"].astype("category").cat.codes
		feature_columns.append("sku_code")

	return panel, feature_columns
