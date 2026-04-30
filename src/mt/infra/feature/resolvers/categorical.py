import pandas as pd


def resolve_categorical_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
	"""Добавить категориальные кодировки для стабильных метаданных"""

	data["category_code"] = data["category"].astype("category").cat.codes
	data["segment_code"] = data["segment_label"].fillna("unknown").astype("category").cat.codes

	feature_columns = ["category_code", "segment_code"]

	if "sku" in data.columns:
		data["sku_code"] = data["sku"].astype("category").cat.codes

		feature_columns.append("sku_code")

	return data, feature_columns
