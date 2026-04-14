from pathlib import Path

import pandas as pd

from mt.domain.dataset import DatasetLoadData
from mt.domain.manifest import DatasetManifest


def load_dataset(manifest: DatasetManifest) -> DatasetLoadData:
	"""Прочитать исходные таблицы M5 с диска"""

	root = Path(manifest.path)

	sales = pd.read_csv(root / "sales_train_evaluation.csv")
	calendar = pd.read_csv(root / "calendar.csv", usecols=["d", "date", "wm_yr_wk"])
	sell_prices = pd.read_csv(root / "sell_prices.csv")

	return DatasetLoadData(sales=sales, calendar=calendar, sell_prices=sell_prices)
