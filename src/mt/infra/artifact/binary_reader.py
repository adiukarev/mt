import gzip
import pickle
from pathlib import Path
from typing import Any


def read_pickle(path: str | Path) -> Any:
	with Path(path).open("rb") as file_obj:
		return pickle.load(file_obj)


def read_gzip_pickle(path: str | Path) -> Any:
	with gzip.open(Path(path), "rb") as file_obj:
		return pickle.load(file_obj)
