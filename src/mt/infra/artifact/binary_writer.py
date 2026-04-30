import gzip
import pickle
from pathlib import Path
from typing import Any


def write_pickle(path: str | Path, payload: Any) -> Path:
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	with target.open("wb") as file_obj:
		pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
	return target


def write_gzip_pickle(path: str | Path, payload: Any) -> Path:
	target = Path(path)
	target.parent.mkdir(parents=True, exist_ok=True)
	# Snapshot handoff is latency-sensitive, so prefer fast compression over max ratio.
	with gzip.open(target, "wb", compresslevel=1) as file_obj:
		pickle.dump(payload, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
	return target
