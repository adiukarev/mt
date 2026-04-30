import os
import random
from pathlib import Path

import numpy as np

from mt.infra.observability.logger.runtime_logger import configure_runtime_logging


def ensure_runtime_env() -> None:
	"""Подготовить только полезные env-настройки для текущего runtime"""

	mpl_dir = Path.cwd() / ".matplotlib"
	mpl_dir.mkdir(parents=True, exist_ok=True)
	os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

	os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
	os.environ.setdefault("OMP_NUM_THREADS", "1")
	os.environ.setdefault("MKL_NUM_THREADS", "1")
	os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
	os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
	os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
	os.environ.setdefault("BLIS_NUM_THREADS", "1")
	os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
	os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
	os.environ.setdefault("KMP_USE_SHM", "0")
	os.environ.setdefault("KMP_CREATE_SHM", "0")


def ensure_runtime_logging() -> None:
	"""Настроить корневой логгер один раз на процесс"""

	configure_runtime_logging()


def ensure_runtime_seed_everything(seed: int):
	"""Зафиксировать случайность процесса без подмены логики библиотек"""

	random.seed(seed)
	np.random.seed(seed)
