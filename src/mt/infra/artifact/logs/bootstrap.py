import logging

from typing import Any


def log_bootstrap_ci(model_a: Any, model_b: Any, blocks: Any, n_bootstrap: Any) -> None:
	logging.info(
		f"bootstrap_ci: model_a={model_a} model_b={model_b} blocks={len(blocks)} n_bootstrap={n_bootstrap}",
	)


def log_bootstrap_ci_fast(model_a: Any, model_b: Any, blocks: Any, n_bootstrap: Any) -> None:
	logging.info(
		f"bootstrap_ci_fast: model_a={model_a} model_b={model_b} blocks={len(blocks)} n_bootstrap={n_bootstrap}",
	)
