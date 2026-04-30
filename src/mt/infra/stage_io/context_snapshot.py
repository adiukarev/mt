from pathlib import Path
from typing import Any

from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.artifact.binary_reader import read_gzip_pickle
from mt.infra.artifact.binary_writer import write_gzip_pickle
from mt.infra.artifact.text_writer import write_json


def save_context_snapshot(
	path: str | Path,
	ctx: BasePipelineContext,
	metadata: dict[str, Any] | None = None,
) -> Path:
	"""Сериализовать контекст пайплайна на диск для межпроцессного stage handoff."""

	target = Path(path)
	write_gzip_pickle(target, ctx)

	if metadata is not None:
		write_json(target.with_suffix(".json"), metadata)

	return target


def load_context_snapshot(path: str | Path) -> BasePipelineContext:
	"""Восстановить контекст пайплайна из stage snapshot."""

	source = Path(path)
	payload = read_gzip_pickle(source)

	if not isinstance(payload, BasePipelineContext):
		raise TypeError(f"Некорректный snapshot type: {type(payload)!r}")

	return payload
