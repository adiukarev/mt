from __future__ import annotations

from datetime import datetime
from pathlib import Path


def archive_existing_artifacts(root_dir: str | Path) -> Path | None:
	"""Перенести непустую директорию артефактов в версионированный архив"""

	root = Path(root_dir)
	if not root.exists() or not any(root.iterdir()):
		return None

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
	archive_root = root.parent / f"{root.name}__{timestamp}"
	root.rename(archive_root)
	return archive_root
