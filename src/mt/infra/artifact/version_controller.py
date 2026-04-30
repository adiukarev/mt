from pathlib import Path


def set_version_existing_artifact(path: str | Path) -> Path | None:
	path = Path(path)
	if not path.exists() or not _has_materialized_artifacts(path):
		return None

	versioned_path = _next_artifact_version_path(path)
	path.rename(versioned_path)

	return versioned_path


def _has_materialized_artifacts(path: Path) -> bool:
	if not path.is_dir():
		return False
	for child in path.rglob("*"):
		if child.is_file():
			return True
	return False


def _next_artifact_version_path(root: Path) -> Path:
	version = 1
	while True:
		candidate = root.parent / f"{root.name}_v{version}"
		if not candidate.exists():
			return candidate
		version += 1
