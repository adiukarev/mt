from pathlib import Path

TRACKED_SUFFIXES = {
	".csv",
	".json",
	".md",
	".pkl",
	".png",
	".svg",
	".txt",
	".yaml",
	".yml",
}


def build_artifact_inventory(root: str | Path) -> dict[str, tuple[int, int]]:
	"""Построить легковесный снимок файловых артефактов для diff по стадиям"""

	root_path = Path(root)
	if not root_path.exists():
		return {}

	inventory: dict[str, tuple[int, int]] = {}
	for path in iter_tracked_artifact_paths(root_path):
		stat = path.stat()
		inventory[str(path.relative_to(root_path))] = (stat.st_mtime_ns, stat.st_size)

	return inventory


def iter_tracked_artifact_paths(root: str | Path):
	root_path = Path(root)
	if not root_path.exists():
		return
	for path in root_path.rglob("*"):
		if not path.is_file():
			continue
		if "_orchestration" in path.parts:
			continue
		if path.suffix.lower() not in TRACKED_SUFFIXES:
			continue
		yield path


def tracked_artifact_paths(root: str | Path) -> list[Path]:
	return list(iter_tracked_artifact_paths(root))


def changed_artifact_paths(
	root: str | Path,
	before: dict[str, tuple[int, int]],
	after: dict[str, tuple[int, int]],
) -> list[Path]:
	"""Вернуть новые или измененные файлы относительно предыдущего инвентаря."""

	root_path = Path(root)
	return [
		root_path / relative_path
		for relative_path, after_signature in after.items()
		if before.get(relative_path) != after_signature
	]
