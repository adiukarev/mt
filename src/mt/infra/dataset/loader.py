from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from zipfile import ZipFile

from mt.domain.dataset.dataset import DatasetLoadData
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.dataset.dataset_registry import resolve_dataset_registry_entry
from mt.infra.artifact.downloader import download_google_drive_file, download_url_to_file
from mt.infra.dataset.factory import build_dataset_adapter


def load_dataset(manifest: DatasetManifest) -> DatasetLoadData:
	return build_dataset_adapter(_resolve_dataset_manifest(manifest)).load()


def _resolve_dataset_manifest(manifest: DatasetManifest) -> DatasetManifest:
	spec = resolve_dataset_registry_entry(manifest.kind)

	resolved_path = _resolve_local_dataset_path(
		requested_path=Path(manifest.path),
		default_extract_dir=spec.extract_dir,
		archive_path=spec.archive_path,
		required_files=spec.required_files,
		download_url=spec.download_url,
	)

	return replace(manifest, path=str(resolved_path))


def _resolve_local_dataset_path(
	*,
	requested_path: Path,
	default_extract_dir: Path,
	archive_path: Path,
	required_files: tuple[str, ...],
	download_url: str | None,
) -> Path:
	candidates: list[Path] = []
	for candidate in (requested_path, default_extract_dir):
		if candidate not in candidates:
			candidates.append(candidate)

	for candidate in candidates:
		if _has_required_files(candidate, required_files):
			return candidate

	for candidate in candidates:
		candidate_archive = _archive_candidate_for_path(candidate, archive_path)
		if candidate_archive.exists():
			return _extract_archive(candidate_archive, candidate, required_files)

	if archive_path.exists():
		return _extract_archive(archive_path, default_extract_dir, required_files)

	if isinstance(download_url, str) and download_url.strip():
		archive_path.parent.mkdir(parents=True, exist_ok=True)
		_download_file(download_url, archive_path)
		return _extract_archive(archive_path, default_extract_dir, required_files)

	raise RuntimeError()


def _archive_candidate_for_path(path: Path, fallback_archive: Path) -> Path:
	if path.suffix.lower() == ".zip":
		return path
	return path.with_suffix(".zip") if path.name else fallback_archive


def _extract_archive(archive_path: Path, target_dir: Path, required_files: tuple[str, ...]) -> Path:
	target_dir.parent.mkdir(parents=True, exist_ok=True)
	with ZipFile(archive_path) as archive:
		target_dir.mkdir(parents=True, exist_ok=True)
		archive.extractall(target_dir)

	nested_root = _find_nested_root(target_dir, required_files)
	if nested_root is not None:
		return nested_root
	if _has_required_files(target_dir, required_files):
		return target_dir

	raise RuntimeError()


def _find_nested_root(target_dir: Path, required_files: tuple[str, ...]) -> Path | None:
	children = [child for child in target_dir.iterdir() if child.name != "__MACOSX"]
	if len(children) != 1 or not children[0].is_dir():
		return None
	nested_root = children[0]
	return nested_root if _has_required_files(nested_root, required_files) else None


def _download_file(download_url: str, destination: Path) -> None:
	if _is_google_drive_url(download_url):
		_download_google_drive_file(download_url, destination)
		return

	download_url_to_file(download_url, destination)


def _download_google_drive_file(download_url: str, destination: Path) -> None:
	file_id = _extract_google_drive_file_id(download_url)
	if file_id is None:
		raise RuntimeError()
	download_google_drive_file(file_id, destination)


def _is_google_drive_url(download_url: str) -> bool:
	netloc = urlparse(download_url).netloc.lower()
	return "drive.google.com" in netloc


def _extract_google_drive_file_id(download_url: str) -> str | None:
	parsed = urlparse(download_url)
	path_parts = [part for part in parsed.path.split("/") if part]
	if "d" in path_parts:
		index = path_parts.index("d")
		if index + 1 < len(path_parts):
			return path_parts[index + 1]

	query_id = parse_qs(parsed.query).get("id")
	if query_id:
		return query_id[0]
	return None


def _has_required_files(path: Path, required_files: tuple[str, ...]) -> bool:
	return path.is_dir() and all((path / name).exists() for name in required_files)
