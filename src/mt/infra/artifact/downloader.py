from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def download_url_to_file(
	url: str,
	destination: str | Path,
	chunk_size: int = 1024 * 1024,
) -> Path:
	target = Path(destination)
	target.parent.mkdir(parents=True, exist_ok=True)

	with urlopen(url) as response, NamedTemporaryFile(
		dir=target.parent,
		delete=False,
		suffix=".tmp",
	) as temp_file:
		while True:
			chunk = response.read(chunk_size)
			if not chunk:
				break
			temp_file.write(chunk)

	temp_path = Path(temp_file.name)
	temp_path.replace(target)
	return target


def download_google_drive_file(file_id: str, destination: str | Path) -> Path:
	try:
		import gdown
	except ImportError as exc:
		raise RuntimeError("gdown is required for Google Drive downloads") from exc

	target = Path(destination)
	target.parent.mkdir(parents=True, exist_ok=True)
	result = gdown.download(id=file_id, output=str(target), quiet=False)
	if result is None or not target.exists():
		raise RuntimeError(f"Failed to download Google Drive file {file_id}")
	return target
