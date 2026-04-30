from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BaseArtifactPathsMap:
	root: Path
	report: Path
	run: Path

	@staticmethod
	def _paths(root: Path) -> dict[str, Path]:
		return {
			"root": root,
			"report": root / "report",
			"run": root / "run",
		}

	@staticmethod
	def _ensure_dirs(*paths: Path) -> None:
		for path in paths:
			path.mkdir(parents=True, exist_ok=True)

	def run_file(self, filename: str) -> Path:
		return self.run / filename

	def report_file(self, filename: str) -> Path:
		return self.report / filename
