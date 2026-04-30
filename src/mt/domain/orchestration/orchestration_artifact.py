from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OrchestrationArtifactPathsMap:
	root: Path
	context_snapshots: Path
	stage_states: Path
	manifest_snapshot: Path
	runtime_log: Path
	tracking_snapshot: Path

	@staticmethod
	def ensure(root: str | Path) -> "OrchestrationArtifactPathsMap":
		root_path = Path(root) / "_orchestration"
		paths = OrchestrationArtifactPathsMap(
			root=root_path,
			context_snapshots=root_path / "context_snapshots",
			stage_states=root_path / "stage_states",
			manifest_snapshot=root_path / "manifest_snapshot.yaml",
			runtime_log=root_path / "runtime.log",
			tracking_snapshot=root_path / "tracking_snapshot.yaml",
		)
		for path in (paths.root, paths.context_snapshots, paths.stage_states):
			path.mkdir(parents=True, exist_ok=True)
		return paths
