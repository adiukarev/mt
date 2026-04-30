from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from mt.domain.synthetic_generation.synthetic_generation_pipeline_manifest import (
	SyntheticGenerationPipelineManifest,
)
from mt.infra.artifact.text_writer import write_csv, write_yaml
from mt.infra.artifact.text_reader import read_yaml_mapping
from mt.infra.synthetic.generator import build_series_metadata, generate_dataset_frame


@dataclass(slots=True)
class SyntheticMaterializationResult:
	output_root: Path
	full_dataset: pd.DataFrame
	history_frame: pd.DataFrame
	recent_frame: pd.DataFrame
	series_metadata: pd.DataFrame
	materialized_paths: list[str]

	@property
	def summary(self) -> dict[str, object]:
		return {
			"dataset_root": str(self.output_root),
			"row_count": int(len(self.full_dataset)),
			"history_rows": int(len(self.history_frame)),
			"recent_rows": int(len(self.recent_frame)),
			"series_count": int(self.full_dataset["series_id"].nunique()),
			"scenario_count": int(self.full_dataset["scenario_name"].nunique()),
			"batch_count": int(self.recent_frame["week_start"].nunique()),
		}


def materialize_synthetic_dataset(
	manifest: SyntheticGenerationPipelineManifest,
	output_root: str | Path,
) -> SyntheticMaterializationResult:
	root = Path(output_root)
	generation_id = _resolve_generation_id(root)
	resolved_manifest = _select_generation_scenario(manifest, generation_id)
	full_dataset = generate_dataset_frame(resolved_manifest)
	full_dataset["week_start"] = pd.to_datetime(full_dataset["week_start"])

	history_frame = full_dataset.loc[full_dataset["is_history"].astype(bool)].copy()
	recent_frame = full_dataset.loc[~full_dataset["is_history"].astype(bool)].copy()

	root.mkdir(parents=True, exist_ok=True)
	batches_dir = root / "batches"
	batches_dir.mkdir(parents=True, exist_ok=True)

	if manifest.output.write_mode == "replace" or not (root / "dataset.csv").exists():
		_cleanup_existing_batches(batches_dir)
		write_csv(root / "dataset.csv", history_frame.reset_index(drop=True))
		materialized_history = history_frame.copy()
		next_batch_index = 1
	else:
		materialized_history = pd.read_csv(root / "dataset.csv", parse_dates=["week_start"])
		next_batch_index = _next_batch_index(batches_dir)

	recent_frame = _align_recent_frame_to_existing_stream(
		recent_frame=recent_frame,
		root=root,
		batches_dir=batches_dir,
		write_mode=manifest.output.write_mode,
	)
	materialized_full_dataset = pd.concat(
		[materialized_history, recent_frame],
		ignore_index=True,
	)
	series_metadata = build_series_metadata(materialized_full_dataset)

	write_yaml(
		root / "source_manifest_snapshot.yaml",
		{
			"synthetic_manifest": resolved_manifest.to_dict(),
			"generation_id": generation_id,
			"write_mode": manifest.output.write_mode,
			"selected_scenario": resolved_manifest.scenarios[0].name,
			"history_rows": int(len(materialized_history)),
			"recent_rows": int(len(recent_frame)),
			"recent_week_count": int(recent_frame["week_start"].nunique()),
			"series_count": int(materialized_full_dataset["series_id"].nunique()),
			"scenario_count": int(materialized_full_dataset["scenario_name"].nunique()),
		},
	)
	write_yaml(
		root / "generation_state.yaml",
		{
			"generation_id": generation_id,
			"next_generation_id": generation_id + 1,
			"last_selected_scenario": resolved_manifest.scenarios[0].name,
			"write_mode": manifest.output.write_mode,
		},
	)
	write_csv(root / "series_metadata.csv", series_metadata.reset_index(drop=True))

	batch_paths: list[Path] = []
	for index, (week_start, frame) in enumerate(
		recent_frame.groupby("week_start", sort=True),
		start=next_batch_index,
	):
		batch_path = batches_dir / f"{index:03d}_{pd.Timestamp(week_start).date()}.csv"
		write_csv(
			batch_path,
			frame.sort_values(["series_id", "week_start"]).reset_index(drop=True),
		)
		batch_paths.append(batch_path)

	return SyntheticMaterializationResult(
		output_root=root,
		full_dataset=materialized_full_dataset.reset_index(drop=True),
		history_frame=materialized_history.reset_index(drop=True),
		recent_frame=recent_frame.reset_index(drop=True),
		series_metadata=series_metadata.reset_index(drop=True),
		materialized_paths=[
			str(root / "dataset.csv"),
			str(root / "series_metadata.csv"),
			str(root / "source_manifest_snapshot.yaml"),
			str(root / "generation_state.yaml"),
			*[str(path) for path in batch_paths],
		],
	)


def _resolve_generation_id(root: Path) -> int:
	state_path = root / "generation_state.yaml"
	if not state_path.exists():
		return 0
	state = read_yaml_mapping(state_path)
	return int(state.get("next_generation_id", 0))


def _select_generation_scenario(
	manifest: SyntheticGenerationPipelineManifest,
	generation_id: int,
) -> SyntheticGenerationPipelineManifest:
	if manifest.output.scenario_policy == "first":
		return replace(manifest, scenarios=[manifest.scenarios[0]])

	cycle = manifest.output.scenario_cycle or [scenario.name for scenario in manifest.scenarios]
	selected_name = cycle[generation_id % len(cycle)]
	for scenario in manifest.scenarios:
		if scenario.name == selected_name:
			return replace(manifest, scenarios=[scenario])
	raise ValueError(f"Unknown synthetic scenario in scenario_cycle: {selected_name}")


def _cleanup_existing_batches(batches_dir: Path) -> None:
	for path in batches_dir.glob("*.csv"):
		path.unlink()


def _next_batch_index(batches_dir: Path) -> int:
	indices: list[int] = []
	for path in batches_dir.glob("*.csv"):
		prefix = path.name.split("_", 1)[0]
		if prefix.isdigit():
			indices.append(int(prefix))
	return max(indices, default=0) + 1


def _align_recent_frame_to_existing_stream(
	recent_frame: pd.DataFrame,
	root: Path,
	batches_dir: Path,
	write_mode: str,
) -> pd.DataFrame:
	if write_mode == "replace":
		return recent_frame.copy()

	latest_week = _latest_materialized_week(root, batches_dir)
	if latest_week is None:
		return recent_frame.copy()

	unique_weeks = sorted(pd.to_datetime(recent_frame["week_start"]).unique())
	week_mapping = {
		week: latest_week + pd.Timedelta(weeks=index)
		for index, week in enumerate(unique_weeks, start=1)
	}
	aligned = recent_frame.copy()
	aligned["week_start"] = aligned["week_start"].map(week_mapping)
	if "time_index" in aligned.columns:
		min_time_index = int(aligned["time_index"].min())
		latest_time_index = _latest_materialized_time_index(root, batches_dir)
		aligned["time_index"] = aligned["time_index"] - min_time_index + latest_time_index + 1
	return aligned


def _latest_materialized_week(root: Path, batches_dir: Path) -> pd.Timestamp | None:
	frames: list[pd.DataFrame] = []
	dataset_path = root / "dataset.csv"
	if dataset_path.exists():
		frames.append(pd.read_csv(dataset_path, parse_dates=["week_start"], usecols=["week_start"]))
	for path in sorted(batches_dir.glob("*.csv")):
		frames.append(pd.read_csv(path, parse_dates=["week_start"], usecols=["week_start"]))
	if not frames:
		return None
	return pd.Timestamp(pd.concat(frames, ignore_index=True)["week_start"].max())


def _latest_materialized_time_index(root: Path, batches_dir: Path) -> int:
	frames: list[pd.DataFrame] = []
	dataset_path = root / "dataset.csv"
	if dataset_path.exists():
		frames.append(pd.read_csv(dataset_path, usecols=["time_index"]))
	for path in sorted(batches_dir.glob("*.csv")):
		frames.append(pd.read_csv(path, usecols=["time_index"]))
	if not frames:
		return -1
	return int(pd.concat(frames, ignore_index=True)["time_index"].max())
