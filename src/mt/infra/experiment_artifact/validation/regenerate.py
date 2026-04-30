import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mt.domain.experiment.experiment_pipeline_manifest import ExperimentPipelineManifest
from mt.infra.backtest.windows_builder import build_backtest_windows
from mt.infra.dataset.loader import load_dataset
from mt.infra.dataset.preparation import prepare_dataset
from mt.infra.experiment_artifact.validation.plot import save_rolling_backtest_schematic


def regenerate_rolling_backtest_schematic(artifact_root: str | Path) -> Path:
	root = Path(artifact_root)
	backtest_dir = root / "backtest"
	windows_path = backtest_dir / "backtest_windows.csv"
	if not windows_path.exists():
		raise FileNotFoundError(windows_path)

	windows = pd.read_csv(windows_path)
	holdout_tail_weeks = _resolve_holdout_tail_weeks(root, windows)
	history_end = _resolve_history_end(root, windows)
	output_path = backtest_dir / "rolling_backtest_schematic.png"
	save_rolling_backtest_schematic(
		windows=windows,
		output_path=output_path,
		holdout_tail_weeks=holdout_tail_weeks,
		history_end=history_end,
	)
	return output_path


def regenerate_backtest_windows_and_schematic(artifact_root: str | Path) -> Path:
	root = Path(artifact_root)
	manifest_path = root / "run" / "manifest_snapshot.yaml"
	if not manifest_path.exists():
		raise FileNotFoundError(manifest_path)

	manifest = ExperimentPipelineManifest.load(manifest_path)
	dataset = prepare_dataset(manifest.dataset, load_dataset(manifest.dataset))
	windows = pd.DataFrame(asdict(window) for window in build_backtest_windows(
		manifest.backtest,
		dataset.weekly,
	))
	windows = _enrich_windows(windows)

	backtest_dir = root / "backtest"
	backtest_dir.mkdir(parents=True, exist_ok=True)
	windows.to_csv(backtest_dir / "backtest_windows.csv", index=False)
	_write_windows_by_horizon(backtest_dir / "backtest_windows_by_horizon.csv", windows)
	_write_window_summary(
		backtest_dir / "backtest_window_summary.csv",
		manifest=manifest,
		windows=windows,
		history_end=pd.Timestamp(dataset.weekly["week_start"].max()),
	)
	return regenerate_rolling_backtest_schematic(root)


def regenerate_many_rolling_backtest_schematics(root: str | Path) -> list[Path]:
	paths: list[Path] = []
	for windows_path in sorted(Path(root).glob("**/backtest/backtest_windows.csv")):
		paths.append(regenerate_rolling_backtest_schematic(windows_path.parents[1]))
	return paths


def main() -> None:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"artifact_roots",
		nargs="+",
		help="Experiment artifact root(s), or a parent directory with --recursive.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Find and regenerate every **/backtest/backtest_windows.csv under each root.",
	)
	parser.add_argument(
		"--windows",
		action="store_true",
		help="Rebuild backtest window CSVs from run/manifest_snapshot.yaml before redrawing.",
	)
	args = parser.parse_args()

	outputs: list[Path] = []
	for artifact_root in args.artifact_roots:
		if args.windows and args.recursive:
			for manifest_path in sorted(Path(artifact_root).glob("**/run/manifest_snapshot.yaml")):
				if _is_experiment_manifest_snapshot(manifest_path):
					outputs.append(regenerate_backtest_windows_and_schematic(manifest_path.parents[1]))
		elif args.windows:
			outputs.append(regenerate_backtest_windows_and_schematic(artifact_root))
		elif args.recursive:
			outputs.extend(regenerate_many_rolling_backtest_schematics(artifact_root))
		else:
			outputs.append(regenerate_rolling_backtest_schematic(artifact_root))

	for output_path in outputs:
		print(output_path)


def _resolve_holdout_tail_weeks(root: Path, windows: pd.DataFrame) -> int:
	manifest = _load_manifest_snapshot(root)
	backtest = _as_dict(manifest.get("backtest"))
	horizon_end = int(backtest.get("horizon_end", windows["horizon"].max()))
	holdout_tail_weeks = backtest.get("holdout_tail_weeks")
	if holdout_tail_weeks is None:
		final_training = _as_dict(manifest.get("final_training"))
		holdout_tail_weeks = final_training.get("holdout_tail_weeks", "auto")
	if holdout_tail_weeks == "auto":
		return horizon_end
	if holdout_tail_weeks is None:
		return 0
	return int(holdout_tail_weeks)


def _resolve_history_end(root: Path, windows: pd.DataFrame) -> pd.Timestamp:
	summary_path = root / "backtest" / "backtest_window_summary.csv"
	if summary_path.exists():
		summary = pd.read_csv(summary_path)
		if "history_observed_end" in summary.columns:
			return pd.Timestamp(summary.loc[0, "history_observed_end"])
		if "target_week_end" in summary.columns:
			return pd.Timestamp(summary.loc[0, "target_week_end"])
	return pd.Timestamp(pd.to_datetime(windows["test_start"]).max())


def _enrich_windows(windows: pd.DataFrame) -> pd.DataFrame:
	if windows.empty:
		return windows
	result = windows.copy()
	for column in ("forecast_origin", "train_start", "train_end", "test_start", "test_end"):
		result[column] = pd.to_datetime(result[column])
	result["target_week"] = result["test_start"]
	result["train_weeks"] = ((result["train_end"] - result["train_start"]).dt.days // 7) + 1
	result["test_weeks"] = ((result["test_end"] - result["test_start"]).dt.days // 7) + 1
	result["gap_weeks"] = ((result["target_week"] - result["train_end"]).dt.days // 7)
	return result


def _write_windows_by_horizon(path: Path, windows: pd.DataFrame) -> None:
	windows_by_horizon = (
		windows.groupby("horizon", as_index=False)
		.agg(
			window_count=("horizon", "count"),
			forecast_origin_count=("forecast_origin", "nunique"),
			first_forecast_origin=("forecast_origin", "min"),
			last_forecast_origin=("forecast_origin", "max"),
			first_target_week=("target_week", "min"),
			last_target_week=("target_week", "max"),
			min_train_weeks=("train_weeks", "min"),
			max_train_weeks=("train_weeks", "max"),
			gap_weeks=("gap_weeks", "max"),
		)
	)
	windows_by_horizon.to_csv(path, index=False)


def _write_window_summary(
	path: Path,
	manifest: ExperimentPipelineManifest,
	windows: pd.DataFrame,
	history_end: pd.Timestamp,
) -> None:
	holdout_tail_weeks = manifest.backtest.resolve_holdout_tail_weeks()
	holdout_start = history_end - pd.Timedelta(weeks=holdout_tail_weeks)
	window_summary = pd.DataFrame(
		[
			{
				"aggregation_level": manifest.dataset.aggregation_level,
				"feature_superset": "unknown",
				"horizon_start": manifest.backtest.horizon_start,
				"horizon_end": manifest.backtest.horizon_end,
				"window_count": len(windows),
				"origin_count": windows["forecast_origin"].nunique(),
				"shared_origin_grid": manifest.backtest.shared_origin_grid,
				"origin_step_weeks": manifest.backtest.step_weeks,
				"min_train_weeks": int(windows["train_weeks"].min()),
				"max_train_weeks": int(windows["train_weeks"].max()),
				"forecast_origin_start": windows["forecast_origin"].min(),
				"forecast_origin_end": windows["forecast_origin"].max(),
				"target_week_start": windows["target_week"].min(),
				"target_week_end": windows["target_week"].max(),
				"history_observed_end": history_end,
				"holdout_tail_weeks": holdout_tail_weeks,
				"history_cutoff_for_final_training": holdout_start,
				"backtest_targets_exclude_holdout_tail": holdout_tail_weeks > 0,
				"availability_rule": "all features must be known at forecast origin",
			}
		]
	)
	window_summary.to_csv(path, index=False)


def _load_manifest_snapshot(root: Path) -> dict[str, Any]:
	manifest_path = root / "run" / "manifest_snapshot.yaml"
	if not manifest_path.exists():
		return {}
	with manifest_path.open("r", encoding="utf-8") as file:
		loaded = yaml.safe_load(file)
	if loaded is None:
		return {}
	if not isinstance(loaded, dict):
		raise ValueError(f"Manifest snapshot must be a mapping: {manifest_path}")
	return loaded


def _is_experiment_manifest_snapshot(path: Path) -> bool:
	with path.open("r", encoding="utf-8") as file:
		loaded = yaml.safe_load(file)
	return isinstance(loaded, dict) and isinstance(loaded.get("models"), list)


def _as_dict(value: Any) -> dict[str, Any]:
	if isinstance(value, dict):
		return value
	return {}


if __name__ == "__main__":
	main()
