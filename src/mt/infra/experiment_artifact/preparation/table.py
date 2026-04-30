from collections.abc import Iterable

import pandas as pd

from mt.domain.experiment.experiment_pipeline_context import ExperimentPipelineContext
from mt.infra.artifact.text_writer import write_csv, write_markdown


def write_preparation_artifacts(ctx: ExperimentPipelineContext) -> None:
	if (
		ctx.raw_dataset is None
		or ctx.dataset is None
		or ctx.feature_registry is None
		or ctx.supervised is None
		or ctx.windows is None
	):
		raise ValueError()

	file_descriptions: list[tuple[str, str]] = []

	raw_table_overview = []
	next_index = 1
	for table_name, frame in sorted(ctx.raw_dataset.tables.items()):
		raw_table_overview.append(
			{
				"table_name": table_name,
				"row_count": int(len(frame)),
				"column_count": int(len(frame.columns)),
				"columns": " | ".join(str(column) for column in frame.columns[:20]),
			}
		)
		filename = _numbered_filename(next_index + 1, f"raw_{table_name}_sample.csv")
		write_csv(
			ctx.artifacts_paths_map.preparation_file(filename),
			frame.head(20).reset_index(drop=True),
		)
		file_descriptions.append((filename, f"Срез сырой таблицы `{table_name}` до адаптации."))
		next_index += 1

	overview_filename = _numbered_filename(1, "raw_table_overview.csv")
	write_csv(
		ctx.artifacts_paths_map.preparation_file(overview_filename),
		pd.DataFrame(raw_table_overview),
	)
	file_descriptions.insert(
		0,
		(overview_filename, "Обзор всех сырых таблиц: размеры, число колонок и список первых полей."),
	)

	prepared_weekly = ctx.dataset.weekly.reset_index(drop=True)
	segmented_weekly = _build_segmented_weekly(prepared_weekly, ctx.segments)
	supervised_sample = _build_supervised_sample(ctx)
	windows_sample = _build_windows_sample(ctx.windows)

	static_files = [
		(
			_numbered_filename(next_index + 1, "prepared_weekly_schema.csv"),
			_build_frame_schema(prepared_weekly),
			"Схема weekly-панели после dataset preparation: типы, заполненность и пример значений.",
		),
		(
			_numbered_filename(next_index + 2, "prepared_weekly_sample.csv"),
			prepared_weekly.head(40),
			"Срез weekly-панели после адаптации и подготовки датасета.",
		),
		(
			_numbered_filename(next_index + 3, "segmented_weekly_sample.csv"),
			segmented_weekly.head(40),
			"Тот же weekly-срез после присоединения сегментации по `series_id`.",
		),
		(
			_numbered_filename(next_index + 4, "supervised_sample.csv"),
			supervised_sample,
			"Срез итоговой supervised-таблицы с id/time, target horizon-ами и частью feature columns.",
		),
		(
			_numbered_filename(next_index + 5, "backtest_windows_sample.csv"),
			windows_sample,
			"Срез rolling backtest windows после генерации origin/horizon сетки.",
		),
	]
	for filename, frame, description in static_files:
		write_csv(ctx.artifacts_paths_map.preparation_file(filename), frame.reset_index(drop=True))
		file_descriptions.append((filename, description))

	write_markdown(
		ctx.artifacts_paths_map.preparation_file("00_preparation_lineage.md"),
		_build_preparation_lineage(file_descriptions),
	)


def _build_segmented_weekly(
	weekly: pd.DataFrame,
	segments: pd.DataFrame | None,
) -> pd.DataFrame:
	if segments is None or segments.empty:
		return weekly.copy()
	segment_columns = [column for column in ("series_id", "segment_label") if
	                   column in segments.columns]
	if segment_columns != ["series_id", "segment_label"]:
		return weekly.copy()
	return weekly.merge(
		segments.loc[:, ["series_id", "segment_label"]].drop_duplicates(),
		on="series_id",
		how="left",
	)


def _build_frame_schema(frame: pd.DataFrame) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	row_count = len(frame)
	for column in frame.columns:
		series = frame[column]
		non_null_count = int(series.notna().sum())
		rows.append(
			{
				"column_name": str(column),
				"dtype": str(series.dtype),
				"non_null_count": non_null_count,
				"non_null_share": float(non_null_count / row_count) if row_count > 0 else 0.0,
				"missing_count": int(series.isna().sum()),
				"sample_value": _sample_value(series),
			}
		)
	return pd.DataFrame(rows)


def _sample_value(series: pd.Series) -> str:
	non_null = series.dropna()
	if non_null.empty:
		return ""
	value = non_null.iloc[0]
	return str(value)


def _build_supervised_sample(ctx: ExperimentPipelineContext) -> pd.DataFrame:
	if ctx.supervised is None:
		raise ValueError()

	base_columns = [
		column for column in ("series_id", "week_start", "category", "segment_label")
		if column in ctx.supervised.columns
	]
	feature_columns = [
		column for column in ctx.feature_columns
		if column in ctx.supervised.columns
	][:8]
	target_columns = sorted(
		[column for column in ctx.supervised.columns if str(column).startswith("target_h")]
	)[:8]
	selected_columns = list(dict.fromkeys([*base_columns, *feature_columns, *target_columns]))
	return ctx.supervised.loc[:, selected_columns].head(40).reset_index(drop=True)


def _build_windows_sample(windows: pd.DataFrame) -> pd.DataFrame:
	if windows.empty:
		return windows.copy()

	sample = windows.copy()
	sample["target_week"] = sample["test_start"]
	sample["train_weeks"] = ((sample["train_end"] - sample["train_start"]).dt.days // 7) + 1
	sample["test_weeks"] = ((sample["test_end"] - sample["test_start"]).dt.days // 7) + 1
	sample["gap_weeks"] = ((sample["target_week"] - sample["train_end"]).dt.days // 7)
	columns = [
		column
		for column in (
			"forecast_origin",
			"horizon",
			"train_start",
			"train_end",
			"target_week",
			"test_start",
			"test_end",
			"train_weeks",
			"test_weeks",
			"gap_weeks",
			"feature_set",
		)
		if column in sample.columns
	]
	return sample.loc[:, columns].head(40).reset_index(drop=True)


def _numbered_filename(index: int, stem: str) -> str:
	return f"{index:02d}_{stem}"


def _build_preparation_lineage(file_descriptions: Iterable[tuple[str, str]]) -> list[str]:
	lines = [
		"# Preparation Lineage",
		"",
		"Порядок файлов отражает реальный flow подготовки данных от сырья к model-ready representation.",
		"",
	]
	for filename, description in file_descriptions:
		lines.append(f"- `{filename}`: {description}")
	lines.extend(
		[
			"- `06_feature_registry.csv`: полный реестр признаков с availability и формулами.",
			"- `07_feature_block_summary.csv`: агрегированный summary по feature groups.",
			"- `08_model_feature_usage.csv`: фактически используемые подмножества признаков по моделям.",
		]
	)
	return lines
