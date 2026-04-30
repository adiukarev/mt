import pandas as pd

from mt.domain.audit.audit_artifact import AuditArtifactData
from mt.domain.audit.audit_pipeline_context import AuditPipelineContext
from mt.infra.artifact.text_writer import write_csv, write_markdown


def write_preparation_artifacts(
	ctx: AuditPipelineContext,
	audit_artifacts: AuditArtifactData,
) -> None:
	if ctx.raw_dataset is None or ctx.dataset is None or ctx.segments is None:
		raise ValueError()

	file_descriptions: list[tuple[str, str]] = []
	raw_table_overview = _build_raw_table_overview(ctx.raw_dataset.tables)
	_write_preparation_frame(
		ctx,
		file_descriptions,
		1,
		"raw_table_overview.csv",
		raw_table_overview,
		"Обзор сырых таблиц: размеры, число колонок и список первых полей.",
	)

	for index, table_name in enumerate(sorted(ctx.raw_dataset.tables), start=2):
		frame = ctx.raw_dataset.tables[table_name].head(20).reset_index(drop=True)
		_write_preparation_frame(
			ctx,
			file_descriptions,
			index,
			f"raw_{table_name}_sample.csv",
			frame,
			f"Срез сырой таблицы `{table_name}` до адаптации.",
		)

	next_index = len(file_descriptions) + 1
	prepared_weekly = ctx.dataset.weekly.reset_index(drop=True)
	segmented_weekly = _build_segmented_weekly(prepared_weekly, ctx.segments)
	series_scope_summary = audit_artifacts.series_diagnostic_table.reset_index(drop=True)

	preparation_frames: list[tuple[str, pd.DataFrame, str]] = [
		(
			"prepared_weekly_schema.csv",
			_build_frame_schema(prepared_weekly),
			"Схема weekly-панели после dataset preparation: типы, заполненность и пример значений.",
		),
		(
			"prepared_weekly_sample.csv",
			prepared_weekly.head(40).reset_index(drop=True),
			"Срез weekly-панели после адаптации и подготовки датасета.",
		),
		(
			"segmented_weekly_sample.csv",
			segmented_weekly.head(40).reset_index(drop=True),
			"Тот же weekly-срез после присоединения сегментации по `series_id`.",
		),
		(
			"series_scope_summary.csv",
			series_scope_summary,
			"Полная таблица row-level diagnostics по рядам, реально вошедшим в текущий audit scope.",
		),
		(
			"data_dictionary.csv",
			audit_artifacts.data_dictionary.reset_index(drop=True),
			"Словарь колонок weekly-панели, используемой в audit.",
		),
		(
			"dataset_profile.csv",
			audit_artifacts.dataset_profile.reset_index(drop=True),
			"Краткий профиль dataset scope после подготовки и сегментации.",
		),
	]

	if not audit_artifacts.category_summary.empty:
		preparation_frames.append(
			(
				"category_summary.csv",
				audit_artifacts.category_summary.reset_index(drop=True),
				"Сводка по категориям внутри текущего audit scope.",
			)
		)
	if not audit_artifacts.sku_summary.empty:
		preparation_frames.append(
			(
				"sku_summary.csv",
				audit_artifacts.sku_summary.reset_index(drop=True),
				"Сводка по SKU внутри текущего audit scope.",
			)
		)

	for offset, (stem, frame, description) in enumerate(preparation_frames, start=next_index):
		_write_preparation_frame(ctx, file_descriptions, offset, stem, frame, description)


def _write_preparation_frame(
	ctx: AuditPipelineContext,
	file_descriptions: list[tuple[str, str]],
	index: int,
	stem: str,
	frame: pd.DataFrame,
	description: str,
) -> None:
	filename = _numbered_filename(index, stem)
	write_csv(ctx.artifacts_paths_map.preparation_file(filename), frame)
	file_descriptions.append((filename, description))


def _build_raw_table_overview(raw_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for table_name, frame in sorted(raw_tables.items()):
		rows.append(
			{
				"table_name": table_name,
				"row_count": int(len(frame)),
				"column_count": int(len(frame.columns)),
				"columns": " | ".join(str(column) for column in frame.columns[:20]),
			}
		)
	return pd.DataFrame(rows)


def _build_segmented_weekly(
	weekly: pd.DataFrame,
	segments: pd.DataFrame,
) -> pd.DataFrame:
	if segments.empty:
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
	return str(non_null.iloc[0])


def _numbered_filename(index: int, stem: str) -> str:
	return f"{index:02d}_{stem}"
