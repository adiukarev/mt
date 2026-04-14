import pandas as pd

from mt.domain.model import ModelResult


def build_run_summary(
	aggregation_level: str,
	feature_set: str,
	windows_count: int,
	seed: int,
	overall_metrics: pd.DataFrame,
	model_feature_usage_rows: list[dict[str, object]],
	executed_stages: list[str],
	stage_timings: list[dict[str, object]],
	pipeline_wall_time_seconds: float | None,
) -> list[str]:
	"""Собрать основной run summary"""

	lines = [
		"# Сводка запуска",
		"",
		"## Что запускалось",
		f"- уровень агрегации: {aggregation_level}",
		f"- набор признаков: {feature_set}",
		f"- число окон backtesting: {windows_count}",
		f"- seed: {seed}",
		f"- полное время пайплайна, сек.: {_format_seconds(pipeline_wall_time_seconds)}",
		"",
		"## Что было проверено",
	]
	if not executed_stages:
		lines.append("- Этапы не выполнялись")
	else:
		stage_timings_map = {
			str(stage_info["stage_name"]): stage_info for stage_info in stage_timings
		}
		for stage_name in executed_stages:
			stage_info = stage_timings_map.get(stage_name, {})
			lines.append(
				f"- {stage_name}: выполнено | wall_time={_format_seconds(stage_info.get('wall_time_seconds'))} сек."
			)
	lines.extend(["", "## Модели"])
	if not model_feature_usage_rows:
		lines.append("- Сводка по моделям недоступна")
	else:
		for row in model_feature_usage_rows:
			lines.append(
				f"- {row['model_name']}: family={row['model_family']} | "
				f"feature_count={row['feature_count']} | "
				f"features={'enabled' if int(row['feature_count']) > 0 else 'disabled'}"
			)
	lines.extend(["", "## Что получилось"])
	if overall_metrics.empty:
		lines.append("- Успешных запусков моделей не было")
	else:
		lines.append("- итоговый артефакт финальной модели: models/best_model")
		for rank, row in enumerate(overall_metrics.itertuples(index=False), start=1):
			lines.append(
				f"- место {rank}: {row.model_name} | WAPE={row.WAPE:.4f} | sMAPE={row.sMAPE:.4f} | "
				f"MAE={row.MAE:.4f} | Bias={row.Bias:.4f}"
			)

	return lines


def build_model_run_summary(result: ModelResult, metrics_overall: pd.DataFrame) -> list[str]:
	"""Собрать краткий run summary для отдельной модели"""

	row = metrics_overall.iloc[0]
	lines = [
		f"# {result.info.model_name}",
		"",
		f"- число использованных признаков: {len(result.used_feature_columns)}",
		f"- время обучения, с: {_format_seconds(result.train_time_seconds)}",
		f"- время инференса, с: {_format_seconds(result.inference_time_seconds)}",
		f"- полное время модели, с: {_format_seconds(result.wall_time_seconds)}",
		"",
		"## Метрики",
		f"- WAPE: {row['WAPE']:.4f}",
		f"- sMAPE: {row['sMAPE']:.4f}",
		f"- MAE: {row['MAE']:.4f}",
		f"- Bias: {row['Bias']:.4f}",
	]

	return lines


def build_comparison_report(
	overall_metrics: pd.DataFrame,
	metrics_by_segment: pd.DataFrame,
	bootstrap_ci: pd.DataFrame,
	rolling_vs_holdout: pd.DataFrame,
	baseline_name: str = "seasonal_naive",
) -> list[str]:
	"""Собрать общий отчет сравнения моделей"""

	lines = ["# Отчет по сравнению", "", "## Рейтинг"]
	if overall_metrics.empty:
		lines.append("- Сопоставимые результаты отсутствуют")
		return lines

	winner = overall_metrics.iloc[0]
	baseline_row = overall_metrics[overall_metrics["model_name"] == baseline_name]
	baseline_wape = float(baseline_row.iloc[0]["WAPE"]) if not baseline_row.empty else None
	for row in overall_metrics.itertuples(index=False):
		delta_to_best = row.WAPE - winner["WAPE"]
		suffix = ""
		if baseline_wape is not None:
			suffix = f", разница_с_{baseline_name}={row.WAPE - baseline_wape:.4f}"
		lines.append(
			f"- {row.model_name}: WAPE={row.WAPE:.4f}, разница_с_лидером={delta_to_best:.4f}{suffix}"
		)

	lines.extend(["", "## Устойчивость по сегментам"])
	if metrics_by_segment.empty:
		lines.append("-")
	else:
		best_segment = (
			metrics_by_segment.sort_values(["segment_label", "WAPE"])
			.groupby("segment_label", as_index=False)
			.first()
		)
		for row in best_segment.itertuples(index=False):
			lines.append(f"- {row.segment_label}: best={row.model_name}, WAPE={row.WAPE:.4f}")

	lines.extend(["", "## Бутстрэп"])
	if bootstrap_ci.empty:
		lines.append("-")
	else:
		for row in bootstrap_ci.itertuples(index=False):
			lines.append(
				f"- {row.model_a} - {row.model_b}: средняя_разность={row.mean_difference:.4f}, "
				f"95% ДИ=[{row.ci95_low:.4f}, {row.ci95_high:.4f}]"
			)

	lines.extend(["", "## Rolling Vs Holdout"])
	if rolling_vs_holdout.empty:
		lines.append("-")
	else:
		lines.append(
			"- При допущении конечной дисперсии потерь по forecast origin средняя rolling-оценка по K окнам имеет дисперсию порядка sigma^2 / K, тогда как одиночный holdout соответствует K = 1."
		)
		for row in rolling_vs_holdout.itertuples(index=False):
			lines.append(
				f"- {row.model_name}: rolling_WAPE={row.rolling_WAPE:.4f}, "
				f"holdout_last_origin={row.holdout_WAPE_last_origin:.4f}, "
				f"std_WAPE_по_origin={row.origin_WAPE_std:.4f}, "
				f"SE_rolling={row.rolling_WAPE_standard_error:.4f}, "
				f"фактор_снижения_дисперсии~{row.variance_reduction_factor_vs_single_holdout:.1f}"
			)

	lines.extend(["", "## Финальная рекомендация"])
	lines.append(f"- выбранная модель: {winner['model_name']}")
	lines.append(f"- метрика выбора: WAPE={winner['WAPE']:.4f}")
	if baseline_wape is not None:
		lines.append(f"- улучшение_относительно_{baseline_name}: {baseline_wape - winner['WAPE']:.4f}")
	lines.append(
		"- область применимости: рекомендация валидна только для проверенных rolling-окон, принятой feature policy и текущей настройки датасета"
	)
	return lines


def _format_seconds(value: float | None) -> str:
	if value is None:
		return "n/a"
	return f"{value:.3f}"
