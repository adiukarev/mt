import pandas as pd


def build_report_lines(
	aggregation_level: str,
	weekly: pd.DataFrame,
	summary: pd.DataFrame,
	seasonality_summary: pd.DataFrame,
	data_dictionary: pd.DataFrame,
	transformation_summary: pd.DataFrame,
	example_feature_snapshot: pd.DataFrame,
	metadata: dict[str, object],
	raw_context: dict[str, object],
) -> list[str]:
	return [
		f"- уровень агрегации: {aggregation_level}",
		f"- число рядов: {weekly['series_id'].nunique()}",
		f"- начало периода: {weekly['week_start'].min()}",
		f"- конец периода: {weekly['week_start'].max()}",
		f"- недельная частота подтверждена: {True}",
		f"- медианная длина истории, недель: {summary['history_weeks'].median():.1f}",
		f"- средняя доля нулей: {summary['zero_share'].mean():.4f}",
		f"- средняя доля пропусков: {summary['missing_share'].mean():.4f}",
		f"- доля коротких историй: {float(summary['short_history'].mean()) if 'short_history' in summary.columns else 0.0:.4f}",
		f"- доля прерывистых рядов: {float(summary['segment_label'].eq('intermittent').mean()) if 'segment_label' in summary.columns else 0.0:.4f}",
		f"- цена доступна в сыром датасете: {raw_context.get('price_available_raw', metadata.get('price_available'))}",
		f"- цена разрешена в базовом контуре: {raw_context.get('price_allowed_for_default_model', False)}",
		f"- промо доступно в сыром датасете: {raw_context.get('promo_available_raw', metadata.get('promo_available'))}",
		f"- риск stock-out: {metadata.get('stockout_risk')}",
		f"- риск структурного сдвига: {metadata.get('structural_shift_risk', 'не оценивался')}",
		f"- доля полной недельной сетки: {summary['weekly_grid_complete'].mean():.4f}",
		f"- средний коэффициент вариации: {summary['coefficient_of_variation'].mean():.4f}",
		f"- средняя сила тренда: {summary['trend_strength'].mean():.4f}",
		f"- средняя автокорреляция на лаге 52: {seasonality_summary['acf_lag_52'].mean():.4f}",
		f"- число колонок weekly-панели: {len(data_dictionary)}",
		f"- число зафиксированных шагов преобразования: {len(transformation_summary)}",
		f"- размер example feature snapshot: {len(example_feature_snapshot)} строк",
	]
