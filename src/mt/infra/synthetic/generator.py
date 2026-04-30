import numpy as np
import pandas as pd

from mt.domain.synthetic_generation.synthetic_generation_pipeline_manifest import SyntheticGenerationPipelineManifest


def generate_dataset_frame(manifest: SyntheticGenerationPipelineManifest) -> pd.DataFrame:
	rng = np.random.default_rng(manifest.runtime.seed)
	scenario_frames = [
		_build_dataset_for_scenario(manifest, scenario, rng)
		for scenario in manifest.scenarios
	]
	return pd.concat(scenario_frames, ignore_index=True)


def _build_dataset_for_scenario(
	manifest: SyntheticGenerationPipelineManifest,
	scenario: object,
	rng: np.random.Generator,
) -> pd.DataFrame:
	scenario_name = getattr(scenario, "name")
	total_weeks = manifest.calendar.history_weeks + manifest.calendar.horizon_weeks
	week_index = pd.date_range(
		start=manifest.calendar.start_date,
		periods=total_weeks,
		freq=manifest.calendar.week_anchor,
	)
	rows: list[dict[str, object]] = []
	category_cycle = [
		manifest.series.categories[idx % len(manifest.series.categories)]
		for idx in range(manifest.series.series_count)
	]
	rng.shuffle(category_cycle)

	for series_number in range(manifest.series.series_count):
		series_id = f"synthetic_series_{series_number + 1:03d}"
		category = category_cycle[series_number]
		base_level = _uniform(rng, manifest.series.base_level_min, manifest.series.base_level_max)
		base_level *= getattr(scenario, "level_scale")
		trend_slope = _uniform(rng, manifest.series.trend_slope_min, manifest.series.trend_slope_max)
		trend_slope *= getattr(scenario, "trend_scale")
		yearly_amplitude = _uniform(
			rng,
			manifest.series.yearly_amplitude_min,
			manifest.series.yearly_amplitude_max,
		)
		yearly_amplitude *= getattr(scenario, "yearly_amplitude_scale")
		monthly_amplitude = _uniform(
			rng,
			manifest.series.monthly_amplitude_min,
			manifest.series.monthly_amplitude_max,
		)
		monthly_amplitude *= getattr(scenario, "monthly_amplitude_scale")
		phase_year = _uniform(rng, 0.0, 2.0 * np.pi)
		phase_month = _uniform(rng, 0.0, 2.0 * np.pi)
		category_bias = 1.0 + 0.03 * (manifest.series.categories.index(category) - 1)

		for step, week_start in enumerate(week_index):
			is_history = step < manifest.calendar.history_weeks
			trend_component = 1.0 + (trend_slope * step / max(total_weeks - 1, 1))
			yearly_component = 1.0 + yearly_amplitude * np.sin((2.0 * np.pi * step / 52.0) + phase_year)
			monthly_component = 1.0 + monthly_amplitude * np.sin(
				(2.0 * np.pi * step / 13.0) + phase_month)
			expected_sales = max(
				base_level * category_bias * trend_component * yearly_component * monthly_component,
				1.0,
			)
			sales_units = _sample_sales(expected_sales, manifest, scenario, rng, is_history=is_history)
			if rng.random() < _zero_inflation_probability(scenario, is_history=is_history):
				sales_units = 0

			rows.append(
				{
					"scenario_name": scenario_name,
					"series_id": series_id,
					"category": category,
					"week_start": week_start,
					"week_of_year": int(week_start.isocalendar().week),
					"time_index": step,
					"is_history": is_history,
					"sales_units": sales_units,
					"expected_sales_units": round(expected_sales, 3),
					"target_name": "sales_units",
				}
			)

	return pd.DataFrame(rows)


def _sample_sales(
	expected_sales: float,
	manifest: SyntheticGenerationPipelineManifest,
	scenario: object,
	rng: np.random.Generator,
	is_history: bool,
) -> int:
	expected_sales *= _recent_multiplier(scenario, "recent_level_scale", is_history=is_history)
	noise_scale = getattr(scenario, "noise_scale") * _recent_multiplier(
		scenario,
		"recent_noise_scale",
		is_history=is_history,
	)
	dispersion_alpha = manifest.noise.dispersion_alpha * max(
		noise_scale,
		1e-6,
	)
	shape = 1.0 / dispersion_alpha
	scale = expected_sales * dispersion_alpha
	latent_rate = max(rng.gamma(shape=shape, scale=scale), 0.1)
	sales_units = int(rng.poisson(latent_rate))

	if rng.random() < min(
		manifest.noise.outlier_probability
		* getattr(scenario, "outlier_probability_scale")
		* _recent_multiplier(scenario, "recent_outlier_probability_scale", is_history=is_history),
		1.0,
	):
		sales_units = int(
			sales_units * _uniform(
				rng,
				manifest.noise.outlier_multiplier_min,
				manifest.noise.outlier_multiplier_max,
			)
		)
	if rng.random() < min(
		manifest.noise.stockout_probability
		* getattr(scenario, "stockout_probability_scale")
		* _recent_multiplier(scenario, "recent_stockout_probability_scale", is_history=is_history),
		1.0,
	):
		sales_units = int(
			sales_units
			* (1.0 - _uniform(rng, manifest.noise.stockout_depth_min, manifest.noise.stockout_depth_max))
		)

	return max(sales_units, 0)


def _recent_multiplier(scenario: object, field_name: str, is_history: bool) -> float:
	if is_history:
		return 1.0
	return float(getattr(scenario, field_name))


def _zero_inflation_probability(scenario: object, is_history: bool) -> float:
	base_probability = float(getattr(scenario, "zero_inflation_probability"))
	if is_history:
		return base_probability
	recent_probability = float(getattr(scenario, "recent_zero_inflation_probability"))
	return min(base_probability + recent_probability, 1.0)


def build_series_metadata(dataset: pd.DataFrame) -> pd.DataFrame:
	metadata = (
		dataset.groupby(["scenario_name", "series_id", "category"], as_index=False)
		.agg(
			history_weeks=("is_history", "sum"),
			total_weeks=("week_start", "count"),
			mean_sales=("sales_units", "mean"),
			std_sales=("sales_units", "std"),
			zero_share=("sales_units", lambda values: float((values == 0).mean())),
		)
	)
	metadata["std_sales"] = metadata["std_sales"].fillna(0.0)
	return metadata


def build_preview_data_dictionary(dataset: pd.DataFrame) -> pd.DataFrame:
	role_map = {
		"scenario_name": "scenario",
		"series_id": "key",
		"category": "static",
		"week_start": "time_index",
		"time_index": "time_index",
		"is_history": "split_flag",
		"sales_units": "target",
		"expected_sales_units": "generator_internal",
	}
	rows: list[dict[str, object]] = []
	for column in dataset.columns:
		rows.append(
			{
				"column_name": column,
				"dtype": str(dataset[column].dtype),
				"role": role_map.get(column, "derived"),
				"non_null_share": float(dataset[column].notna().mean()),
				"example_value": str(dataset[column].dropna().iloc[0]) if dataset[
					column].notna().any() else "",
			}
		)
	return pd.DataFrame(rows)


def build_preview_dataset_profile(dataset: pd.DataFrame) -> pd.DataFrame:
	return pd.DataFrame(
		[
			{"metric": "number_of_rows", "value": int(len(dataset))},
			{"metric": "number_of_series", "value": int(dataset["series_id"].nunique())},
			{"metric": "number_of_scenarios", "value": int(dataset["scenario_name"].nunique())},
			{"metric": "number_of_categories", "value": int(dataset["category"].nunique())},
			{"metric": "period_start", "value": str(pd.Timestamp(dataset["week_start"].min()).date())},
			{"metric": "period_end", "value": str(pd.Timestamp(dataset["week_start"].max()).date())},
			{"metric": "history_rows", "value": int(dataset["is_history"].astype(bool).sum())},
			{"metric": "future_rows", "value": int((~dataset["is_history"].astype(bool)).sum())},
			{"metric": "mean_sales_units", "value": float(dataset["sales_units"].mean())},
			{"metric": "zero_share", "value": float((dataset["sales_units"] == 0).mean())},
		]
	)


def build_preview_weekly_panel_sample(dataset: pd.DataFrame) -> pd.DataFrame:
	columns = [
		"scenario_name",
		"series_id",
		"category",
		"week_start",
		"is_history",
		"sales_units",
	]
	return dataset.loc[:, [column for column in columns if column in dataset.columns]].head(
		20).reset_index(drop=True)


def build_preview_example_series_sample(dataset: pd.DataFrame) -> pd.DataFrame:
	sample_keys = (
		dataset[["scenario_name", "series_id"]]
		.drop_duplicates()
		.sort_values(["scenario_name", "series_id"])
		.head(3)
	)
	frames: list[pd.DataFrame] = []
	for row in sample_keys.itertuples(index=False):
		frame = dataset.loc[
			(dataset["scenario_name"] == row.scenario_name) & (dataset["series_id"] == row.series_id),
			["scenario_name", "series_id", "category", "week_start", "is_history", "sales_units"],
		].tail(12)
		frames.append(frame)
	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_generation_report(
	manifest: SyntheticGenerationPipelineManifest,
	dataset: pd.DataFrame,
	metadata: pd.DataFrame,
) -> str:
	category_counts = dataset.groupby(["scenario_name", "category"])[
		"series_id"].nunique().sort_index()
	scenario_names = ", ".join(scenario.name for scenario in manifest.scenarios)
	lines = [
		"# Synthetic Weekly Dataset",
		"",
		"## Что внутри",
		f"- Рядов: {manifest.series.series_count}",
		f"- Сценарии: {scenario_names}",
		f"- Категорий: {', '.join(manifest.series.categories)}",
		f"- История: {manifest.calendar.history_weeks} недель",
		f"- Горизонт future actual: {manifest.calendar.horizon_weeks} недель",
		f"- Всего строк: {len(dataset)}",
		f"- Средняя продажа по ряду: {metadata['mean_sales'].mean():.2f}",
		"",
		"## Классы ковариат",
		"- `sales_units`: target / observed over time",
		"- `expected_sales_units`: скрытая вспомогательная величина генератора, не для честного forecast-контура",
	]
	lines.extend(["", "## Распределение рядов по категориям"])
	for (scenario_name, category), count in category_counts.items():
		lines.append(f"- {scenario_name} / {category}: {count}")
	lines.append("")
	return "\n".join(lines)


def _uniform(rng: np.random.Generator, minimum: float, maximum: float) -> float:
	return float(rng.uniform(minimum, maximum))
