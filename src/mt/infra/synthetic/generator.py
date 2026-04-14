from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt

from mt.domain.synthetic import SyntheticManifest
from mt.infra.artifact.plot_labels import set_axis_labels, translate_label


@dataclass(slots=True)
class SyntheticGenerationArtifacts:
	"""Ссылки на ключевые synthetic-артефакты."""

	dataset_csv: Path
	metadata_csv: Path
	preview_files: list[Path]
	report_md: Path
	manifest_snapshot_yaml: Path


def generate_synthetic_dataset(manifest: SyntheticManifest) -> SyntheticGenerationArtifacts:
	"""Сгенерировать synthetic weekly retail-датасет и демонстрационные артефакты"""

	from mt.app.synthetic_pipeline import SyntheticGenerationPipeline

	ctx = SyntheticGenerationPipeline().run(manifest)
	output_dir = ctx.artifacts_paths_map.root

	dataset_csv = output_dir / "dataset" / f"{manifest.runtime.dataset_name}.csv"
	metadata_csv = output_dir / "dataset" / "series_metadata.csv"
	manifest_snapshot_yaml = output_dir / "run" / "generation_manifest_snapshot.yaml"
	report_md = output_dir / "dataset" / "README.md"
	preview_files = [
		output_dir / "preview" / "data_dictionary.csv",
		output_dir / "preview" / "dataset_profile.csv",
		output_dir / "preview" / "weekly_panel_sample.csv",
		output_dir / "preview" / "example_series_sample.csv",
		output_dir / "preview" / "scenario_weekly_sales.png",
		output_dir / "preview" / "sample_series_grid.png",
		output_dir / "preview" / "series_sales_distribution.png",
	]

	return SyntheticGenerationArtifacts(
		dataset_csv=dataset_csv,
		metadata_csv=metadata_csv,
		preview_files=preview_files,
		report_md=report_md,
		manifest_snapshot_yaml=manifest_snapshot_yaml,
	)


def generate_dataset_frame(manifest: SyntheticManifest) -> pd.DataFrame:
	rng = np.random.default_rng(manifest.runtime.seed)
	scenario_frames = [_build_dataset_for_scenario(manifest, scenario, rng) for scenario in
	                   manifest.scenarios]
	return pd.concat(scenario_frames, ignore_index=True)


def _build_dataset(manifest: SyntheticManifest, rng: np.random.Generator) -> pd.DataFrame:
	return _build_dataset_for_scenario(manifest, manifest.scenarios[0], rng)


def _build_dataset_for_scenario(
	manifest: SyntheticManifest,
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
		promo_lift = (
			_uniform(rng, manifest.promo.lift_min, manifest.promo.lift_max)
			* getattr(scenario, "promo_lift_scale")
			if manifest.promo.enabled
			else 0.0
		)
		base_price = (
			_uniform(rng, manifest.price.base_price_min, manifest.price.base_price_max)
			if manifest.price.enabled
			else np.nan
		)
		elasticity = (
			_uniform(rng, manifest.price.elasticity_min, manifest.price.elasticity_max)
			* getattr(scenario, "price_elasticity_scale")
			if manifest.price.enabled
			else 0.0
		)
		category_bias = 1.0 + 0.03 * (manifest.series.categories.index(category) - 1)

		for step, week_start in enumerate(week_index):
			promo_flag = (
				int(rng.random() < min(
					manifest.promo.probability * getattr(scenario, "promo_probability_scale"), 1.0))
				if manifest.promo.enabled
				else 0
			)
			discount_depth = (
				_uniform(
					rng,
					manifest.price.discount_depth_min,
					manifest.price.discount_depth_max,
				)
				if manifest.price.enabled and rng.random() < min(
					manifest.price.discount_probability * getattr(scenario, "discount_probability_scale"),
					1.0)
				else 0.0
			)
			price = (
				base_price * (1.0 - discount_depth)
				if manifest.price.enabled
				else np.nan
			)
			trend_component = 1.0 + (trend_slope * step / max(total_weeks - 1, 1))
			yearly_component = 1.0 + yearly_amplitude * np.sin((2.0 * np.pi * step / 52.0) + phase_year)
			monthly_component = 1.0 + monthly_amplitude * np.sin(
				(2.0 * np.pi * step / 13.0) + phase_month)
			promo_component = 1.0 + (promo_lift * promo_flag)
			price_component = (
				(price / base_price) ** elasticity
				if manifest.price.enabled and base_price > 0
				else 1.0
			)
			expected_sales = max(
				base_level * category_bias * trend_component * yearly_component * monthly_component
				* promo_component * price_component,
				1.0,
			)
			sales_units = _sample_sales(expected_sales, manifest, scenario, rng)
			if rng.random() < getattr(scenario, "zero_inflation_probability"):
				sales_units = 0

			rows.append(
				{
					"scenario_name": scenario_name,
					"series_id": series_id,
					"category": category,
					"week_start": week_start,
					"week_of_year": int(week_start.isocalendar().week),
					"time_index": step,
					"is_history": step < manifest.calendar.history_weeks,
					"sales_units": sales_units,
					"expected_sales_units": round(expected_sales, 3),
					"promo_planned": promo_flag,
					"discount_depth": round(discount_depth, 4),
					"price": round(float(price), 4) if manifest.price.enabled else np.nan,
					"known_in_advance_price": manifest.price.enabled,
					"known_in_advance_promo": manifest.promo.enabled,
					"covariate_price_class": "known_in_advance" if manifest.price.enabled else "not_used",
					"covariate_promo_class": "known_in_advance" if manifest.promo.enabled else "not_used",
					"target_name": "sales_units",
				}
			)

	return pd.DataFrame(rows)


def _sample_sales(
	expected_sales: float,
	manifest: SyntheticManifest,
	scenario: object,
	rng: np.random.Generator,
) -> int:
	# Gamma-Poisson смесь дает controllable overdispersion без отрицательных значений.
	dispersion_alpha = manifest.noise.dispersion_alpha * max(getattr(scenario, "noise_scale"), 1e-6)
	shape = 1.0 / dispersion_alpha
	scale = expected_sales * dispersion_alpha
	latent_rate = max(rng.gamma(shape=shape, scale=scale), 0.1)
	sales_units = int(rng.poisson(latent_rate))

	if rng.random() < min(
		manifest.noise.outlier_probability * getattr(scenario, "outlier_probability_scale"), 1.0):
		sales_units = int(
			sales_units * _uniform(
				rng,
				manifest.noise.outlier_multiplier_min,
				manifest.noise.outlier_multiplier_max,
			)
		)
	if rng.random() < min(
		manifest.noise.stockout_probability * getattr(scenario, "stockout_probability_scale"), 1.0):
		sales_units = int(
			sales_units * (
				1.0 - _uniform(rng, manifest.noise.stockout_depth_min, manifest.noise.stockout_depth_max))
		)

	return max(sales_units, 0)


def build_series_metadata(dataset: pd.DataFrame) -> pd.DataFrame:
	metadata = (
		dataset.groupby(["scenario_name", "series_id", "category"], as_index=False)
		.agg(
			history_weeks=("is_history", "sum"),
			total_weeks=("week_start", "count"),
			mean_sales=("sales_units", "mean"),
			std_sales=("sales_units", "std"),
			zero_share=("sales_units", lambda values: float((values == 0).mean())),
			promo_share=("promo_planned", "mean"),
			avg_price=("price", "mean"),
		)
	)
	metadata["std_sales"] = metadata["std_sales"].fillna(0.0)
	return metadata


def build_demo_forecast_frame(dataset: pd.DataFrame, manifest: SyntheticManifest) -> pd.DataFrame:
	overlay_series_id = manifest.demo_forecast.overlay_series_id or str(dataset["series_id"].iloc[0])
	default_scenario = str(dataset["scenario_name"].iloc[0])
	series_frame = (
		dataset.loc[
			(dataset["scenario_name"] == default_scenario) & (dataset["series_id"] == overlay_series_id)]
		.sort_values("week_start")
		.reset_index(drop=True)
	)
	if series_frame.empty:
		raise ValueError()

	history_weeks = manifest.calendar.history_weeks
	horizon_weeks = manifest.calendar.horizon_weeks
	observed = series_frame.iloc[:history_weeks].copy()
	future = series_frame.iloc[history_weeks:history_weeks + horizon_weeks].copy()

	predictions: list[float] = []
	history_values = observed["sales_units"].tolist()
	for horizon_idx in range(horizon_weeks):
		if manifest.demo_forecast.model_name == "seasonal_naive" and len(history_values) >= 52:
			prediction = float(history_values[-52 + horizon_idx])
		else:
			prediction = float(history_values[-1])
		predictions.append(max(prediction, 0.0))

	return pd.DataFrame(
		{
			"scenario_name": default_scenario,
			"series_id": overlay_series_id,
			"category": str(series_frame["category"].iloc[0]),
			"model_name": manifest.demo_forecast.model_name,
			"forecast_origin": observed["week_start"].iloc[-1],
			"target_date": future["week_start"].to_numpy(),
			"horizon": np.arange(1, horizon_weeks + 1, dtype=int),
			"actual": future["sales_units"].to_numpy(),
			"prediction": np.round(predictions, 3),
		}
	)


def write_preview_plots(dataset: pd.DataFrame, output_dir: Path) -> list[Path]:
	scenario_weekly_sales_path = output_dir / "scenario_weekly_sales.png"
	sample_series_grid_path = output_dir / "sample_series_grid.png"
	series_sales_distribution_path = output_dir / "series_sales_distribution.png"
	_write_scenario_weekly_sales_plot(dataset, scenario_weekly_sales_path)
	_write_sample_series_grid_plot(dataset, sample_series_grid_path)
	_write_series_sales_distribution_plot(dataset, series_sales_distribution_path)
	return [
		scenario_weekly_sales_path,
		sample_series_grid_path,
		series_sales_distribution_path,
	]


def _write_scenario_weekly_sales_plot(dataset: pd.DataFrame, output_path: Path) -> None:
	weekly = (
		dataset.groupby(["scenario_name", "week_start"], as_index=False)
		.agg(total_sales_units=("sales_units", "sum"))
	)
	scenario_names = sorted(weekly["scenario_name"].unique())
	fig, axes = plt.subplots(
		nrows=len(scenario_names),
		ncols=1,
		figsize=(14, max(4, 3.5 * len(scenario_names))),
		sharex=True,
	)
	if len(scenario_names) == 1:
		axes = [axes]
	for ax, scenario_name in zip(axes, scenario_names):
		frame = weekly.loc[weekly["scenario_name"] == scenario_name].sort_values("week_start")
		ax.plot(frame["week_start"], frame["total_sales_units"], color="#1f4e79", linewidth=2.0)
		ax.fill_between(frame["week_start"], frame["total_sales_units"], color="#98c1d9", alpha=0.35)
		set_axis_labels(
			ax,
			title=f"Суммарные недельные продажи: {scenario_name}",
			ylabel="sales_units",
		)
		ax.grid(alpha=0.2)
	axes[-1].set_xlabel(translate_label("week_start"))
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=170)
	plt.close(fig)


def _write_sample_series_grid_plot(dataset: pd.DataFrame, output_path: Path) -> None:
	sample_keys = (
		dataset[["scenario_name", "series_id"]]
		.drop_duplicates()
		.sort_values(["scenario_name", "series_id"])
		.head(4)
	)
	fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=False)
	flat_axes = list(axes.flatten())
	for ax, row in zip(flat_axes, sample_keys.itertuples(index=False)):
		series_frame = dataset.loc[
			(dataset["scenario_name"] == row.scenario_name) & (dataset["series_id"] == row.series_id)
			].sort_values("week_start")
		plot_frame = series_frame.tail(60)
		ax.plot(
			plot_frame["week_start"],
			plot_frame["sales_units"],
			color="#1f4e79",
			linewidth=1.9,
		)
		ax.set_title(f"{row.scenario_name} / {row.series_id}")
		ax.set_xlabel(translate_label("week_start"))
		ax.set_ylabel(translate_label("sales_units"))
		ax.grid(alpha=0.2)
	for ax in flat_axes[len(sample_keys):]:
		ax.axis("off")
	fig.suptitle("Сетка примеров рядов", y=0.98)
	fig.tight_layout(rect=(0, 0, 1, 0.96))
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=170)
	plt.close(fig)


def _write_series_sales_distribution_plot(dataset: pd.DataFrame, output_path: Path) -> None:
	summary = (
		dataset.groupby(["scenario_name", "series_id"], as_index=False)
		.agg(mean_sales_units=("sales_units", "mean"))
	)
	scenario_names = sorted(summary["scenario_name"].unique())
	grouped_values = [
		summary.loc[summary["scenario_name"] == scenario_name, "mean_sales_units"].to_numpy()
		for scenario_name in scenario_names
	]
	fig, ax = plt.subplots(figsize=(12, 5))
	ax.boxplot(grouped_values, tick_labels=scenario_names, patch_artist=True)
	set_axis_labels(
		ax,
		title="Распределение средних продаж по рядам",
		xlabel="scenario_name",
		ylabel="mean_sales_units",
	)
	ax.grid(axis="y", alpha=0.2)
	fig.tight_layout()
	output_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_path, dpi=170)
	plt.close(fig)


def build_preview_artifacts(dataset: pd.DataFrame, output_dir: Path) -> list[Path]:
	output_dir.mkdir(parents=True, exist_ok=True)
	data_dictionary_path = output_dir / "data_dictionary.csv"
	dataset_profile_path = output_dir / "dataset_profile.csv"
	weekly_panel_sample_path = output_dir / "weekly_panel_sample.csv"
	example_series_sample_path = output_dir / "example_series_sample.csv"

	build_preview_data_dictionary(dataset).to_csv(data_dictionary_path, index=False)
	build_preview_dataset_profile(dataset).to_csv(dataset_profile_path, index=False)
	build_preview_weekly_panel_sample(dataset).to_csv(weekly_panel_sample_path, index=False)
	build_preview_example_series_sample(dataset).to_csv(example_series_sample_path, index=False)

	plot_paths = write_preview_plots(dataset, output_dir)

	return [
		data_dictionary_path,
		dataset_profile_path,
		weekly_panel_sample_path,
		example_series_sample_path,
		*plot_paths,
	]


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
		"promo_planned": "known_in_advance",
		"discount_depth": "known_in_advance",
		"price": "known_in_advance",
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
			{"metric": "promo_share", "value": float(dataset["promo_planned"].mean())},
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
		"price",
		"promo_planned",
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
			["scenario_name", "series_id", "category", "week_start", "is_history", "sales_units", "price",
			 "promo_planned"],
		].tail(12)
		frames.append(frame)
	return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_generation_report(
	manifest: SyntheticManifest,
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
		"- `promo_planned`: known-in-advance",
		"- `price`: known-in-advance",
		"- `sales_units`: target / observed over time",
		"- `expected_sales_units`: скрытая вспомогательная величина генератора, не для честного forecast-контура",
		"",
		"## Ограничения",
		"- Спрос синтетический и задан формулой генератора, поэтому он проще реального retail-процесса.",
		"- Промо и цена считаются заранее известными по определению манифеста.",
		"- Не моделируются реальные межтоварные замещения, supply-chain лаги, календарь праздников и иерархия M5.",
		"- Preview-каталог содержит audit-like артефакты и графики synthetic данных.",
	]
	lines.extend(["", "## Распределение рядов по категориям"])
	for (scenario_name, category), count in category_counts.items():
		lines.append(f"- {scenario_name} / {category}: {count}")
	lines.append("")
	return "\n".join(lines)


def _uniform(rng: np.random.Generator, minimum: float, maximum: float) -> float:
	return float(rng.uniform(minimum, maximum))


def _render_manifest_yaml(manifest: SyntheticManifest) -> str:
	return yaml.safe_dump(manifest.as_dict(), sort_keys=False, allow_unicode=True)
