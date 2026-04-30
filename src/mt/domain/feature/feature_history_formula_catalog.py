from mt.domain.feature.feature_history_formula_default import EXTENDED_ROLLING_WINDOWS
from mt.domain.feature.feature_history_formula_name import (
	build_lag_feature_name,
	build_rolling_feature_name,
)
from mt.domain.feature.feature_history_formula_spec import LagFeatureSpec, RollingFeatureSpec


def build_lag_feature_specs(lags: list[int]) -> list[LagFeatureSpec]:
	return [
		LagFeatureSpec(
			name=build_lag_feature_name(lag),
			formula=f"lag_{lag}(t) = y_(t-{lag})",
			lag=lag,
		)
		for lag in sorted(set(lags))
	]


def build_rolling_feature_specs(windows: list[int]) -> list[RollingFeatureSpec]:
	specs: list[RollingFeatureSpec] = []
	for window in sorted(set(windows)):
		specs.extend(
			[
				RollingFeatureSpec(
					name=build_rolling_feature_name("mean", window),
					formula=f"rolling_mean_{window}(t) = mean(y_(t-{window}), ..., y_(t-1))",
					window=window,
					metric="mean",
				),
				RollingFeatureSpec(
					name=build_rolling_feature_name("median", window),
					formula=f"rolling_median_{window}(t) = median(y_(t-{window}), ..., y_(t-1))",
					window=window,
					metric="median",
				),
				RollingFeatureSpec(
					name=build_rolling_feature_name("mad", window),
					formula=(
						f"rolling_mad_{window}(t) = median(|y_(t-{window}), ..., y_(t-1) "
						f"- rolling_median_{window}(t)|)"
					),
					window=window,
					metric="mad",
				),
				RollingFeatureSpec(
					name=build_rolling_feature_name("iqr", window),
					formula=(
						f"rolling_iqr_{window}(t) = q75(y_(t-{window}), ..., y_(t-1)) - "
						f"q25(y_(t-{window}), ..., y_(t-1))"
					),
					window=window,
					metric="iqr",
				),
				RollingFeatureSpec(
					name=build_rolling_feature_name("robust_zscore", window),
					formula=(
						f"rolling_robust_zscore_{window}(t) = 0.6745 * "
						f"(y_(t-1) - rolling_median_{window}(t)) / rolling_mad_{window}(t)"
					),
					window=window,
					metric="robust_zscore",
				),
				RollingFeatureSpec(
					name=build_rolling_feature_name("recent_outlier_flag", window),
					formula=(
						f"rolling_recent_outlier_flag_{window}(t) = "
						f"|rolling_robust_zscore_{window}(t)| > 3.5"
					),
					window=window,
					metric="recent_outlier_flag",
				),
			]
		)
		if window in EXTENDED_ROLLING_WINDOWS:
			specs.extend(
				[
					RollingFeatureSpec(
						name=build_rolling_feature_name("std", window),
						formula=f"rolling_std_{window}(t) = std(y_(t-{window}), ..., y_(t-1))",
						window=window,
						metric="std",
					),
					RollingFeatureSpec(
						name=build_rolling_feature_name("max", window),
						formula=f"rolling_max_{window}(t) = max(y_(t-{window}), ..., y_(t-1))",
						window=window,
						metric="max",
					),
					RollingFeatureSpec(
						name=build_rolling_feature_name("min", window),
						formula=f"rolling_min_{window}(t) = min(y_(t-{window}), ..., y_(t-1))",
						window=window,
						metric="min",
					),
				]
			)
	return specs
