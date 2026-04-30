import pandas as pd

REQUIRED_FEATURE_REGISTRY_COLUMNS = {
	"name",
	"group",
	"source",
	"calculation",
	"covariate_class",
	"expected_effect_mechanism",
	"availability_at_forecast_time",
	"enabled",
	"reason_if_disabled",
}


def validate_feature_registry(registry: pd.DataFrame) -> None:
	missing_columns = sorted(REQUIRED_FEATURE_REGISTRY_COLUMNS.difference(registry.columns))
	if missing_columns:
		raise ValueError()

	invalid_mask = (
		registry["covariate_class"].fillna("").eq("")
		| registry["expected_effect_mechanism"].fillna("").eq("")
		| registry["calculation"].fillna("").eq("")
	)
	if bool(invalid_mask.any()):
		raise ValueError()
