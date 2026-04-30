def build_lag_feature_name(lag: int) -> str:
	return f"lag_{lag}"


def build_rolling_feature_name(metric: str, window: int) -> str:
	return f"rolling_{metric}_{window}"
