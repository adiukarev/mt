def resolve_tracking_backend_name(execution_mode: str | None) -> str:
	if execution_mode == "airflow":
		return "mlflow"
	return "noop"
