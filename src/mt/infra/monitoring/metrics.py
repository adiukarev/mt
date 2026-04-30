from datetime import datetime, timezone

import numpy as np
import pandas as pd


def build_monitoring_metrics(
	reference_frame: pd.DataFrame,
	recent_actuals: pd.DataFrame,
	predictions: pd.DataFrame,
	reference_weeks: int,
) -> dict[str, float]:
	metrics: dict[str, float] = {}
	reference_window = _tail_weeks(reference_frame, reference_weeks)
	recent_window = recent_actuals.copy()

	metrics["data_freshness_hours"] = _data_freshness_hours(recent_window, reference_frame)
	metrics["row_count_delta"] = _relative_delta(len(recent_window), len(reference_window))
	metrics["zero_share_delta"] = _share(recent_window, "sales_units", 0) - _share(
		reference_window,
		"sales_units",
		0,
	)
	metrics["mean_sales_delta"] = _relative_delta(
		float(recent_window["sales_units"].mean()) if not recent_window.empty else 0.0,
		float(reference_window["sales_units"].mean()) if not reference_window.empty else 0.0,
	)
	metrics["distribution_shift_score"] = _distribution_shift_score(reference_window, recent_window)
	metrics["recent_wape"] = _wape(predictions)
	metrics["reference_rows"] = float(len(reference_frame))
	metrics["recent_rows"] = float(len(recent_actuals))
	metrics["prediction_rows"] = float(len(predictions))
	return metrics


def build_quality_gate_summary(
	metrics: dict[str, float],
	max_recent_wape: float,
	max_distribution_shift_score: float,
	max_zero_share_delta: float,
	max_row_count_delta: float,
	max_alert_score: float,
	issues: list[str] | None = None,
) -> dict[str, object]:
	alert_score = 0.0
	reasons: list[str] = []
	resolved_issues = list(issues or [])
	if resolved_issues:
		return {
			"alert_score": alert_score,
			"alert_level": "warning",
			"passed": False,
			"reasons": resolved_issues,
			"decision_action": "manual_review",
		}

	if metrics.get("recent_wape", 0.0) > max_recent_wape:
		alert_score += 1.0
		reasons.append("recent_wape_exceeded")
	if abs(metrics.get("distribution_shift_score", 0.0)) > max_distribution_shift_score:
		alert_score += 1.0
		reasons.append("distribution_shift_exceeded")
	if abs(metrics.get("zero_share_delta", 0.0)) > max_zero_share_delta:
		alert_score += 1.0
		reasons.append("zero_share_delta_exceeded")
	if abs(metrics.get("row_count_delta", 0.0)) > max_row_count_delta:
		alert_score += 1.0
		reasons.append("row_count_delta_exceeded")

	if alert_score >= 3.0:
		alert_level = "high"
	elif alert_score >= 1.0:
		alert_level = "warning"
	else:
		alert_level = "info"

	if "recent_wape_exceeded" in reasons or alert_score >= max_alert_score:
		decision_action = "retrain_required"
	else:
		decision_action = "no_action"

	return {
		"alert_score": alert_score,
		"alert_level": alert_level,
		"passed": decision_action == "no_action",
		"reasons": reasons,
		"decision_action": decision_action,
	}


def _tail_weeks(frame: pd.DataFrame, weeks: int) -> pd.DataFrame:
	if frame.empty:
		return frame.copy()
	unique_weeks = sorted(pd.to_datetime(frame["week_start"]).unique())
	selected = set(unique_weeks[-weeks:])
	return frame.loc[frame["week_start"].isin(selected)].copy()


def _relative_delta(current: float, baseline: float) -> float:
	if baseline == 0:
		return 0.0 if current == 0 else 1.0
	return float((current - baseline) / baseline)


def _share(frame: pd.DataFrame, column: str, value: object) -> float:
	if frame.empty:
		return 0.0
	return float((frame[column] == value).mean())


def _distribution_shift_score(reference_frame: pd.DataFrame, recent_frame: pd.DataFrame) -> float:
	if reference_frame.empty or recent_frame.empty:
		return 0.0
	reference_mean = float(reference_frame["sales_units"].mean())
	recent_mean = float(recent_frame["sales_units"].mean())
	reference_std = float(reference_frame["sales_units"].std(ddof=0) or 0.0)
	recent_std = float(recent_frame["sales_units"].std(ddof=0) or 0.0)
	mean_delta = abs(_relative_delta(recent_mean, reference_mean))
	std_delta = abs(_relative_delta(recent_std, reference_std))
	zero_delta = abs(_share(recent_frame, "sales_units", 0) - _share(reference_frame, "sales_units", 0))
	return float(np.mean([mean_delta, std_delta, zero_delta]))


def _wape(predictions: pd.DataFrame) -> float:
	if predictions.empty:
		return 0.0
	denominator = float(predictions["actual"].abs().sum())
	if denominator == 0:
		return 0.0
	numerator = float((predictions["actual"] - predictions["prediction"]).abs().sum())
	return numerator / denominator


def _data_freshness_hours(recent_frame: pd.DataFrame, reference_frame: pd.DataFrame) -> float:
	source = recent_frame if not recent_frame.empty else reference_frame
	if source.empty:
		return 0.0
	latest_week = pd.Timestamp(source["week_start"].max())
	now = datetime.now(timezone.utc)
	latest_point = (latest_week + pd.Timedelta(days=7)).to_pydatetime().replace(tzinfo=timezone.utc)
	return max((now - latest_point).total_seconds() / 3600.0, 0.0)
