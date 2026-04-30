from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from mt.domain.series_segmentation.series_segmentation_label import SegmentLabel
from mt.domain.series_segmentation.series_segmentation_policy import DEFAULT_SEGMENTATION_POLICY
from mt.domain.series_segmentation.series_segmentation_policy import SegmentationPolicy


def segment_series(
	weekly: pd.DataFrame,
	policy: SegmentationPolicy = DEFAULT_SEGMENTATION_POLICY,
) -> pd.DataFrame:
	rows: list[dict[str, object]] = []
	for series_id, group in weekly.groupby("series_id"):
		sales = group["sales_units"].astype(float)
		history_len = int(len(group))
		zero_share = float((sales == 0).mean())
		mean_value = float(sales.mean()) if history_len else 0.0
		std_value = float(sales.std(ddof=0)) if history_len else 0.0
		cov = std_value / mean_value if mean_value else float("inf")
		variance = float(np.var(sales))

		label = _classify_segment_label(history_len, zero_share, cov, policy)

		rows.append(
			{
				"series_id": series_id,
				"segment_label": label.value,
				"history_weeks": history_len,
				"zero_share": zero_share,
				"short_history": history_len < policy.short_history_weeks,
				"high_zero_share": zero_share >= policy.high_zero_share_threshold,
				"variance": variance,
				"high_variance": cov > policy.high_variance_cov_threshold if np.isfinite(cov) else True,
			}
		)

	return pd.DataFrame(rows)


def _classify_segment_label(
	history_weeks: int,
	zero_share: float,
	cov: float,
	policy: SegmentationPolicy,
) -> SegmentLabel:
	if history_weeks < policy.problematic_history_weeks:
		return SegmentLabel.PROBLEMATIC
	if zero_share >= policy.intermittent_zero_share_threshold:
		return SegmentLabel.INTERMITTENT
	if cov <= policy.stable_cov_threshold:
		return SegmentLabel.STABLE
	return SegmentLabel.MEDIUM_NOISE
