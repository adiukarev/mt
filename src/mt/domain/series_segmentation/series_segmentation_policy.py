from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
	problematic_history_weeks: int = 26
	short_history_weeks: int = 52
	intermittent_zero_share_threshold: float = 0.4
	high_zero_share_threshold: float = 0.4
	stable_cov_threshold: float = 0.3
	high_variance_cov_threshold: float = 1.0


DEFAULT_SEGMENTATION_POLICY = SegmentationPolicy()
