from dataclasses import dataclass, field

from mt.domain.feature.feature_set import FeatureSet, normalize_feature_set
from mt.domain.feature.feature_history_formula_default import DEFAULT_LAGS, DEFAULT_ROLLING_WINDOWS


@dataclass(slots=True)
class FeatureManifest:
	enabled: bool = False
	feature_set: FeatureSet = FeatureSet.F4

	lags: list[int] = field(default_factory=lambda: list(DEFAULT_LAGS))
	rolling_windows: list[int] = field(default_factory=lambda: list(DEFAULT_ROLLING_WINDOWS))
	use_calendar: bool = True
	use_category_encodings: bool = True

	def __post_init__(self) -> None:
		self.feature_set = normalize_feature_set(self.feature_set)

		if any(lag <= 0 for lag in self.lags):
			raise ValueError()
		if any(window <= 1 for window in self.rolling_windows):
			raise ValueError()

		self.lags = sorted(set(self.lags))
		self.rolling_windows = sorted(set(self.rolling_windows))
