from enum import StrEnum


class FeatureGroup(StrEnum):
	LAGS = "lags"
	ROLLING = "rolling"
	CALENDAR = "calendar"
	CATEGORICAL = "categorical"


ALLOWED_FEATURE_GROUPS = frozenset(FeatureGroup)


class FeatureSet(StrEnum):
	F0 = "F0"
	F1 = "F1"
	F2 = "F2"
	F3 = "F3"
	F4 = "F4"
	F5 = "F5"
	F6 = "F6"


ALLOWED_FEATURE_SETS = frozenset(FeatureSet)

DEFAULT_FEATURE_SET = FeatureSet.F4

FEATURE_SET_GROUPS: dict[FeatureSet, frozenset[FeatureGroup]] = {
	FeatureSet.F0: frozenset(),
	FeatureSet.F1: frozenset({FeatureGroup.LAGS}),
	FeatureSet.F2: frozenset({FeatureGroup.LAGS, FeatureGroup.ROLLING}),
	FeatureSet.F3: frozenset({FeatureGroup.LAGS, FeatureGroup.ROLLING, FeatureGroup.CALENDAR}),
	FeatureSet.F4: frozenset(
		{
			FeatureGroup.LAGS,
			FeatureGroup.ROLLING,
			FeatureGroup.CALENDAR,
			FeatureGroup.CATEGORICAL,
		}
	),
	FeatureSet.F5: frozenset(
		{
			FeatureGroup.LAGS,
			FeatureGroup.ROLLING,
			FeatureGroup.CALENDAR,
			FeatureGroup.CATEGORICAL,
		}
	),
	FeatureSet.F6: frozenset(
		{
			FeatureGroup.LAGS,
			FeatureGroup.ROLLING,
			FeatureGroup.CALENDAR,
			FeatureGroup.CATEGORICAL,
		}
	),
}

FEATURE_SET_ALIASES: dict[str, FeatureSet] = {
	"F4_smoke": FeatureSet.F4,
	"F5": FeatureSet.F4,
	"F6": FeatureSet.F4,
}

FEATURE_SET_RANKS: dict[FeatureSet, int] = {
	FeatureSet.F0: 0,
	FeatureSet.F1: 1,
	FeatureSet.F2: 2,
	FeatureSet.F3: 3,
	FeatureSet.F4: 4,
	FeatureSet.F5: 5,
	FeatureSet.F6: 6,
}


def normalize_feature_set(value: str | FeatureSet) -> FeatureSet:
	if isinstance(value, FeatureSet):
		return value

	raw_value = value.strip()

	try:
		return FeatureSet(raw_value)
	except ValueError:
		pass

	canonical_feature_set = FEATURE_SET_ALIASES.get(raw_value)
	if canonical_feature_set is not None:
		return canonical_feature_set

	prefix = raw_value.split("_", maxsplit=1)[0]
	try:
		return FeatureSet(prefix)
	except ValueError:
		return DEFAULT_FEATURE_SET


def get_feature_groups(value: str | FeatureSet) -> frozenset[FeatureGroup]:
	return FEATURE_SET_GROUPS[normalize_feature_set(value)]


def max_feature_set(values: list[str | FeatureSet]) -> FeatureSet:
	return max((normalize_feature_set(value) for value in values), key=FEATURE_SET_RANKS.__getitem__)
