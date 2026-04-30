from enum import StrEnum

from mt.infra.helper.enum import normalize_enum_by_key


class SegmentLabel(StrEnum):
	PROBLEMATIC = "problematic"
	INTERMITTENT = "intermittent"
	STABLE = "stable"
	MEDIUM_NOISE = "medium_noise"


ALLOWED_SEGMENT_LABELS = frozenset(SegmentLabel)
SEGMENT_LABEL_BY_VALUE = {segment_label.value: segment_label for segment_label in SegmentLabel}


def normalize_segment_label(value: str | SegmentLabel) -> SegmentLabel:
	return normalize_enum_by_key(
		value,
		enum_type=SegmentLabel,
		by_value=SEGMENT_LABEL_BY_VALUE,
	)
