from collections.abc import Mapping
from enum import StrEnum
from typing import Callable, TypeVar

EnumT = TypeVar("EnumT", bound=StrEnum)
ValueT = TypeVar("ValueT")


def normalize_lower_key(value: str) -> str:
	return value.strip().lower()


def normalize_enum_by_key(
	value: str | EnumT,
	*,
	enum_type: type[EnumT],
	by_value: Mapping[str, EnumT],
	error_message: str | None = None,
) -> EnumT:
	if isinstance(value, enum_type):
		return value

	normalized_key = normalize_lower_key(value)
	try:
		return by_value[normalized_key]
	except KeyError as error:
		if error_message is not None:
			raise ValueError(error_message) from error
		raise ValueError() from error


def resolve_required_mapping(
	key: str | EnumT,
	*,
	mapping: Mapping[EnumT, ValueT],
	key_normalizer: Callable[[str | EnumT], EnumT],
	error_message: str | None = None,
) -> ValueT:
	resolved_key = key_normalizer(key)
	try:
		return mapping[resolved_key]
	except KeyError as error:
		if error_message is not None:
			raise ValueError(error_message) from error
		raise ValueError() from error
