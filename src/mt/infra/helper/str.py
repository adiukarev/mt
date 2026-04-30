import re


def to_snake_case_without_suffix(value: str, suffix: str) -> str:
	if suffix and value.endswith(suffix):
		value = value[: -len(suffix)]
	return re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
