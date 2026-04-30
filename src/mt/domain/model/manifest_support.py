def normalize_optional_string(value: str | None) -> str | None:
	if value is None:
		return None
	resolved = value.strip()
	return resolved or None


def normalize_required_string(value: str | None, field_name: str) -> str:
	resolved = normalize_optional_string(value)
	if resolved is None:
		raise ValueError(f"{field_name} must be a non-empty string")
	return resolved
