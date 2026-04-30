from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(slots=True)
class ModelConfigEtsManifest:
	trend: str = "add"
	seasonal: str | None = None
	seasonal_periods: int | None = None
	initialization_method: str = "estimated"

	def __post_init__(self) -> None:
		if not self.trend:
			raise ValueError()
		if not self.initialization_method:
			raise ValueError()
		if self.seasonal_periods is not None and self.seasonal_periods < 2:
			raise ValueError()

	@classmethod
	def from_mapping(cls, payload: dict[str, Any] | None) -> "ModelConfigEtsManifest":
		if payload is None:
			return cls()
		allowed_fields = {field.name for field in fields(cls)}
		return cls(**{key: value for key, value in payload.items() if key in allowed_fields})

	def to_dict(self) -> dict[str, Any]:
		return {
			key: value
			for key, value in asdict(self).items()
			if value is not None
		}
