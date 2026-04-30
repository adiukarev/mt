from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LagFeatureSpec:
	name: str
	formula: str
	lag: int


@dataclass(frozen=True, slots=True)
class RollingFeatureSpec:
	name: str
	formula: str
	window: int
	metric: str
