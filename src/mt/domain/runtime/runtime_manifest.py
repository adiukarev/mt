from dataclasses import dataclass


@dataclass(slots=True)
class RuntimeManifest:
	artifacts_dir: str
	seed: int = 42

	def __post_init__(self) -> None:
		if not self.artifacts_dir:
			raise ValueError()
