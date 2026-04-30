from mt.domain.tracking.tracking_backend import TrackingBackend
from mt.infra.tracking.backends.noop import NoopTrackingBackend


def build_tracking_backend(name: str | None) -> TrackingBackend:
	if name == "mlflow":
		from mt.infra.tracking.backends.mlflow import MlflowTrackingBackend

		return MlflowTrackingBackend()
	return NoopTrackingBackend()
