from mt.domain.dataset.dataset import DatasetBundle
from mt.domain.probabilistic.probabilistic import ProbabilisticSource
from mt.infra.forecast.reference_model import ReferenceModelConfig


def resolve_forecast_inference_policy(
	dataset_bundle: DatasetBundle,
	reference_model: ReferenceModelConfig,
	horizon_weeks: int,
) -> dict[str, object]:
	reasons: list[str] = []
	artifact = reference_model.artifact
	if artifact is None:
		raise FileNotFoundError(
			f"Saved model artifact is unavailable: {reference_model.source_dir / 'model.pkl'}"
		)
	training_manifest = artifact.dataset_manifest

	if dataset_bundle.kind != training_manifest.kind:
		reasons.append("dataset_kind_mismatch_requires_refit")
	if dataset_bundle.aggregation_level != artifact.training_aggregation_level:
		reasons.append("aggregation_level_mismatch_requires_refit")
	if dataset_bundle.target_name != training_manifest.target_name:
		reasons.append("target_name_mismatch_requires_refit")
	if horizon_weeks > max(artifact.horizons):
		reasons.append("requested_horizon_exceeds_saved_artifact")

	source_by_horizon = artifact.probabilistic_source_by_horizon
	conformal_state = artifact.conformal_calibrator_state
	if (
		not any(source == ProbabilisticSource.NATIVE for source in source_by_horizon.values())
		and not conformal_state
	):
		reasons.append("saved_artifact_has_no_probabilistic_state")

	if reasons:
		raise ValueError(
			"Saved model artifact is incompatible with forecast request: "
			+ ", ".join(reasons)
		)
	return {
		"mode": "saved_artifact",
		"reasons": ["saved_artifact_compatible"],
	}
