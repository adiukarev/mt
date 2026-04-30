from mt.domain.pipeline.pipeline_context import BasePipelineContext
from mt.infra.tracking.backend_resolver import resolve_tracking_backend_name


def build_tracking_summary(ctx: BasePipelineContext) -> dict[str, object]:
	if ctx.observability is None:
		return {
			"run_key": None,
			"pipeline_type": None,
			"tracking_backend": None,
			"tracking_run_id": None,
			"tracking_namespace": None,
			"execution_mode": None,
		}

	return {
		"run_key": ctx.observability.run_key,
		"pipeline_type": ctx.observability.pipeline_type,
		"tracking_backend": resolve_tracking_backend_name(ctx.observability.execution_mode),
		"tracking_run_id": ctx.observability.tracking_run_id,
		"tracking_namespace": ctx.observability.tracking_namespace,
		"execution_mode": ctx.observability.execution_mode,
	}
