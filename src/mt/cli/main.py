import argparse

from mt.infra.runtime.runtime import (
	ensure_runtime_env,
	ensure_runtime_logging,
	ensure_runtime_seed_everything,
)


def main() -> None:
	ensure_runtime_env()

	from mt.orchestration.pipeline_resolver import resolve_cli_pipeline_definitions

	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command", required=True)

	for command_name, definition in resolve_cli_pipeline_definitions().items():
		if command_name in definition.cli_aliases:
			continue
		command_parser = subparsers.add_parser(
			command_name,
			aliases=list(definition.cli_aliases),
			help=definition.cli_help or f"Запустить {definition.pipeline_type}",
		)
		command_parser.add_argument(
			"--manifest",
			required=True,
			help=f"Путь к YAML манифесту pipeline `{definition.pipeline_type}`",
		)

	args = parser.parse_args()

	ensure_runtime_logging()

	definition = resolve_cli_pipeline_definitions()[args.command]
	manifest = definition.manifest_loader(args.manifest)
	seed = getattr(getattr(manifest, "runtime", None), "seed", None)
	if isinstance(seed, int):
		ensure_runtime_seed_everything(seed)
	ctx = definition.pipeline_factory().run(manifest=manifest)
	if definition.pipeline_type == "monitoring":
		_print_monitoring_result(ctx)


def _print_monitoring_result(ctx: object) -> None:
	decision_artifact = getattr(ctx, "decision_artifact", None)
	if decision_artifact is None:
		print("Monitoring completed without decision artifact")
		return

	print(f"Monitoring decision: {decision_artifact.decision_action}")
	print(f"Quality gate passed: {decision_artifact.quality_gate_passed}")
	print(f"Should run experiment: {decision_artifact.should_run_experiment}")
	if decision_artifact.reasons:
		print(f"Reasons: {', '.join(decision_artifact.reasons)}")
	metrics = dict(decision_artifact.monitoring_metrics)
	for key in ("recent_wape", "distribution_shift_score", "zero_share_delta", "row_count_delta"):
		if key in metrics:
			print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
	main()
