import argparse
from mt.infra.runtime.runtime import (
	ensure_runtime_env,
	ensure_runtime_logging,
	ensure_runtime_seed_everything
)


def main() -> None:
	parser = argparse.ArgumentParser()
	subparsers = parser.add_subparsers(dest="command", required=True)

	audit_parser = subparsers.add_parser(
		"audit",
		help="Запустить аудит данных"
	)
	audit_parser.add_argument(
		"--manifest",
		required=True,
		help="Путь к YAML манифесту аудита"
	)

	experiment_parser = subparsers.add_parser(
		"run-experiment",
		help="Запустить эксперимент"
	)
	experiment_parser.add_argument(
		"--manifest",
		required=True,
		help="Путь к YAML манифесту эксперимента"
	)

	synthetic_parser = subparsers.add_parser(
		"generate-synthetic",
		help="Сгенерировать synthetic weekly retail-датасет"
	)
	synthetic_parser.add_argument(
		"--manifest",
		required=True,
		help="Путь к YAML манифесту synthetic generator"
	)

	predict_parser = subparsers.add_parser(
		"predict",
		help="Построить прогноз по synthetic weekly dataset"
	)
	predict_parser.add_argument(
		"--manifest",
		required=True,
		help="Путь к YAML манифесту predict pipeline"
	)

	args = parser.parse_args()

	ensure_runtime_env()
	ensure_runtime_logging()

	if args.command == "audit":
		from mt.app.audit_pipeline import AuditPipeline
		from mt.domain.manifest import load_audit_manifest

		manifest = load_audit_manifest(args.manifest)
		ensure_runtime_seed_everything(manifest.runtime.seed)
		AuditPipeline().run(manifest=manifest)

	if args.command == "run-experiment":
		from mt.domain.manifest import load_experiment_manifest
		from mt.app.experiment_pipeline import ExperimentPipeline

		manifest = load_experiment_manifest(args.manifest)
		ensure_runtime_seed_everything(manifest.runtime.seed)
		ExperimentPipeline().run(manifest=manifest)

	if args.command == "generate-synthetic":
		from mt.app.synthetic_pipeline import SyntheticGenerationPipeline
		from mt.domain.synthetic import load_synthetic_manifest

		manifest = load_synthetic_manifest(args.manifest)
		ensure_runtime_seed_everything(manifest.runtime.seed)
		SyntheticGenerationPipeline().run(manifest=manifest)

	if args.command == "predict":
		from mt.app.predict_pipeline import PredictPipeline
		from mt.domain.predict_manifest import load_predict_manifest

		manifest = load_predict_manifest(args.manifest)
		ensure_runtime_seed_everything(manifest.runtime.seed)
		PredictPipeline().run(manifest=manifest)


if __name__ == "__main__":
	main()
