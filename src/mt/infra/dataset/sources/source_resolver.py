from mt.domain.dataset.dataset_source_manifest import DatasetSourceManifest
from mt.domain.dataset.dataset_source_type import DatasetSourceType
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.domain.dataset.dataset_kind import DatasetKind
from mt.infra.dataset.sources.base import DatasetSourceService
from mt.infra.dataset.sources.clickhouse import ClickhouseDatasetSourceService
from mt.infra.dataset.sources.local_files import LocalFilesDatasetSourceService
from mt.infra.dataset.sources.postgres import PostgresDatasetSourceService
from mt.infra.dataset.sources.synthetic_refresh import SyntheticRefreshDatasetSourceService


DATASET_SOURCE_SERVICE_REGISTRY: dict[DatasetSourceType, type[DatasetSourceService]] = {
	DatasetSourceType.LOCAL: LocalFilesDatasetSourceService,
	DatasetSourceType.SYNTHETIC: SyntheticRefreshDatasetSourceService,
	DatasetSourceType.POSTGRES: PostgresDatasetSourceService,
	DatasetSourceType.CLICKHOUSE: ClickhouseDatasetSourceService,
}


def build_dataset_source_service(
	source_manifest: DatasetSourceManifest,
) -> DatasetSourceService:
	return DATASET_SOURCE_SERVICE_REGISTRY[source_manifest.source_type](source_manifest)


def build_dataset_source_service_for_dataset(
	dataset_manifest: DatasetManifest,
) -> DatasetSourceService:
	return build_dataset_source_service(infer_dataset_source_manifest(dataset_manifest))


def infer_dataset_source_manifest(
	dataset_manifest: DatasetManifest,
) -> DatasetSourceManifest:
	if dataset_manifest.kind == DatasetKind.SYNTHETIC:
		return DatasetSourceManifest(source_type=DatasetSourceType.SYNTHETIC)
	return DatasetSourceManifest(source_type=DatasetSourceType.LOCAL)
