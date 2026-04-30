from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.infra.dataset.sources.base import DatasetSourceRefreshResult, DatasetSourceService


class ClickhouseDatasetSourceService(DatasetSourceService):
	def refresh_dataset(
		self,
		dataset_manifest: DatasetManifest,
	) -> DatasetSourceRefreshResult:
		raise NotImplementedError("ClickHouse dataset source is not implemented yet")
