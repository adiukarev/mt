from mt.domain.dataset.dataset_adapter_name import DatasetAdapterName, resolve_dataset_adapter_name
from mt.domain.dataset.dataset import DatasetAdapter
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.infra.dataset.adapters.favorita import FavoritaDatasetAdapter
from mt.infra.dataset.adapters.m5 import M5DatasetAdapter
from mt.infra.dataset.adapters.synthetic import SyntheticDatasetAdapter


DATASET_ADAPTER_CLASS_REGISTRY: dict[DatasetAdapterName, type[DatasetAdapter]] = {
	DatasetAdapterName.FAVORITA: FavoritaDatasetAdapter,
	DatasetAdapterName.M5: M5DatasetAdapter,
	DatasetAdapterName.SYNTHETIC: SyntheticDatasetAdapter,
}


def build_dataset_adapter(manifest: DatasetManifest) -> DatasetAdapter:
	adapter_name = resolve_dataset_adapter_name(manifest.kind)
	return DATASET_ADAPTER_CLASS_REGISTRY[adapter_name](manifest)
