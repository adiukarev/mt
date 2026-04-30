from mt.domain.dataset.dataset import DatasetBundle, DatasetLoadData
from mt.domain.dataset.dataset_manifest import DatasetManifest
from mt.infra.dataset.factory import build_dataset_adapter


def prepare_dataset(manifest: DatasetManifest, data: DatasetLoadData) -> DatasetBundle:
	return build_dataset_adapter(manifest).prepare(data)
