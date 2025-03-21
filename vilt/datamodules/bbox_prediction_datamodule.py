from vilt.datasets import BBoxPredictionDataset
from .datamodule_base import BaseDataModule

class BBoxPredictionDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return BBoxPredictionDataset

    @property
    def dataset_cls_no_false(self):
        return BBoxPredictionDataset

    @property
    def dataset_name(self):
        return "bbox_prediction"
