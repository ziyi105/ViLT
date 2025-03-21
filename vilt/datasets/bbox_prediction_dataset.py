from .base_dataset import BaseDataset
import torch

class BBoxPredictionDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        # Use only the training dataset
        names = ["bbox_prediction_train"]

        # Assuming your text descriptions are in a column named "description"
        super().__init__(*args, **kwargs, names=names, text_column_name="text")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        # Add bounding box information to the suite
        bbox = self.table["bbox"][index].as_py()
        suite.update({"bbox": torch.tensor(bbox, dtype=torch.float)})

        return suite
