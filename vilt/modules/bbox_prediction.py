import torch
from vilt.modules import ViLTransformerSS

class ViLTForBBoxPrediction(ViLTransformerSS):
    def __init__(self, config):
        super().__init__(config)
        self.bbox_head = torch.nn.Linear(config.hidden_size, 4)

    def forward(self, batch):
        outputs = super().forward(batch)
        bbox_pred = self.bbox_head(outputs["cls_feats"])
        return {"bbox_pred": bbox_pred, **outputs}

def compute_bbox_loss(pred, target):
    return torch.nn.functional.mse_loss(pred, target)
