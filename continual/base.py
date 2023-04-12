import torch.nn as nn

class BaseContinualModel(nn.Module):
    def __init__(self, backbone) -> None:
        super().__init__()

        self.backbone = backbone
        
    def update_model(self, *args, **kwargs):
        raise NotImplementedError()
    
    def forward_features(self, x):
        raise NotImplementedError()
    
    def forward_classifier(self, feats):
        raise NotImplementedError()


class BaseContinualTrainer:
    def __init__(self) -> None:
        pass
