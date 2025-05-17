import torch.nn as nn
import timm
from timm.data import resolve_data_config, create_transform


class EVAModel(nn.Module):
    def __init__(self):
        super().__init__()
        model_name = "eva_giant_patch14_224.clip_ft_in1k"
        self.net = timm.create_model(model_name, pretrained=True).cuda()
        
        config = resolve_data_config({}, model=self.net)
        self.preprocess = create_transform(**config)

    def forward(self, x, return_features=False):

        features = self.net.forward_features(x)
        pred = self.net.forward_head(features)
        
        if return_features:
            pre_logits_features = self.net.forward_head(features, pre_logits=True)
            return pred, pre_logits_features
        else:
            return pred
        
    def get_fc(self):
        fc = self.net.head
        return fc.weight.cpu().detach().numpy(), fc.bias.cpu().detach().numpy()
    
    