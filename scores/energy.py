import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch

class Energy:
    def __init__(self, args):
        self.args = args
        
    def setup(self, net, id_train_loader):
        pass
    
    @torch.no_grad()
    def get_score(self, net, loader):
        energy_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating Energy score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits = net(images, return_features=False).float()
            energy =  torch.logsumexp(logits, dim=1) # take temperature = 1
            energy_score_all.append(energy.cpu().numpy())

        return np.concatenate(energy_score_all, axis=0)