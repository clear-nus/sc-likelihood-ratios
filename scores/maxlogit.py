import numpy as np
from tqdm import tqdm
import torch


class MaxLogit:
    def __init__(self, args):
        self.args = args
        
    def setup(self, net, id_train_loader):
        pass
    
    @torch.no_grad()
    def get_score(self, net, loader):
        maxlogit_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating MaxLogit score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits = net(images, return_features=False).float()
            maxlogit_score_all.append(np.max(logits.cpu().numpy(), axis=1))
            
        return np.concatenate(maxlogit_score_all, axis=0)