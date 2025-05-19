import numpy as np
from tqdm import tqdm
import torch

class RLog:
    def __init__(self, args):
        self.args = args
        
    def setup(self, net, id_train_loader):
        pass
    
    @torch.no_grad()
    def get_score(self, net, loader):
        rlog_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating RLog score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits = net(images, return_features=False).float()

            top2_vals, _ = torch.topk(logits, 2, axis=1)
            rlog_score_all.append((top2_vals[:, 0] - top2_vals[:, 1]).cpu().numpy())
            
        return np.concatenate(rlog_score_all, axis=0)