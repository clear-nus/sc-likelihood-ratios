import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import entropy
import torch.nn.functional as F


class MSP:
    def __init__(self, args):
        self.args = args
        
    def setup(self, net, id_train_loader):
        pass
    
    @torch.no_grad()
    def get_score(self, net, loader):
        msp_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating MSP score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits = net(images, return_features=False).float()
            smax = F.softmax(logits, dim=1)
            msp_score_all.append(np.max(smax.cpu().numpy(), axis=1))
            
        return np.concatenate(msp_score_all, axis=0)
 