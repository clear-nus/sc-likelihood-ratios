import torch
import numpy as np
from tqdm import tqdm
import faiss
import os
from data_utils.helper import get_im_split_names

class KNN:
    def __init__(self, args):
        self.args = args
        self.k = args.k
        
    def setup(self, net, id_train_loader):
        os.makedirs('knn_embeddings', exist_ok=True)
        name = get_im_split_names(self.args)
        knn_path = f'knn_embeddings/{self.args.model_type}_{name}.npy'

        if not os.path.isfile(knn_path):
            self.knn_features = self.get_knn_embeddings(net, id_train_loader)
            np.save(knn_path, self.knn_features)
        else:
            self.knn_features = np.load(knn_path)

    @torch.no_grad()
    def get_score(self, net, loader):
        index = faiss.IndexFlatL2(self.knn_features.shape[1])
        index.add(self.knn_features)
        knn_score_all = []
        
        tqdm_object = tqdm(loader, total=len(loader),  desc='calculating KNN score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits, features = net(images, return_features=True)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()
            D, _ = index.search(features, self.k)
            kth_dist = -D[:, -1]
            knn_score_all.extend(kth_dist)

        return np.asarray(knn_score_all, dtype=np.float32)


    @torch.no_grad()
    def get_knn_embeddings(self, net, loader):

        all_features = []
        tqdm_object = tqdm(loader, total=len(loader), desc='calculating KNN embeddings')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits, features = net(images, return_features=True)
            features /= features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu().numpy())
        
        all_features = np.asarray(np.concatenate(all_features, axis=0), dtype=np.float32)
        return all_features

