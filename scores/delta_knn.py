import torch
import numpy as np
from tqdm import tqdm
import faiss
import os
from data_utils.helper import get_im_split_names

# def top1_residuals(output, target):
#     """
#     Compute per‐sample top‐1 residuals: 1 if the model’s top prediction is wrong, 0 if it’s right.
#     """
#     # Get the index of the highest‐scoring class for each sample
#     pred = output.argmax(dim=1)
    
#     # Compare to target and convert to 1/0 residuals
#     residuals = pred.ne(target).cpu().numpy().astype(int).tolist()
#     return residuals


def residuals_per_sample(output, target, topk=(1, 5)):
    """
        1 if incorrect, 0 if correct
    """
    # Get the top-k predictions for each sample
    maxk = max(topk)
    pred = output.topk(maxk, 1, True, True)[1]  # Top-k indices
    correct = pred.eq(target.view(-1, 1))  # Compare predictions with target

    # Initialize lists to store per-sample correctness
    top1_residuals = (1 - correct[:, 0].cpu().numpy().astype(int)).tolist()  # Top-1
    top5_residuals = (1 - correct[:, :5].any(dim=1).cpu().numpy().astype(int)).tolist()  # Top-5

    return top1_residuals, top5_residuals


class DeltaKNN:
    def __init__(self, args):
        self.args = args
        self.k = args.k
        
    def setup(self, net, id_train_loader):
        os.makedirs('delta_knn_embeddings', exist_ok=True)
        name = get_im_split_names(self.args)
        knn_path = f'delta_knn_embeddings/{self.args.model_type}_{name}.npz'

        if not os.path.isfile(knn_path):
            self.knn_features_correct, self.knn_features_wrong = self.get_delta_knn_embeddings(net, id_train_loader)
            np.savez(knn_path, correct_features=self.knn_features_correct, wrong_features=self.knn_features_wrong)
        else:
            all_features = np.load(knn_path)
            self.knn_features_correct, self.knn_features_wrong = all_features['correct_features'], all_features['wrong_features']

    @torch.no_grad()
    def get_score(self, net, loader):
        correct_index = faiss.IndexFlatL2(self.knn_features_correct.shape[1])
        wrong_index = faiss.IndexFlatL2(self.knn_features_wrong.shape[1])
        
        correct_index.add(self.knn_features_correct)
        wrong_index.add(self.knn_features_wrong)
        
        knn_score_all = []
        
        tqdm_object = tqdm(loader, total=len(loader),  desc='calculating Delta KNN score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits, features = net(images, return_features=True)
            features /= features.norm(dim=-1, keepdim=True)
            features = features.cpu().numpy()
            
            correct_D, _ = correct_index.search(features, self.k)
            wrong_D, _ = wrong_index.search(features, self.k)
            avg_topk_correct_dist = -np.log(correct_D + 1e-6).mean(axis=-1)
            avg_topk_wrong_dist = -np.log(wrong_D + 1e-6).mean(axis=-1)
            
            knn_score = avg_topk_correct_dist - avg_topk_wrong_dist
            knn_score_all.extend(knn_score)
            
        return np.asarray(knn_score_all, dtype=np.float32)

    @torch.no_grad()
    def get_delta_knn_embeddings(self, net, loader):

        all_features_correct = []
        all_features_wrong = []
        
        tqdm_object = tqdm(loader, total=len(loader), desc='calculating Delta KNN embeddings')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images, labels = images.cuda(), labels.cuda()
            logits, features = net(images, return_features=True)
            features /= features.norm(dim=-1, keepdim=True)
            
            residuals, _ = residuals_per_sample(logits, labels, topk=(1, 5))
            residuals = torch.tensor(residuals)
            
            all_features_correct.append(features[residuals.eq(0)].cpu())
            all_features_wrong.append(features[residuals.eq(1)].cpu())
        
        all_features_correct = np.asarray(np.concatenate(all_features_correct, axis=0), dtype=np.float32)
        all_features_wrong = np.asarray(np.concatenate(all_features_wrong, axis=0), dtype=np.float32)
    
        return all_features_correct, all_features_wrong

