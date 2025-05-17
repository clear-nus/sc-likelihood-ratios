import torch
import numpy as np
from tqdm import tqdm
import sklearn.covariance
from copy import deepcopy
import os
from data_utils.helper import get_im_split_names


class MDS:
    def __init__(self, args):
        self.args = args
    
    def setup(self, net, id_train_loader):
        os.makedirs('mds_stats', exist_ok=True)
        name = get_im_split_names(self.args)
        mean_prec_path = f'mds_stats/{self.args.model_type}_{name}.pt'
        
        if not os.path.isfile(mean_prec_path):
            class_mean, precision, whole_mean, whole_precision = self.get_mean_prec(net, id_train_loader)
            torch.save([class_mean, precision, whole_mean, whole_precision], mean_prec_path)
        else:
            class_mean, precision, whole_mean, whole_precision = torch.load(mean_prec_path, map_location='cpu', weights_only=True)
        
        self.class_mean = class_mean.cuda()
        self.precision = precision.cuda()

    @torch.no_grad()
    def get_score(self, net, loader):
        Mahalanobis_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating MDS score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits, features = net(images, return_features=True)
            class_scores = torch.zeros((features.shape[0], self.args.n_cls)).cuda()
            
            for c in range(self.args.n_cls):
                tensor = features - self.class_mean[c].view(1, -1)
                class_scores[:, c] = -torch.matmul(torch.matmul(tensor, self.precision), tensor.t()).diag()
            
            Mahalanobis_score = torch.max(class_scores, dim=1)[0]
            Mahalanobis_score_all.extend(Mahalanobis_score.cpu().numpy())

        return np.asarray(Mahalanobis_score_all, dtype=np.float32)

    @torch.no_grad()
    def get_mean_prec(self, net, loader):
        all_features = []
        all_labels = []
        for idx, (images, labels) in enumerate(tqdm(loader, desc="calculating MDS stats")):
            images = images.cuda()   
            logits, features = net(images, return_features=True)
            all_labels.append(deepcopy(labels))
            all_features.append(features.cpu())
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        
        # compute class-conditional statistics
        class_mean = []
        centered_data = []
        for c in range(self.args.n_cls):
            class_samples = all_features[all_labels.eq(c)].data
            class_mean.append(class_samples.mean(0))
            centered_data.append(class_samples - class_mean[c].view(1, -1))

        class_mean = torch.stack(class_mean)  # shape [number of classes, feature dim]
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(torch.cat(centered_data).cpu().numpy().astype(np.float32))
        precision = torch.from_numpy(group_lasso.precision_).float()
        
        # for RMDS, calculate mean and precision over all features irrespective of class
        whole_mean = all_features.mean(0)
        centered_data = all_features - whole_mean.view(1, -1)
        group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso.fit(centered_data.cpu().numpy().astype(np.float32))
        whole_precision = torch.from_numpy(group_lasso.precision_).float()
        
        return class_mean, precision, whole_mean, whole_precision