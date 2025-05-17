import torch
import numpy as np
from tqdm import tqdm
import sklearn.covariance
import os
from data_utils.helper import get_im_split_names


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


class DeltaMDS:
    def __init__(self, args):
        self.args = args
    
    def setup(self, net, id_train_loader):
        os.makedirs('delta_mds_stats', exist_ok=True)
        name = get_im_split_names(self.args)
        mean_prec_path = f'delta_mds_stats/{self.args.model_type}_{name}.pt'
        
        if not os.path.isfile(mean_prec_path):
            class_mean_correct, precision_correct, class_mean_wrong, precision_wrong = self.get_mean_prec(net, id_train_loader)
            torch.save([class_mean_correct, precision_correct, class_mean_wrong, precision_wrong], mean_prec_path)
        else:
            class_mean_correct, precision_correct, class_mean_wrong, precision_wrong = torch.load(mean_prec_path, map_location='cpu', weights_only=True)
        
        self.class_mean = class_mean_correct.cuda()
        self.precision__correct = precision_correct.cuda()
        self.class_mean_wrong = class_mean_wrong.cuda()
        self.precision_wrong = precision_wrong.cuda()

    @torch.no_grad()
    def get_score(self, net, loader):
        Mahalanobis_score_all = []

        tqdm_object = tqdm(loader, total=len(loader), desc='calculating Delta MDS score')
        for batch_idx, (images, labels) in enumerate(tqdm_object):
            images = images.cuda()
            logits, features = net(images, return_features=True)
            class_scores_correct = torch.zeros((features.shape[0], self.args.n_cls)).cuda()
            class_scores_wrong = torch.zeros((features.shape[0], self.args.n_cls)).cuda()
            
            for c in range(self.args.n_cls):
                
                tensor_correct = features - self.class_mean_correct[c].view(1, -1)
                class_scores_correct[:, c] = -torch.matmul(torch.matmul(tensor_correct, self.precision_correct), tensor_correct.t()).diag()
                
                tensor_wrong = features - self.class_mean_wrong[c].view(1, -1)
                class_scores_wrong[:, c] = -torch.matmul(torch.matmul(tensor_wrong, self.precision_wrong), tensor_wrong.t()).diag()
            
            # if a class has all points classified correctly or wrongly, there will be nans
            # setting to -inf means we ignore this class and the max operation will consider all other classes instead
            class_scores_correct = torch.nan_to_num(class_scores_correct, nan=float('-inf'))
            class_scores_wrong = torch.nan_to_num(class_scores_wrong, nan=float('-inf'))
            
            mds_correct = torch.max(class_scores_correct, dim=1)[0]
            mds_wrong = torch.max(class_scores_wrong, dim=1)[0]
            Mahalanobis_score = mds_correct - mds_wrong 
            Mahalanobis_score_all.extend(Mahalanobis_score.cpu().numpy())

        return np.asarray(Mahalanobis_score_all, dtype=np.float32)

    @torch.no_grad()
    def get_mean_prec(self, net, loader):
        all_features_correct = []
        all_labels_correct = []
        all_features_wrong = []
        all_labels_wrong = []
    
        for idx, (images, labels) in enumerate(tqdm(loader, desc="calculating MDS stats")):
            images, labels = images.cuda(), labels.cuda()
            logits, features = net(images, return_features=True)
            residuals_top1, _ = residuals_per_sample(logits, labels, topk=(1, 5))
            residuals_top1 = torch.tensor(residuals_top1)
        
            all_labels_correct.append(labels[residuals_top1.eq(0)])
            all_features_correct.append(features[residuals_top1.eq(0)])
            all_labels_wrong.append(labels[residuals_top1.eq(1)])
            all_features_wrong.append(features[residuals_top1.eq(1)])
        
        all_labels_correct = torch.cat(all_labels_correct).cpu()
        all_labels_wrong = torch.cat(all_labels_wrong).cpu()
        all_features_correct = torch.cat(all_features_correct).cpu()
        all_features_wrong = torch.cat(all_features_wrong).cpu()
        
        # compute class-conditional statistics
        class_mean_correct = []
        centered_data_correct = []
        
        for c in range(self.args.n_cls):
            class_samples_correct = all_features_correct[all_labels_correct.eq(c)].data
            class_mean_correct.append(class_samples_correct.mean(0))
            centered_data_correct.append(class_samples_correct - class_mean_correct[c].view(1, -1))

        class_mean_correct = torch.stack(class_mean_correct)  # shape [number of classes, feature dim]
        group_lasso_correct = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso_correct.fit(torch.cat(centered_data_correct).cpu().numpy().astype(np.float32))
        precision_correct = torch.from_numpy(group_lasso_correct.precision_).float()
        
        class_mean_wrong = []
        centered_data_wrong = []
        
        for c in range(self.args.n_cls):
            class_samples_wrong = all_features_wrong[all_labels_wrong.eq(c)].data
            class_mean_wrong.append(class_samples_wrong.mean(0))
            centered_data_wrong.append(class_samples_wrong - class_mean_wrong[c].view(1, -1))

        class_mean_wrong = torch.stack(class_mean_wrong)
        group_lasso_wrong = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
        group_lasso_wrong.fit(torch.cat(centered_data_wrong).cpu().numpy().astype(np.float32))
        precision_wrong = torch.from_numpy(group_lasso_wrong.precision_).float()
        
        return class_mean_correct, precision_correct, class_mean_wrong, precision_wrong