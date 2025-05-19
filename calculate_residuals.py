import argparse
import os
import torch
from tqdm import tqdm
import numpy as np

from models.helper import create_model
# from data_utils.residuals_data_util import get_classification_datasets, get_val_loader, residuals_per_sample
from data_utils.helper import get_im_split_names, get_cov_shift_dataset_names, get_im_val_loader, get_cov_shift_loader


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


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default="/home/users/alvin/MCM/datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU index to use')
    parser.add_argument('--batch_size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--model_type', default='dfn', choices=['dfn', 'eva'], type=str, help='model type')
    parser.add_argument('--task', type=str, default='imagenet1k',
                        choices=['imagenetv2', 'imagenet-sketch', 'imagenet-c-blur', 'imagenet-c-noise', 'imagenet-c-digital',
                                 'imagenet-c-weather', 'objectnet', 'imagenet-a', 'imagenet-r'])
    
    args = parser.parse_args()
    args.compute_logit_scores = False
    return args


def model_check(args):
    if args.model_type == 'eva':
        assert args.task in ['imagenetv2', 'imagenet-sketch', 'imagenet-c-blur', 'imagenet-c-noise', 
                             'imagenet-c-digital', 'imagenet-c-weather'], "EVA model only supports tasks with full 1K class coverage."
        

def calculate_and_save_residuals(args, name, net, loader):
    residuals_path = f"residuals/{args.model_type}_{name}.npy"
    if os.path.isfile(residuals_path):
        print(f"Residuals already calculated for {name}.")
        return
    
    residuals = []
    
    print(f"Calculating residuals for {name} with model {args.model_type}...")
    tqdm_object = tqdm(loader, total=len(loader), desc=f'current task: {name}')
    with torch.no_grad():
        for batch_idx, (images, true_label) in enumerate(tqdm_object):

            images, true_label = images.cuda(), true_label.cuda()
            logits = net(images, return_features=False).float()
            
            residuals_top1, _ = residuals_per_sample(logits, true_label, topk=(1, 5))
            residuals.extend(residuals_top1)
    
    np.save(residuals_path, residuals)
    return


def main():
    
    args = process_args()
    model_check(args)
    
    assert torch.cuda.is_available(), "No CUDA device found."
    torch.cuda.set_device(args.gpu)
    os.makedirs('residuals', exist_ok=True)
    
    net = create_model(args).cuda()
    
    # loop through imagenet val set corresponding to the task
    im_val_name = get_im_split_names(args)
    im_val_loader = get_im_val_loader(args, net.preprocess, shuffle=False)
    calculate_and_save_residuals(args, im_val_name, net, im_val_loader)
    
    # Loop through cov shift datasets. Will not excute if args.task == imagenet1k 
    cov_shift_dataset_names = get_cov_shift_dataset_names(args)
    for name in cov_shift_dataset_names:
        cov_shift_loader = get_cov_shift_loader(args, net.preprocess, name, shuffle=False)
        calculate_and_save_residuals(args, name, net, cov_shift_loader)
        

if __name__ == "__main__":
    main()