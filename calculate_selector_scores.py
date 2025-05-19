import argparse
import os
import torch
import numpy as np

from models.helper import create_model
from scores.helper import get_score_fn
# from data_utils.selector_scores_data_util import get_cov_shift_datasets, get_im1k_val_loader, get_cov_shift_loader, get_im1k_train_loader
from data_utils.helper import get_cov_shift_dataset_names, get_im_train_loader, get_im_val_loader, get_cov_shift_loader, get_im_split_names, get_num_cls


def process_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', default="/home/users/alvin/MCM/datasets", type=str, help='root dir of datasets')
    parser.add_argument('--gpu', default=0, type = int, help='the GPU index to use')
    parser.add_argument('--batch_size', default=512, type=int, help='mini-batch size')
    parser.add_argument('--model_type', default='dfn', choices=['dfn', 'eva'], type=str, help='model type')
    parser.add_argument('--score', default='msp', type=str, 
                        choices=['msp', 'maxlogit', 'energy', 'mds', 'knn', 'rlog', 'delta-mds', 'delta-knn'], help='score options')
    parser.add_argument('--task', type=str, choices=['imagenetv2', 'imagenet-sketch', 'imagenet-c-blur', 
                                                     'imagenet-c-noise', 'imagenet-c-digital',
                                                     'imagenet-c-weather', 'objectnet', 'imagenet-a', 'imagenet-r'])
    parser.add_argument('--k', type=int, default=25, help="kth nearest neigbor distance to take") # only for knn-based scores
    args = parser.parse_args()
    
    if args.score in ['msp', 'maxlogit', 'energy', 'rlog']:
        args.compute_logit_scores = True
    else:
        args.compute_logit_scores = False

    args.n_cls = get_num_cls(args)
    return args


def model_check(args):
    if args.model_type == 'eva':
        assert args.task in ['imagenetv2', 'imagenet-sketch', 'imagenet-c-blur', 'imagenet-c-noise', 
                             'imagenet-c-digital', 'imagenet-c-weather'], "EVA model only supports tasks with full 1K class coverage."
    

def calculate_and_save_selector_scores(args, name, net, score_fn, loader):
    os.makedirs('selector_scores', exist_ok=True)
    selector_scores_path = f"selector_scores/{args.model_type}_{args.score}_{name}.npy"
    if os.path.isfile(selector_scores_path):
        print(f"Selector scores already calculated for {name}.")
        return
    
    print(f"Calculating selector scores for {name} with model {args.model_type} using score {args.score}...")
    selector_scores = score_fn.get_score(net, loader)
    np.save(selector_scores_path, selector_scores)


def main():
    args = process_args()
    model_check(args)
    
    assert torch.cuda.is_available(), "No CUDA device found."
    torch.cuda.set_device(args.gpu)
    os.makedirs('residuals', exist_ok=True)
    
    # setup model
    net = create_model(args)
    
    # setup selector score object
    score_fn = get_score_fn(args)
    im_train_loader = get_im_train_loader(args, net.preprocess, shuffle=False)
    score_fn.setup(net, im_train_loader)
    
    # evaluate for imagenet val set
    im_val_name = get_im_split_names(args)
    im_val_loader = get_im_val_loader(args, net.preprocess, shuffle=False)
    calculate_and_save_selector_scores(args, im_val_name, net, score_fn, im_val_loader)
    
    # evaluate for cov shift datasets
    datasets = get_cov_shift_dataset_names(args)
    for name in datasets:
        cov_shift_loader = get_cov_shift_loader(args, net.preprocess, name, shuffle=False)
        calculate_and_save_selector_scores(args, name, net, score_fn, cov_shift_loader)


if __name__ == "__main__":
    main()