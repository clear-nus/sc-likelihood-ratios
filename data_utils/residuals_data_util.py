"""
Helper functions to load data for classification (calculating residuals).
"""

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from imagenet import FilteredImageFolder
from objectnet import ObjectNet
from imagenetv2_pytorch import ImageNetV2Dataset


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


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


def get_val_loader(args, domain, preprocess, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    if args.task == 'imagenet1k':
        ds = ImageFolder(os.path.join(args.root_dir, "imagenet/val"), transform=preprocess)

    elif args.task == 'imagenet1k-a-200-split':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                 transform=preprocess)
    
    elif args.task == 'imagenet1k-r-200-split':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                 transform=preprocess)

    elif args.task == 'imagenet1k-objectnet-113-split':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'objectnet-113-classes.txt'),
                                 transform=preprocess)

    elif args.task == 'objectnet-113':
        ds = ObjectNet(args.root_dir, train=False, transform=preprocess)
    
    elif args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-r'), 
                                 os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                 transform=preprocess)
    elif args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-a'), 
                                 os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                 transform=preprocess)
        
    elif args.task == 'imagenetv2':
        ds = ImageNetV2Dataset(location=args.root_dir, variant="matched-frequency", transform=preprocess)
        
    elif args.task == 'imagenet-sketch':
        ds = ImageFolder(os.path.join(args.root_dir, "sketch"), transform=preprocess)
        
    elif args.task == 'imagenet-c-blur':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/blur/{domain}/5"), transform=preprocess)

    elif args.task == 'imagenet-c-digital':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/digital/{domain}/5"), transform=preprocess)
        
    elif args.task == 'imagenet-c-noise':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/noise/{domain}/5"), transform=preprocess)
        
    elif args.task == 'imagenet-c-weather':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/weather/{domain}/5"), transform=preprocess)
        
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return loader


def get_classification_datasets(task):
    
    if task == 'imagenet1k':
        datasets = ['imagenet1k']
    elif task == 'imagenet1k-a-200-split':
        datasets = ['imagenet1k-a-200-split']
    elif task == 'imagenet1k-r-200-split':
        datasets = ['imagenet1k-r-200-split']
    elif task == 'imagenet1k-objectnet-113-split':
        datasets = ['imagenet1k-objectnet-113-split']
    elif task == 'imagenet-a':
        datasets = ['imagenet-a']
    elif task == 'imagenet-r':
        datasets = ['imagenet-r']
    elif task == 'objectnet-113':
        datasets = ['objectnet-113']
    elif task == 'imagenet-sketch':
        datasets = ['imagenet-sketch']
    elif task == 'imagenet-c-blur':
        datasets = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
    elif task == 'imagenet-c-digital':
        datasets = ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate']
    elif task == 'imagenet-c-noise':
        datasets = ['gaussian_noise', 'impulse_noise', 'shot_noise']
    elif task == 'imagenet-c-weather':
        datasets = ['brightness', 'fog', 'frost', 'snow']
    elif task == 'imagenetv2':
        datasets = ['imagenetv2']
    return datasets