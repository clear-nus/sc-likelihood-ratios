"""
Helper functions to load data for computing selector scores.
"""

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from imagenet import FilteredImageFolder
from objectnet import ObjectNet
from imagenetv2_pytorch import ImageNetV2Dataset


def get_cov_shift_datasets(task):
    
    if task == 'imagenet-a':
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


def get_im_val_loader(args, preprocess, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    if args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                 transform=preprocess)
    
    elif args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                 transform=preprocess)
    
    elif args.task == 'objectnet-113':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'objectnet-113-classes.txt'),
                                 transform=preprocess)
        
    else:
        ds = ImageFolder(os.path.join(args.root_dir, "imagenet/val"), transform=preprocess)
        
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return loader


def get_cov_shift_loader(args, domain, preprocess, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    if args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-r'), 
                                 os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                 transform=preprocess)
    elif args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-a'), 
                                 os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                 transform=preprocess)

    elif args.task == 'objectnet-113':
        ds = ObjectNet(args.root_dir, train=False, transform=preprocess)

    elif args.task == 'imagenetv2':
        ds = ImageNetV2Dataset(location=args.root_dir, variant="matched-frequency", transform=preprocess)
        
    elif args.task == 'imagenet-sketch':
        ds = ImageFolder(os.path.join(args.root_dir, "sketch"), transform=preprocess)
        
    elif args.task == 'imagenet-c-blur':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/blur/{domain}/{args.corruption_level}"), transform=preprocess)
        
    elif args.task == 'imagenet-c-digital':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/digital/{domain}/{args.corruption_level}"), transform=preprocess)
        
    elif args.task == 'imagenet-c-noise':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/noise/{domain}/{args.corruption_level}"), transform=preprocess)
        
    elif args.task == 'imagenet-c-weather':
        ds = ImageFolder(os.path.join(args.root_dir, f"imagenet-c/weather/{domain}/{args.corruption_level}"), transform=preprocess)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return loader


def get_im1k_train_loader(args, preprocess, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
            
    if args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                      transform=preprocess)
    elif args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                      transform=preprocess)
    elif args.task == 'objectnet-113':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'objectnet-113-classes.txt'),
                                      transform=preprocess)
    else:
        ds = ImageFolder(os.path.join(args.root_dir, "imagenet/train"), transform=preprocess)
    
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return train_loader