import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from .imagenet import FilteredImageFolder
from .objectnet import ObjectNet
from imagenetv2_pytorch import ImageNetV2Dataset


def get_num_cls(args):
    NUM_CLS_DICT = {
        'imagenet-sketch': 1000,
        'imagenetv2': 1000,
        'objectnet': 113,
        'imagenet-a': 200,
        'imagenet-r': 200,
        'imagenet-c-blur': 1000,
        'imagenet-c-noise': 1000,
        'imagenet-c-digital': 1000,
        'imagenet-c-weather': 1000
    }
    n_cls = NUM_CLS_DICT[args.task]
    return n_cls


def get_im_split_names(args):
    
    if args.task == 'imagenet-a':
        return 'imagenet1k-a-200-split'
    elif args.task == 'imagenet-r':
        return 'imagenet1k-r-200-split'
    elif args.task == 'objectnet':
        return 'imagenet1k-objectnet-split'
    else:
        return 'imagenet1k'


def get_cov_shift_dataset_names(args):
    
    if args.task == 'imagenet-a':
        datasets = ['imagenet-a']
    elif args.task == 'imagenet-r':
        datasets = ['imagenet-r']
    elif args.task == 'objectnet':
        datasets = ['objectnet']
    elif args.task == 'imagenet-sketch':
        datasets = ['imagenet-sketch']
    elif args.task == 'imagenet-c-blur':
        datasets = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
    elif args.task == 'imagenet-c-digital':
        datasets = ['contrast', 'elastic_transform', 'jpeg_compression', 'pixelate']
    elif args.task == 'imagenet-c-noise':
        datasets = ['gaussian_noise', 'impulse_noise', 'shot_noise']
    elif args.task == 'imagenet-c-weather':
        datasets = ['brightness', 'fog', 'frost', 'snow']
    elif args.task == 'imagenetv2':
        datasets = ['imagenetv2']
    elif args.task == 'imagenet1k':
        datasets = []
    return datasets


def get_im_train_loader(args, preprocess, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
            
    if args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                      transform=preprocess)
    elif args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                      transform=preprocess)
    elif args.task == 'objectnet':
        ds = FilteredImageFolder(os.path.join(args.root_dir, "imagenet/train"), 
                                      os.path.join(args.root_dir, 'objectnet-classes.txt'),
                                      transform=preprocess)
    else:
        ds = ImageFolder(os.path.join(args.root_dir, "imagenet/train"), transform=preprocess)
    
    train_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return train_loader


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
    
    elif args.task == 'objectnet':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet/val'), 
                                 os.path.join(args.root_dir, 'objectnet-classes.txt'),
                                 transform=preprocess)

    else:
        ds = ImageFolder(os.path.join(args.root_dir, "imagenet/val"), transform=preprocess)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, **kwargs)
    return loader


def get_cov_shift_loader(args, preprocess, domain, shuffle=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    
    if args.task == 'imagenet-r':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-r'), 
                                 os.path.join(args.root_dir, 'imagenet-r-classes.txt'),
                                 transform=preprocess)
    elif args.task == 'imagenet-a':
        ds = FilteredImageFolder(os.path.join(args.root_dir, 'imagenet-a'), 
                                 os.path.join(args.root_dir, 'imagenet-a-classes.txt'),
                                 transform=preprocess)

    elif args.task == 'objectnet':
        ds = ObjectNet(args.root_dir, train=False, transform=preprocess)

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

