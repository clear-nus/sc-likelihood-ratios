from models.clip import CLIPModel
from models.supervised import EVAModel
from data_utils.imagenet import get_imagenet_labels

def create_model(args):

    if args.model_type == "dfn":
        classnames = get_imagenet_labels(args)
        model = CLIPModel(args.model_type, "ViT-H14", classnames, compute_logit_scores=args.compute_logit_scores)
    elif args.model_type == "eva":
        model = EVAModel()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model