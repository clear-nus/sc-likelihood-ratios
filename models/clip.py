"""
Unify all CLIP into a single class
"""

import torch
import torch.nn as nn
import open_clip
import torch
from tqdm import tqdm


class CLIPModel(nn.Module):
    
    def __init__(self, model_type, model_size, classnames, compute_logit_scores=False):
        super().__init__()
        
        assert model_type == "dfn"
        
        ckpt_mapping = {"ViT-H14":("ViT-H-14-378-quickgelu","dfn5b")}
        ckpt = ckpt_mapping[model_size]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(ckpt[0], pretrained=ckpt[1])
        self.tokenizer = open_clip.get_tokenizer(ckpt[0])

        self.model.eval()
        self.model = self.model.cuda()
        
        if compute_logit_scores:
            templates = logit_scores_clip_templates()
        else:
            templates = default_clip_templates()

        self.zeroshot_weights = self.zeroshot_classifier(classnames, templates)
        
    
    def forward(self, x, return_features=False):
        image_features = self.model.encode_image(x).float()
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        logits = image_features_norm @ self.zeroshot_weights
        
        if return_features:
            return logits, image_features
        else:
            return logits


    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames, desc='getting class templates embeddings'):
                texts = [template.format(classname) for template in templates] 
                text_inputs = self.tokenizer(texts)
                class_embeddings = self.model.encode_text(text_inputs.cuda()).float()
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
        return zeroshot_weights


def logit_scores_clip_templates():
    templates = [
        'a real, high-quality, clear and clean photo of a {}'
    ]
    return templates

def default_clip_templates():
    templates = [
        'a bad photo of a {}.',
        'a photo of many {}.',
        'a sculpture of a {}.',
        'a photo of the hard to see {}.',
        'a low resolution photo of the {}.',
        'a rendering of a {}.',
        'graffiti of a {}.',
        'a bad photo of the {}.',
        'a cropped photo of the {}.',
        'a tattoo of a {}.',
        'the embroidered {}.',
        'a photo of a hard to see {}.',
        'a bright photo of a {}.',
        'a photo of a clean {}.',
        'a photo of a dirty {}.',
        'a dark photo of the {}.',
        'a drawing of a {}.',
        'a photo of my {}.',
        'the plastic {}.',
        'a photo of the cool {}.',
        'a close-up photo of a {}.',
        'a black and white photo of the {}.',
        'a painting of the {}.',
        'a painting of a {}.',
        'a pixelated photo of the {}.',
        'a sculpture of the {}.',
        'a bright photo of the {}.',
        'a cropped photo of a {}.',
        'a plastic {}.',
        'a photo of the dirty {}.',
        'a jpeg corrupted photo of a {}.',
        'a blurry photo of the {}.',
        'a photo of the {}.',
        'a good photo of the {}.',
        'a rendering of the {}.',
        'a {} in a video game.',
        'a photo of one {}.',
        'a doodle of a {}.',
        'a close-up photo of the {}.',
        'a photo of a {}.',
        'the origami {}.',
        'the {} in a video game.',
        'a sketch of a {}.',
        'a doodle of the {}.',
        'a origami {}.',
        'a low resolution photo of a {}.',
        'the toy {}.',
        'a rendition of the {}.',
        'a photo of the clean {}.',
        'a photo of a large {}.',
        'a rendition of a {}.',
        'a photo of a nice {}.',
        'a photo of a weird {}.',
        'a blurry photo of a {}.',
        'a cartoon {}.',
        'art of a {}.',
        'a sketch of the {}.',
        'a embroidered {}.',
        'a pixelated photo of a {}.',
        'itap of the {}.',
        'a jpeg corrupted photo of the {}.',
        'a good photo of a {}.',
        'a plushie {}.',
        'a photo of the nice {}.',
        'a photo of the small {}.',
        'a photo of the weird {}.',
        'the cartoon {}.',
        'art of the {}.',
        'a drawing of the {}.',
        'a photo of the large {}.',
        'a black and white photo of a {}.',
        'the plushie {}.',
        'a dark photo of a {}.',
        'itap of a {}.',
        'graffiti of the {}.',
        'a toy {}.',
        'itap of my {}.',
        'a photo of a cool {}.',
        'a photo of a small {}.',
        'a tattoo of the {}.',
    ]
    return templates

