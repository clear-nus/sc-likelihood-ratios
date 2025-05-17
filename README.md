# Repository for "Know When to Abstain: Optimal Selective Classification with Likelihood Ratios"

## How to run
Install requirements.
```
conda create --name sc-likelihood-ratios python=3.10
conda activate sc-likelihood-ratios
pip install -r requirements.txt
```

See below on how to format the datasets for ImageNet and their covariate shifts.

We will show an example of how to compute the AURC/NAURC for ImageNet-Sketch with a linear combination of Delta-MDS and RLog with DFN CLIP. 
See the arguments in code for more options.
1. First compute the residuals (0/1 loss values).
    ```
    python calculate_residuals.py --task imagenet-sketch --model_type dfn --root-dir datasets --gpu 0 --batch_size 256
    ```
    This will save the residuals in `./residuals`.

2. Compute selector scores.
    ```
    python calculate_selector_scores.py --task imagenet-sketch --model_type dfn --root-dir datasets --gpu 0 --batch_size 256 --score delta-mds
    python calculate_selector_scores.py --task imagenet-sketch --model_type dfn --root-dir datasets --gpu 0 --batch_size 256 --score rlog
    ```
    This will save the MDS statistics (mean, covariance) in `./delta_mds_stats` and the selector scores in `./selector_scores`.

3. Compute AURC/NAURC
    ```
    python calculate_risk_coverate_curve.py --task imagenet-sketch --model_type dfn --root-dir datasets --gpu 0 --batch_size 256 \
                                            --score1 delta-mds --score2 rlog --lam 10000
    ```
    The results will be printed to screen.

## Dataset Preparation
Datasets go into the `datasets` folder. It should be structured as follows:

```
datasets
├── imagenet
├── imagenet-a
├── imagenet-r
├── ImageNetV2-matched-frequenxy
├── objectnet-1.0
├── sketch
├── imagenet-a-classes.txt
├── imagenet-r-classes.txt
└── objectnet-113-classes.txt
```
The `.txt` files and general structure are already included. Inside each folder, the datasets should be prepared in the following formats. Most datasets are already formatted as such when downloaded from their official source.
```

imagenet
├── train
    ├── n01440764
    ├── ...
└── val
    ├── n01440764
    ├── ... 
```
Refer to https://github.com/DoranLyong/ImageNet2012-download on how to obtain ImageNet1K in this format.

```
imagenet-a
├── n01498041
├── ...
```

```
imagenet-c
├── blur
    └── 5
       ├── n01440764
       ├── ...
├── digital
├── noise
└── weather
```
We evaluate on level 5 corruption strength for imagenet-c.

```
imagenet-r
├── n01443537
├── ...
```

```
ImageNetV2-matched-frequency
├── 0 
├── 1
├── ...
```
The ImageNetV2 folder structure uses a numbering system that works with the ImageNetV2 loader. Refer to official ImageNetV2 download repository.

```
objectnet-1.0
├── images
    ├── air_freshener
    ├── alarm_clock
    ├── ...
└── mapping
```
Refer to the official ObjectNet download repository.

```
sketch
├── n01440764
├── ...
```