<div align="center">
<h1> From Similarity to Evidence: Image Copy Detection by Predicting Transformation Sequences </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1,2 :email:</sup>, [Kseniia Varlamova](https://github.com/varyxi)<sup>1,2</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>2</sup>

<sup>1</sup> Advacheck, Tallinn, Estonia &nbsp;&nbsp; <sup>2</sup> MIRAI, Moscow, Russia

<sup>:email:</sup> Corresponding author

</div>

## Abstract

Image copy detection asks whether a suspect image is a transformed copy of a reference image, a critical question in academic publishing, media verification, and e-commerce. Existing methods rely on pairwise similarity scores, which offer little interpretability and confuse visual similarity with derivability. We reformulate the problem as *evidential image matching*: the model predicts the sequence of operations from a fixed vocabulary that derives the suspect image from the reference image. An empty sequence means that no derivation exists. We propose an encoder-decoder model that recovers these sequences. We also introduce the Canonical Jaccard Index, a reconstruction metric that respects the algebraic structure of the dihedral group $D_4$ and scores non-geometric operations as an unordered set. On DomainNet and a curated multi-domain negative dataset, the models fine-tuned on hard negatives reach a lower false positive rate on content-similar negatives than similarity-based baselines. Their recall stays on par with these baselines on every operation except crops, alone or in a composite. It also exceeds that of a widely used copy-detection descriptor on rotations, vertical flips, and composite sequences. A threshold sweep of that descriptor shows that no single similarity threshold gives both a low false positive rate and high recall. Our models need no threshold tuning. They also recover the applied operations, unlike a zero-shot vision-language model. The predicted sequence serves both as the decision and as a human-readable evidence trail that explains how the suspect image was obtained.

## Models

The paper studies an encoder–decoder that maps an image pair to a transformation sequence. Two encoder backbones and one contrastive baseline are provided:

| Model | Encoder | Config |
| --- | --- | --- |
| EffNet | EfficientNet-B3 | [`configs/train_config_effnet.yaml`](configs/train_config_effnet.yaml) |
| ViT | ViT-B/16 | [`configs/train_config_vit.yaml`](configs/train_config_vit.yaml) |
| SiamNet | EfficientNet (contrastive baseline) | [`configs/train_config_siamnet.yaml`](configs/train_config_siamnet.yaml) |

Each config holds both the **pretrain** (DomainNet) and **fine-tune** (negative pairs) settings; the active values target fine-tuning and the commented lines switch to pretraining.

## Repository Structure

```
.
├── configs/                       # one config per paper model (pretrain + tune)
│   ├── train_config_effnet.yaml
│   ├── train_config_vit.yaml
│   └── train_config_siamnet.yaml
├── scripts/                       # training / tuning entry points
│   ├── run_train.py               # pretrain EffNet/ViT on DomainNet
│   ├── run_tune.py                # fine-tune EffNet/ViT on negative pairs
│   ├── run_train_siamnet.py       # pretrain SiamNet baseline
│   └── run_tune_siamnet.py        # fine-tune SiamNet baseline
├── src/
│   ├── dataset/                   # tokenizer, augmentations, dataset loaders
│   │   ├── tokenizer.py
│   │   ├── augmentation.py
│   │   ├── dataset.py             # DomainNet image pairs
│   │   ├── negative_dataset.py    # curated negative pairs
│   │   └── load_data.py           # DomainNet downloader
│   ├── model/
│   │   ├── decoder.py             # autoregressive transform decoder
│   │   ├── efficientnet_encoder.py
│   │   ├── vit_encoder.py
│   │   ├── model.py               # ImageTransformPredictor (EffNet/ViT)
│   │   └── siamnet.py             # SiamNet baseline
│   ├── train.py / tuning.py       # EffNet/ViT pretrain & tune loops
│   └── train_siamnet.py / tuning_siamnet.py
├── benchmarking/                  # evaluation scripts & notebooks
└── notebooks/                     # experiments, VLM baseline, data filtering
```

## Usage

Install dependencies and run the entry points from the repository root:

```bash
pip install -r requirements.txt

# 1) Pretrain on DomainNet
python -m scripts.run_train --config configs/train_config_effnet.yaml --data_path data

# 2) Fine-tune on negative pairs (after switching the config to its fine-tune block)
python -m scripts.run_tune --config configs/train_config_effnet.yaml --data_path data/negative_pairs
```

Swap `train_config_effnet.yaml` for `train_config_vit.yaml` to use the ViT backbone, or use the `*_siamnet` scripts for the contrastive baseline. Checkpoints and TensorBoard logs are written under `checkpoints/` and `logs/` (configurable in each YAML).
