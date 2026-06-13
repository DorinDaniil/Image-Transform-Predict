<div align="center">
<h1> Evidential Image Matching: Predicting Transformation Sequences to Derive one Image from Another </h1>

[Daniil Dorin](https://github.com/DorinDaniil)<sup>1,2 :email:</sup>, [Kseniia Varlamova](https://github.com/varyxi)<sup>1,2</sup>, [Andrey Grabovoy](https://github.com/andriygav)<sup>2</sup>

<sup>1</sup> Advacheck, Tallinn, Estonia &nbsp;&nbsp; <sup>2</sup> MIRAI, Moscow, Russia

<sup>:email:</sup> Corresponding author

</div>

## Abstract

Detecting image plagiarism and near-duplicate content remains a critical challenge in academic publishing, media verification, and e-commerce. Existing methods typically rely on pairwise similarity scores, which provide limited interpretability and often struggle to distinguish visual similarity from true transformational derivability.
To address this limitation, we reformulate the problem as *evidential image matching*: given a reference image and a suspect image, the model predicts the sequence of transformations that derives one image from the other. An empty sequence indicates non-plagiarism. We propose an encoder-decoder architecture trained to recover transformation sequences from a predefined vocabulary. We further introduce the Canonical Jaccard Index, a reconstruction metric that accounts for equivalent transformation sequences by respecting the algebraic structure of the dihedral group $D_4$ and the permutation invariance of commutative operations.
Experiments on DomainNet and a curated multi-domain negative dataset show that the proposed approach substantially outperforms similarity-based baselines and a strong zero-shot vision-language model in both plagiarism detection and transformation reconstruction. In addition to improved accuracy, the model provides a human-readable evidence trail explaining its decisions.

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
