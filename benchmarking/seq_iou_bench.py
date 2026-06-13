import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Callable, Optional
from PIL import Image
from tqdm import tqdm
import json
import re
import sys
import os
import ast
import hashlib
from qwen_vl_utils import process_vision_info
from torch.utils.data import DataLoader

src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from dataset import SimpleDomainNetDataset


# Define transformation matrices
def get_matrix(op):
    if op == "rotate_90":
        return np.array([[0, -1], [1, 0]])
    elif op == "rotate_180":
        return np.array([[-1, 0], [0, -1]])
    elif op == "rotate_270":
        return np.array([[0, 1], [-1, 0]])
    elif op == "horizontal_flip":
        return np.array([[-1, 0], [0, 1]])
    elif op == "vertical_flip":
        return np.array([[1, 0], [0, -1]])
    else:
        return np.eye(2)

# Canonical mapping: matrix -> token
CANONICAL_TOKENS = {
    (1, 0, 0, 1): "geom_id",
    (0, -1, 1, 0): "geom_r90",
    (-1, 0, 0, -1): "geom_r180",
    (0, 1, -1, 0): "geom_r270",
    (-1, 0, 0, 1): "geom_fh",
    (1, 0, 0, -1): "geom_fv",
    (0, -1, -1, 0): "geom_fh_r90",
    (0, 1, 1, 0): "geom_fv_r90"
}

GEOMETRIC_OPS = {"rotate_90", "rotate_180", "rotate_270", "horizontal_flip", "vertical_flip"}

def normalize_geometric_subsequence(geom_ops):
    """Deterministically map any sequence to canonical token using matrix composition.
    Returns None if the result is identity (geom_id).
    """
    if not geom_ops:
        return None
    
    M = np.eye(2)
    for op in geom_ops:
        M = get_matrix(op) @ M
    
    M = np.round(M).astype(int)
    key = tuple(M.flatten())
    
    if key in CANONICAL_TOKENS:
        token = CANONICAL_TOKENS[key]
        if token == "geom_id":
            return None
        return token
    else:
        raise ValueError(f"Unexpected matrix: {M}")


def canonicalize_sequence(seq):
    """
    Extract geometric ops (preserving order), normalize to canonical token (if non-identity),
    and combine with non-geometric ops as a set.
    """
    geom_ops = [op for op in seq if op in GEOMETRIC_OPS]
    other_ops = [op for op in seq if op not in GEOMETRIC_OPS]

    canonical_geom = normalize_geometric_subsequence(geom_ops)
    if canonical_geom is not None:
        return set(other_ops + [canonical_geom])
    else:
        return set(other_ops)


class LengthWiseAccuracyBenchmark:
    def __init__(
        self,
        model,
        dataset_root: str,
        json_path: str,
        preprocess: Callable[[Image.Image], torch.Tensor],
        transformer,
        tokenizer,
        config,
        device: torch.device,
        n_samples_per_length: int = 200,
        max_gen_len: int = 15,
        seed: int = 2026
    ):
        self.model = model
        self.dataset_root = Path(dataset_root)
        self.json_path = json_path
        self.preprocess = preprocess
        self.config = config
        self.device = device
        self.n_samples_per_length = n_samples_per_length
        self.max_gen_len = max_gen_len
        self.seed = seed

        self.transformer = transformer
        self.tokenizer = tokenizer

        self.all_image_paths = []
        self._load_all_paths()

    def _load_all_paths(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    
        seen_paths = set()
        for entry in metadata:
            img1_class = entry.get("image_1_class", "")
            img2_class = entry.get("image_2_class", "")
            if img1_class == "monotone" or img2_class == "monotone":
                continue
    
            batch_name = entry["batch_name"]
            for img_key in ["image_1", "image_2"]:
                img_name = entry[img_key]
                img_path = self.dataset_root / batch_name / "dataset" / img_name
                if img_path.is_file() and str(img_path) not in seen_paths:
                    self.all_image_paths.append(img_path)
                    seen_paths.add(str(img_path))

    def _set_image_seed(self, image: Image.Image) -> None:
        img_hash = hash(image.tobytes()) % (2**32)
        random.seed(img_hash)
        np.random.seed(img_hash % (2**32))
        torch.manual_seed(img_hash)

    def run(self, model_name: str, output_path: str, verbose: bool = True) -> Dict[int, Dict[str, float]]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
    
        results = {}
    
        self.model.eval()
        with torch.no_grad():
            n = min(self.n_samples_per_length, len(self.all_image_paths))
            sampled_paths = random.sample(self.all_image_paths, n)
            for k in range(1, 6): # k = 1 to 5
    
                total_intersection = 0
                total_union = 0
                correct_tokens_k = 0
                total_tokens_k = 0
                count = 0
    
                for img_path in tqdm(sampled_paths, desc=f"k={k}", leave=False):
                    try:
                        orig = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
    
                    self._set_image_seed(orig)
                    transformed, target_seq = self.transformer.transform_by_length(orig, k=k)
    
                    input_ids, targets = self.tokenizer.encode(
                        target_seq,
                        add_special_tokens=True,
                        return_targets=True
                    )
    
                    orig_tensor = self.preprocess(orig).unsqueeze(0).to(self.device)
                    trans_tensor = self.preprocess(transformed).unsqueeze(0).to(self.device)
    
                    # === 1. Macro Jaccard Index ===
                    generated_ids = self.model.generate(
                        image_batch_1=orig_tensor,
                        image_batch_2=trans_tensor,
                        max_new_tokens=self.max_gen_len - 1,
                        do_sample=False
                    ).squeeze(0)
    
                    pred_tokens = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    gt_tokens = self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
                    pred_canonical = canonicalize_sequence(pred_tokens)
                    gt_canonical = canonicalize_sequence(gt_tokens)
    
                    intersection = len(pred_canonical & gt_canonical)
                    union = len(pred_canonical | gt_canonical)
                    total_intersection += intersection
                    total_union += union
    
                    # === 2. Token Accuracy (teacher forcing) ===
                    input_tensor = input_ids.unsqueeze(0).to(self.device)
                    target_tensor = targets.unsqueeze(0).to(self.device)
    
                    logits, _ = self.model(
                        orig_tensor,
                        trans_tensor,
                        idx=input_tensor,
                    )
    
                    pred_ids = logits.argmax(dim=-1)
                    pad_token_id = self.tokenizer.vocab["[PAD]"]
                    mask = (target_tensor != pad_token_id)
                    correct = (pred_ids == target_tensor) & mask
    
                    correct_tokens_k += correct.sum().item()
                    total_tokens_k += mask.sum().item()
    
                    count += 1
    
                macro_jaccard = total_intersection / total_union if total_union > 0 else 0.0
                token_acc = correct_tokens_k / total_tokens_k if total_tokens_k > 0 else 0.0
    
                results[k] = {
                    "jaccard": macro_jaccard,
                    "token_acc": token_acc,
                    "count": count
                }
    
                if verbose:
                    print(f"  k={k}: Jaccard={macro_jaccard:.4f}, TokAcc={token_acc:.4f} (N={count})")
    
        # === Save to JSON ===
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
        if output_path_obj.exists():
            with open(output_path_obj, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    all_data = {}
        else:
            all_data = {}
    
        metadata = {
            "n_samples_per_length": self.n_samples_per_length,
            "seed": self.seed,
            "max_gen_len": self.max_gen_len,
            "metrics": ["canonical_seq_acc", "token_acc"]
        }
    
        all_data[model_name] = {
            "metadata": metadata,
            "results": results
        }
    
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
    
        if verbose:
            print(f"\nSaved length-wise accuracy for '{model_name}' to '{output_path}'")
    
        return results


class QwenLengthWiseBenchmark:
    def __init__(
        self,
        model,
        processor,
        dataset_root: str,
        json_path: str,
        transformer,
        n_samples_per_length: int = 200,
        seed: int = 2026
    ):
        self.model = model
        self.processor = processor
        self.dataset_root = Path(dataset_root)
        self.json_path = json_path
        self.transformer = transformer
        self.n_samples_per_length = n_samples_per_length
        self.seed = seed

        self.all_image_paths = []
        self._load_all_paths()

        # Define prompt once
        self.prompt = """You are given two images: Image A (original) and Image B (transformed).  
        Your task is to predict the sequence of transformations applied to Image A to obtain Image B, using only the following allowed operations:  
        "noop", "grayscale", "rotate_90", "rotate_180", "rotate_270", "color_jitter", "noise_adding", "jpeg_artefacts", "crop", "horizontal_flip", "vertical_flip".
        
        - The sequence may contain zero, one, or multiple transformations applied in order.  
        - If Image A and Image B are identical, return: ["noop"]
        - If Image B can be obtained by applying a sequence of the allowed transformations (in the correct order), return that sequence as a JSON list, e.g.: ["color_jitter", "noise_adding", "rotate_270", "horizontal_flip"]  
        - If the transformation from Image A to Image B requires any operation not in the allowed list (e.g., blur, resize, perspective distortion, custom warping, etc.), or if the images are unrelated, return an empty list: []  
        
        Output only the JSON list. Do not add explanations, comments, or extra text."""

    def _load_all_paths(self) -> None:
        with open(self.json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    
        seen_paths = set()
        for entry in metadata:
            img1_class = entry.get("image_1_class", "")
            img2_class = entry.get("image_2_class", "")
            if img1_class == "monotone" or img2_class == "monotone":
                continue
    
            batch_name = entry["batch_name"]
            for img_key in ["image_1", "image_2"]:
                img_name = entry[img_key]
                img_path = self.dataset_root / batch_name / "dataset" / img_name
                if img_path.is_file() and str(img_path) not in seen_paths:
                    self.all_image_paths.append(img_path)
                    seen_paths.add(str(img_path))

    def _set_image_seed(self, image: Image.Image) -> None:
        img_hash = hash(image.tobytes()) % (2**32)
        random.seed(img_hash)
        np.random.seed(img_hash % (2**32))
        torch.manual_seed(img_hash)

    def _parse_qwen_output(self, output_str: str) -> List[str]:
        """Parse Qwen's output string into a list of tokens."""
        try:
            # Extract JSON-like list using regex
            match = re.search(r'\[.*\]', output_str)
            if match:
                list_str = match.group(0)
                parsed = ast.literal_eval(list_str)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
        except Exception:
            pass
        return []  # fallback to empty list on error

    def run(self, model_name: str, output_path: str, verbose: bool = True) -> Dict[int, Dict[str, float]]:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
    
        results = {}
    
        self.model.eval()
        with torch.no_grad():
            n = min(self.n_samples_per_length, len(self.all_image_paths))
            sampled_paths = random.sample(self.all_image_paths, n)
            for k in range(1, 6): # k = 1 to 5
    
                total_intersection = 0
                total_union = 0
                count = 0
    
                for img_path in tqdm(sampled_paths, desc=f"k={k}", leave=False):
                    try:
                        orig = Image.open(img_path).convert("RGB")
                    except Exception:
                        continue
    
                    self._set_image_seed(orig)
                    transformed, target_seq = self.transformer.transform_by_length(orig, k=k)
    
                    # Prepare Qwen input
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": orig},
                                {"type": "image", "image": transformed},
                                {"type": "text", "text": self.prompt},
                            ],
                        }
                    ]
    
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    inputs = inputs.to("cuda")
    
                    # Inference
                    generated_ids = self.model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]
    
                    # Parse prediction
                    pred_tokens = self._parse_qwen_output(output_text)
                    gt_tokens = target_seq  # already a list of strings
    
                    # Canonicalize both
                    pred_canonical = canonicalize_sequence(pred_tokens)
                    gt_canonical = canonicalize_sequence(gt_tokens)
    
                    intersection = len(pred_canonical & gt_canonical)
                    union = len(pred_canonical | gt_canonical)
                    total_intersection += intersection
                    total_union += union
                    count += 1
    
                macro_jaccard = total_intersection / total_union if total_union > 0 else 0.0
    
                results[k] = {
                    "jaccard": macro_jaccard,
                    "count": count
                }
    
                if verbose:
                    print(f"  k={k}: Jaccard={macro_jaccard:.4f} (N={count})")
    
        # === Save to JSON ===
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
        if output_path_obj.exists():
            with open(output_path_obj, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    all_data = {}
        else:
            all_data = {}
    
        metadata = {
            "n_samples_per_length": self.n_samples_per_length,
            "seed": self.seed,
            "metrics": ["jaccard"]
        }
    
        all_data[model_name] = {
            "metadata": metadata,
            "results": results
        }
    
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
    
        if verbose:
            print(f"\nSaved Qwen length-wise accuracy for '{model_name}' to '{output_path}'")
    
        return results


class DomainNetFullBenchmark:
    def __init__(
        self,
        model,
        dataset_root: str,
        transformer,
        tokenizer=None,
        preprocess: Optional[Callable] = None,
        device: torch.device = torch.device("cpu"),
        transform_prob: float = 0.3,
        max_gen_len: int = 15,
        val_size: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 1,
        model_type: str = "custom"
    ):
        self.model = model
        self.dataset_root = dataset_root
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.device = device
        self.transform_prob = transform_prob
        self.max_gen_len = max_gen_len
        self.batch_size = batch_size
        self.model_type = model_type

        self.val_dataset = SimpleDomainNetDataset(
            data_dir=dataset_root,
            split="val",
            val_size=val_size,
            random_seed=random_seed
        )

        self.dataloader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda x: x
        )

    def _set_image_seed(self, image: Image.Image) -> None:
        img_hash = int(hashlib.sha256(image.tobytes()).hexdigest(), 16) % (2**32)
        random.seed(img_hash)
        np.random.seed(img_hash % (2**32))
        torch.manual_seed(img_hash)

    def _parse_qwen_output(self, output_str: str) -> List[str]:
        try:
            match = re.search(r'\[.*\]', output_str)
            if match:
                parsed = ast.literal_eval(match.group(0))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
        except Exception:
            pass
        return []

    def run(self, model_name: str, output_path: str, verbose: bool = True):
        self.model.eval()
        total_intersection = 0
        total_union = 0
        count = 0
        domain_stats = {}

        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="DomainNet val"):
                for orig_pil, domain in batch:
                    try:
                        orig = orig_pil.convert("RGB")
                    except Exception:
                        continue

                    if domain not in domain_stats:
                        domain_stats[domain] = {"intersection": 0, "union": 0, "count": 0}

                    self._set_image_seed(orig)
                    transformed, target_seq = self.transformer.transform(orig, p=self.transform_prob)

                    if self.model_type == "custom":
                        if self.preprocess is None:
                            raise ValueError("preprocess must be provided for custom models")
                        orig_tensor = self.preprocess(orig).unsqueeze(0).to(self.device)
                        trans_tensor = self.preprocess(transformed).unsqueeze(0).to(self.device)
                        generated_ids = self.model.generate(
                            image_batch_1=orig_tensor,
                            image_batch_2=trans_tensor,
                            max_new_tokens=self.max_gen_len - 1,
                            do_sample=False
                        ).squeeze(0)
                        pred_tokens = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                        gt_tokens = target_seq

                    elif self.model_type == "qwen":
                        prompt = """You are given two images: Image A (original) and Image B (transformed).  
                        Your task is to predict the sequence of transformations applied to Image A to obtain Image B, using only the following allowed operations:  
                        "noop", "grayscale", "rotate_90", "rotate_180", "rotate_270", "color_jitter", "noise_adding", "jpeg_artefacts", "crop", "horizontal_flip", "vertical_flip".
                        
                        - The sequence may contain zero, one, or multiple transformations applied in order.  
                        - If Image A and Image B are identical, return: ["noop"]
                        - If Image B can be obtained by applying a sequence of the allowed transformations (in the correct order), return that sequence as a JSON list, e.g.: ["color_jitter", "noise_adding", "rotate_270", "horizontal_flip"]  
                        - If the transformation from Image A to Image B requires any operation not in the allowed list (e.g., blur, resize, perspective distortion, etc.), or if the images are unrelated, return an empty list: []  
                        
                        Output only the JSON list. Do not add explanations, comments, or extra text."""
                    
                        try:
                            # --- Подготовка входов ---
                            messages = [{
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": orig},
                                    {"type": "image", "image": transformed},
                                    {"type": "text", "text": prompt},
                                ],
                            }]
                    
                            text = self.preprocess.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                            image_inputs, _ = process_vision_info(messages)
                            inputs = self.preprocess(
                                text=[text],
                                images=image_inputs,
                                padding=True,
                                return_tensors="pt"
                            )
                    
                            inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                            generated_ids = self.model.generate(
                                **inputs,
                                max_new_tokens=128,
                                do_sample=False,
                                pad_token_id=self.preprocess.tokenizer.pad_token_id,
                                eos_token_id=self.preprocess.tokenizer.eos_token_id,
                                use_cache=False
                            )
                    
                            input_len = inputs["input_ids"].size(1)
                            out_ids = generated_ids[0, input_len:].cpu().detach()
                            output_text = self.preprocess.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    
                            pred_tokens = self._parse_qwen_output(output_text)
                            gt_tokens = target_seq
                    
                            del inputs, generated_ids, out_ids, image_inputs
                            torch.cuda.empty_cache()
                    
                        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                            if "out of memory" in str(e).lower():
                                if verbose:
                                    print(f"\nOOM on pair, skipping... (domain={domain})")
                                continue
                            else:
                                raise e
                    else:
                        raise ValueError(f"Unknown model_type: {self.model_type}")

                    pred_canonical = canonicalize_sequence(pred_tokens)
                    gt_canonical = canonicalize_sequence(gt_tokens)
                    intersection = len(pred_canonical & gt_canonical)
                    union = len(pred_canonical | gt_canonical)

                    total_intersection += intersection
                    total_union += union
                    domain_stats[domain]["intersection"] += intersection
                    domain_stats[domain]["union"] += union
                    domain_stats[domain]["count"] += 1
                    count += 1

        macro_jaccard = total_intersection / total_union if total_union > 0 else 0.0
        domain_results = {}
        for domain, stats in domain_stats.items():
            jaccard = stats["intersection"] / stats["union"] if stats["union"] > 0 else 0.0
            domain_results[domain] = {"jaccard": jaccard, "count": stats["count"]}

        results = {
            "overall": {"jaccard": macro_jaccard, "count": count},
            "per_domain": domain_results
        }

        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        all_data = {}
        if output_path_obj.exists():
            with open(output_path_obj, "r", encoding="utf-8") as f:
                try:
                    all_data = json.load(f)
                except json.JSONDecodeError:
                    pass

        metadata = {
            "dataset": "DomainNet",
            "split": "val",
            "transform_prob": self.transform_prob,
            "total_samples": count,
            "domains": list(domain_results.keys()),
            "metrics": ["jaccard"]
        }

        all_data[model_name] = {"metadata": metadata, "results": results}
        with open(output_path_obj, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)

        if verbose:
            print(f"\nDomainNet full validation benchmark completed for '{model_name}'")
            print(f"   Transform probability (p): {self.transform_prob}")
            print(f"   Overall Jaccard = {macro_jaccard:.4f} (N={count})")
            print("\nPer-domain results:")
            for domain, res in domain_results.items():
                print(f"   {domain}: Jaccard = {res['jaccard']:.4f} (N={res['count']})")
            print(f"\nSaved to: {output_path}")

        return results