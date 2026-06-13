#!/usr/bin/env python3
"""
Extract component weights from a full ImageTransformPlagiarismPredictor
checkpoint and save them as individual state-dict files.

Mapping (training model -> standalone component):

    image_pair_encoder.image_encoder.*  ->  encoder  (ImageEncoder)
    image_pair_encoder.fuser.*          ->  fuser    (ImageFuser)
    match_head.*                        ->  match_head (BinaryMatchHead)
    transform_decoder.*                 ->  decoder  (TransformationDecoder)
    projection_head.*                   ->  (skipped, training-only)

Usage (run from the repository root):
    # CLI: extract from checkpoint -> 4 files
    python -m plagiarism.export_plagiarism_weights \\
        --checkpoint checkpoints/plagiarism_pretrain/checkpoint_epoch_30.pth \\
        --output_dir weights/plagiarism/

    # Python: load directly into standalone model
    from plagiarism.model_plagiarism_torch import PlagiarismDetectionModelV2
    model = PlagiarismDetectionModelV2()
    model.load_from_full_checkpoint("checkpoints/.../checkpoint_epoch_30.pth")
"""

import argparse
import os
from collections import OrderedDict
from typing import Dict

import torch

# Prefix mapping: training model prefix -> component name
PREFIX_MAP: Dict[str, str] = {
    "image_pair_encoder.image_encoder.": "encoder",
    "image_pair_encoder.fuser.": "fuser",
    "match_head.": "match_head",
    "transform_decoder.": "decoder",
}

# Skipped prefixes (training-only, not needed for inference)
SKIP_PREFIXES = ("projection_head.",)


def split_state_dict(
    full_sd: Dict[str, torch.Tensor],
) -> Dict[str, OrderedDict]:
    """Split a full model state_dict into per-component state_dicts.

    Args:
        full_sd: state_dict from ImageTransformPlagiarismPredictor.

    Returns:
        Dict with keys ``encoder``, ``fuser``, ``match_head``, ``decoder``.
        Each value is an OrderedDict ready for ``component.load_state_dict()``.
    """
    parts: Dict[str, OrderedDict] = {
        name: OrderedDict() for name in PREFIX_MAP.values()
    }
    unmatched = []

    for key, value in full_sd.items():
        if any(key.startswith(sp) for sp in SKIP_PREFIXES):
            continue

        matched = False
        for prefix, component_name in PREFIX_MAP.items():
            if key.startswith(prefix):
                stripped_key = key[len(prefix):]
                parts[component_name][stripped_key] = value
                matched = True
                break

        if not matched:
            unmatched.append(key)

    if unmatched:
        print(f"[warn] {len(unmatched)} unmatched keys (skipped):")
        for k in unmatched[:10]:
            print(f"  {k}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")

    for name, sd in parts.items():
        print(f"  {name:12s}: {len(sd)} params")

    return parts


def export_components(
    checkpoint_path: str,
    output_dir: str,
) -> Dict[str, str]:
    """Load a full checkpoint and save each component as a separate file.

    Args:
        checkpoint_path: Path to training checkpoint (.pth).
        output_dir: Directory to save individual state-dict files.

    Returns:
        Dict mapping component name to saved file path.
    """
    print(f"[load] {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "model_state_dict" in ck:
        sd = ck["model_state_dict"]
        epoch = ck.get("epoch", "?")
        print(f"[load] epoch={epoch}, total keys={len(sd)}")
    else:
        sd = ck
        print(f"[load] raw state_dict, total keys={len(sd)}")

    parts = split_state_dict(sd)

    os.makedirs(output_dir, exist_ok=True)
    saved_paths = {}
    for name, component_sd in parts.items():
        path = os.path.join(output_dir, f"{name}.pt")
        torch.save(component_sd, path)
        saved_paths[name] = path
        print(f"[save] {name} -> {path}")

    return saved_paths


def verify_loading(output_dir: str) -> None:
    """Quick verification: load all components into standalone model."""
    from plagiarism.model_plagiarism_torch import PlagiarismDetectionModelV2

    model = PlagiarismDetectionModelV2(device="cpu")
    model.load_components(
        encoder_path=os.path.join(output_dir, "encoder.pt"),
        fuser_path=os.path.join(output_dir, "fuser.pt"),
        match_head_path=os.path.join(output_dir, "match_head.pt"),
        decoder_path=os.path.join(output_dir, "decoder.pt"),
    )
    print("[verify] all 4 components loaded successfully (strict=True)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export plagiarism model components from a full checkpoint.",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to full training checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory to save individual component state-dicts.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="After export, verify by loading into standalone model.",
    )
    args = parser.parse_args()

    export_components(args.checkpoint, args.output_dir)
    if args.verify:
        verify_loading(args.output_dir)
