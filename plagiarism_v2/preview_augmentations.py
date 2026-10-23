#!/usr/bin/env python3
"""
Visual sanity check for the v2 augmentations.

Writes a grid image (original | augmented, caption = sampled tokens and the
fragment-edit kinds) and prints token / kind / language statistics. Use it to
judge realism before launching a training run.

Examples (from the repository root):
    python -m plagiarism_v2.preview_augmentations --images data/real --out plagiarism_v2/preview
    python -m plagiarism_v2.preview_augmentations --images data/real --tokens text_overlay --n 24
    python -m plagiarism_v2.preview_augmentations --images data/real --layouts caption_bar meme subtitle
    python -m plagiarism_v2.preview_augmentations --images data/real --kinds blur_censor black_box pixelate_censor
    python -m plagiarism_v2.preview_augmentations --images data/real --tokens jpeg_artefacts fragment_edit
"""

import argparse
import os
import random
from collections import Counter
from typing import List

from PIL import Image, ImageDraw, ImageFont

from plagiarism_v2.dataset_v2 import FRAGMENT_KINDS, TEXT_LAYOUTS, ImageTransformerV2


def _list_images(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    out.append(os.path.join(p, f))
        elif os.path.isfile(p):
            out.append(p)
    if not out:
        raise SystemExit(f"No images found in {paths}")
    return out


def _thumb(img: Image.Image, side: int) -> Image.Image:
    """Letterbox to a square thumbnail for display only (training uses plain 300x300 resize)."""
    im = img.convert("RGB").copy()
    im.thumbnail((side, side), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (40, 40, 40))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", default=["assets"], help="Image files or folders.")
    ap.add_argument("--out", default="plagiarism_v2/preview", help="Output folder.")
    ap.add_argument("--n", type=int, default=12, help="Number of samples in the grid.")
    ap.add_argument("--p", type=float, default=0.3, help="Per-token probability (ignored with --tokens/--kinds).")
    ap.add_argument("--tokens", nargs="*", default=None,
                    help="Force these tokens (one per sample, cycling) instead of random sampling.")
    ap.add_argument("--kinds", nargs="*", default=None, choices=list(FRAGMENT_KINDS),
                    help="Force fragment_edit with these kinds (one per sample, cycling).")
    ap.add_argument("--layouts", nargs="*", default=None, choices=list(TEXT_LAYOUTS),
                    help="Force text_overlay with these layouts (one per sample, cycling).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--side", type=int, default=320, help="Thumbnail side in the grid.")
    ap.add_argument("--cols", type=int, default=3, help="Pairs per row.")
    ap.add_argument("--sticker-dir", default="plagiarism_v2/assets/stickers")
    ap.add_argument("--text-corpus-dir", default=None)
    ap.add_argument("--stats-samples", type=int, default=300,
                    help="Extra random samples for token statistics (0 to skip).")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    paths = _list_images(args.images)
    tr = ImageTransformerV2(sticker_dir=args.sticker_dir, text_corpus_dir=args.text_corpus_dir)
    fe = tr.fragment_editor
    print(f"fonts: {len(tr.font_bank.paths)} truetype | sticker assets: {len(fe.sticker_bank.paths)}")
    print(f"text languages: {tr.text_langs}")
    print(f"fragment kinds: {dict(fe.kind_weights)}")

    try:
        cap_font = ImageFont.load_default(size=14)
    except TypeError:
        cap_font = ImageFont.load_default()

    side, cols = args.side, args.cols
    cap_h = 22
    cell_w, cell_h = 2 * side + 6, side + cap_h
    rows = (args.n + cols - 1) // cols
    grid = Image.new("RGB", (cols * cell_w + (cols + 1) * 8, rows * cell_h + (rows + 1) * 8), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    for i in range(args.n):
        path = paths[i % len(paths)]
        img = Image.open(path).convert("RGB")
        tr.last_fragment_kinds, tr.last_fragment_details, tr.last_text_detail = [], [], None
        if args.kinds:
            kind = args.kinds[i % len(args.kinds)]
            aug, ok = tr._apply_fragment_edit(img, kinds=[kind])
            seq = ["fragment_edit"] if ok else ["noop"]
        elif args.layouts:
            aug, ok = tr._apply_text_overlay(img, layout=args.layouts[i % len(args.layouts)])
            seq = ["text_overlay"] if ok else ["noop"]
        elif args.tokens:
            aug, seq = tr._apply_sequence(img, [args.tokens[i % len(args.tokens)]])
        else:
            aug, seq = tr.transform(img, p=args.p)
        kinds = list(tr.last_fragment_kinds)
        if tr.last_text_detail:
            kinds.append(f"text:{tr.last_text_detail}")
        r, c = divmod(i, cols)
        x = 8 + c * (cell_w + 8)
        y = 8 + r * (cell_h + 8)
        grid.paste(_thumb(img, side), (x, y))
        grid.paste(_thumb(aug, side), (x + side + 6, y))
        caption = f"{' '.join(seq)}" + (f"  [{', '.join(kinds)}]" if kinds else "")
        draw.text((x + 4, y + side + 4), caption[:90], fill=(230, 230, 230), font=cap_font)
        print(f"[{i:02d}] {os.path.basename(path)} {img.size} -> {aug.size}  {seq}  {kinds}")

    out_path = os.path.join(args.out, f"preview_seed{args.seed}.png")
    grid.save(out_path)
    print(f"saved {out_path}")

    if args.stats_samples > 0 and not (args.tokens or args.kinds or args.layouts):
        tok_counter, kind_counter, lang_counter, layout_counter, lengths = Counter(), Counter(), Counter(), Counter(), Counter()
        dropped = Counter()
        for j in range(args.stats_samples):
            img = Image.open(paths[j % len(paths)]).convert("RGB")
            seq_sampled = tr.sample_transformations(img, args.p)
            _, seq = tr._apply_sequence(img, seq_sampled)
            for t in ("fragment_edit", "text_overlay"):
                if t in seq_sampled and t not in seq:
                    dropped[t] += 1
            tok_counter.update(seq)
            kind_counter.update(tr.last_fragment_kinds)
            if tr.last_text_detail:
                layout, lang = tr.last_text_detail.split("/")
                layout_counter[layout] += 1
                lang_counter[lang] += 1
            lengths[len([t for t in seq if t != "noop"])] += 1
        n = args.stats_samples
        print(f"\ntoken frequency over {n} samples at p={args.p}:")
        for tok, cnt in tok_counter.most_common():
            print(f"  {tok:16s} {cnt / n:6.3f}")
        print("fragment kinds:", dict(kind_counter.most_common()))
        print("text layouts:", dict(layout_counter.most_common()))
        print("text languages:", dict(lang_counter.most_common()))
        print("sequence length distribution:", dict(sorted(lengths.items())))
        print(f"tokens dropped by the visibility check: {dict(dropped)}")


if __name__ == "__main__":
    main()
