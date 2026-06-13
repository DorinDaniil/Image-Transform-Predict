"""Filter negative_test_dataset_meta.json against good_pairs_after_loss_90.json.

Keeps only the meta entries whose pair also appears in good_pairs (matched by
batch name + both image file names), then drops any pair where either image
class is "monotone". Output preserves the negative_test_dataset_meta entry
shape exactly: image_1, image_2, batch_name, image_1_class, image_2_class.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
GOOD_PAIRS_PATH = _REPO_ROOT / "data_clean" / "good_pairs_after_loss_90.json"
NEGATIVE_META_PATH = Path(__file__).resolve().parent / "negative_test_dataset_meta.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "filtered_negative_test_dataset_meta.json"

EXCLUDED_CLASS = "monotone"

MetaIndex = dict[tuple[str, str, str], dict[str, Any]]


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_meta_index(meta: list[dict[str, Any]]) -> MetaIndex:
    """Map (batch_name, image_1, image_2) -> full meta entry."""
    index: MetaIndex = {}
    for entry in meta:
        key = (entry["batch_name"], entry["image_1"], entry["image_2"])
        index[key] = entry
    return index


def pair_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    """Derive a (batch_name, img1_name, img2_name) key from a good_pairs entry."""
    img1 = PurePosixPath(entry["img1"])
    img2 = PurePosixPath(entry["img2"])
    batch_name = img1.parts[0]
    return batch_name, img1.name, img2.name


def filter_pairs(
    good_pairs: list[dict[str, Any]],
    meta_index: MetaIndex,
) -> tuple[list[dict[str, Any]], int, int]:
    """Return (filtered_meta_entries, missing_in_meta_count, dropped_monotone_count).

    Output entries are the original meta entries (same schema as the input
    negative_test_dataset_meta.json).
    """
    filtered: list[dict[str, Any]] = []
    missing = 0
    dropped_monotone = 0

    for entry in good_pairs:
        meta_entry = meta_index.get(pair_key(entry))
        if meta_entry is None:
            missing += 1
            continue

        if EXCLUDED_CLASS in (meta_entry["image_1_class"], meta_entry["image_2_class"]):
            dropped_monotone += 1
            continue

        filtered.append(meta_entry)

    return filtered, missing, dropped_monotone


def main() -> None:
    good_pairs = load_json(GOOD_PAIRS_PATH)
    meta = load_json(NEGATIVE_META_PATH)

    meta_index = build_meta_index(meta)
    filtered, missing, dropped_monotone = filter_pairs(good_pairs, meta_index)

    OUTPUT_PATH.write_text(
        json.dumps(filtered, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"good_pairs total:       {len(good_pairs)}")
    print(f"negative_meta total:    {len(meta)}")
    print(f"kept after filter:      {len(filtered)}")
    print(f"missing in meta (skip): {missing}")
    print(f"dropped (monotone):     {dropped_monotone}")
    print(f"output:                 {OUTPUT_PATH}")


if __name__ == "__main__":
    main()