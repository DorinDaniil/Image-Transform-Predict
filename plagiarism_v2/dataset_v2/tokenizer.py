"""
Transform vocabulary for plagiarism_v2.

Ids 0..12 are identical to the vocabulary of the deployed v1 plagiarism model,
id 13 matches ``jpeg_artefacts`` in ``src.dataset.tokenizer``; ids 14-15 are new.
Old ids are kept stable so that v1 and v2 predictions stay comparable.

``crop`` (id 10) -- in v2 means "field of view changed": cut away (v1 crop) OR
margins added (frame / document margins / letterbox, see ``reframe.py``).
``text_overlay``  -- realistic text put on the image (own category).
``fragment_edit`` -- ONE token for every other local change (sticker, censoring
blur / mosaic / box, copy-move, inpaint fill, recolor); the concrete kind is
sampled from the fixed list ``plagiarism_v2.dataset_v2.overlays.FRAGMENT_KINDS``.
"""

from typing import Dict

from src.dataset.tokenizer import TransformTokenizer

TRANSFORM_TOKENS_V2: Dict[str, int] = {
    "[PAD]": 0,
    "[START]": 1,
    "[END]": 2,
    "noop": 3,
    "grayscale": 4,
    "rotate_90": 5,
    "rotate_180": 6,
    "rotate_270": 7,
    "color_jitter": 8,
    "noise_adding": 9,
    "crop": 10,
    "horizontal_flip": 11,
    "vertical_flip": 12,
    "jpeg_artefacts": 13,
    "text_overlay": 14,
    "fragment_edit": 15,
}

ID_TO_TOKEN_V2: Dict[int, str] = {v: k for k, v in TRANSFORM_TOKENS_V2.items()}
VOCAB_SIZE_V2: int = len(TRANSFORM_TOKENS_V2)  # 16

# Tokens that did not exist in the deployed v1 vocabulary (ids 0..12).
NEW_TOKENS_V2 = ("jpeg_artefacts", "text_overlay", "fragment_edit")


class TransformTokenizerV2(TransformTokenizer):
    """
    Same encode/decode behaviour as ``src.dataset.TransformTokenizer`` (special
    token ids 0/1/2 are shared), but over the 16-token v2 vocabulary.
    """

    def __init__(self):
        # Deliberately do not call super().__init__(): it would bind the
        # module-level v1 vocabulary. Special-token ids are the same (0, 1, 2).
        self.vocab = dict(TRANSFORM_TOKENS_V2)
        self.ids_to_tokens = dict(ID_TO_TOKEN_V2)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
