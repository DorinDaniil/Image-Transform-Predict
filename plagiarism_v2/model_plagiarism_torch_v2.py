"""
Standalone inference wrapper for the v2 vocabulary.

Reuses the four ONNX-exportable components from
``plagiarism.model_plagiarism_torch`` unchanged; the only difference is the
decoder's ``vocab_size`` (16) and the token-name table used for decoding.
Weights exported with ``plagiarism.export_plagiarism_weights`` from a v2
checkpoint load directly.
"""

from typing import List, Optional, Union

import torch

from plagiarism.model_plagiarism_torch import (
    BinaryMatchHead,
    ImageEncoder,
    ImageFuser,
    PlagiarismDetectionModelV2,
    TransformationDecoder,
    preprocess,
)
from plagiarism_v2.dataset_v2.tokenizer import ID_TO_TOKEN_V2, VOCAB_SIZE_V2

__all__ = ["PlagiarismDetectionModelV2Vocab", "TRANSFORM_TOKEN_NAMES_V2", "preprocess"]

TRANSFORM_TOKEN_NAMES_V2 = dict(ID_TO_TOKEN_V2)


class PlagiarismDetectionModelV2Vocab(PlagiarismDetectionModelV2):
    """Same inference modes as the v1 wrapper, decoder built with the 16-token vocabulary."""

    def __init__(self, device: Optional[Union[str, torch.device]] = None) -> None:
        torch.nn.Module.__init__(self)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = ImageEncoder()
        self.fuser = ImageFuser()
        self.match_head = BinaryMatchHead()
        self.decoder = TransformationDecoder(vocab_size=VOCAB_SIZE_V2)

        self.to(self.device)

    @staticmethod
    def decode_tokens(token_ids: torch.Tensor, skip_special: bool = True) -> List[List[str]]:
        """[B, T] token ids -> list of transform-name lists."""
        out = []
        for row in token_ids.tolist():
            names = []
            for tid in row:
                if skip_special and tid in (0, 1, 2):
                    continue
                names.append(TRANSFORM_TOKEN_NAMES_V2.get(int(tid), "[UNK]"))
            out.append(names)
        return out
