from transformers import PreTrainedTokenizerBase
from typing import List, Union, Dict, Optional, Any
import torch

TRANSFORM_TOKENS = {
    "[PAD]": 0,           # Padding — всегда специальный
    "[START]": 1,         # Special: начало последовательности
    "[END]": 2,           # Special: конец последовательности
    "noop": 3,            # Тождественное преобразование
    "grayscale": 4,
    "rotate_90": 5,
    "rotate_180": 6,
    "rotate_270": 7,
    "color_jitter": 8,
    "noise_adding": 9,
    "crop": 10,
    "horizontal_flip": 11,
    "vertical_flip": 12,
    "resize": 13,
}

ID_TO_TOKEN = {v: k for k, v in TRANSFORM_TOKENS.items()}
VOCAB_SIZE = len(TRANSFORM_TOKENS)

PAD_TOKEN_ID = 0
START_TOKEN_ID = 1
END_TOKEN_ID = 2

ALL_TRANSFORMS = [t for t in TRANSFORM_TOKENS.keys() if t not in ["[PAD]", "[START]", "[END]"]]
SPECIAL_TOKENS = ["[PAD]", "[START]", "[END]"]


class TransformTokenizer(PreTrainedTokenizerBase):
    """
    A custom tokenizer for image augmentation sequences.
    Accepts List[str] of transform names and converts them to token IDs.
    Supports encoding/decoding with special tokens and Hugging Face Trainer compatibility.
    """

    def __init__(self, **kwargs):
        super().__init__(
            pad_token="[PAD]",
            bos_token="[START]",
            eos_token="[END]",
            unk_token="[PAD]",
            **kwargs
        )
        self.vocab = TRANSFORM_TOKENS
        self.ids_to_tokens = ID_TO_TOKEN
        self.model_max_length = 32

    def encode(self, texts: Union[str, List[str]], add_special_tokens: bool = True, **kwargs) -> List[int]:
        """
        Encode a list of transform names into token IDs.
        Example: ["grayscale", "crop"] --> [1, 4, 10, 2] (with special tokens)
        """
        if isinstance(texts, str):
            texts = texts.split()

        token_ids = [self._convert_token_to_id(token) for token in texts]

        if add_special_tokens:
            start_id = self.convert_token_to_id("[START]")
            end_id = self.convert_token_to_id("[END]")
            token_ids = [start_id] + token_ids + [end_id]

        return token_ids

    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> List[str]:
        """
        Decode a list of token IDs back to transform names.
        Example: [1, 4, 5, 10, 2] --> ['grayscale', 'rotate_90', 'crop']
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if not isinstance(token_ids, list):
            raise TypeError(f"Expected token_ids to be a list or tensor, got {type(token_ids)}")

        for i, tid in enumerate(token_ids):
            if not isinstance(tid, int):
                raise TypeError(f"Element at index {i} is not an integer: {tid}")

        tokens = []
        for tid in token_ids:
            token = self._convert_id_to_token(tid)
            if skip_special_tokens and token in ["[START]", "[END]", "[PAD]"]:
                continue
            tokens.append(token)
        return tokens

    # --- REQUIRED METHODS FOR PreTrainedTokenizerBase ---

    def _convert_token_to_id(self, token: str) -> int:
        """Private method: Convert token string to ID."""
        return self.vocab.get(token, self.vocab["[PAD]"])

    def convert_token_to_id(self, token: str) -> int:
        """Public method: Required by Hugging Face. Delegates to _convert_token_to_id."""
        return self._convert_token_to_id(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Private method: Convert ID to token string."""
        return self.ids_to_tokens.get(index, "[PAD]")

    def convert_id_to_token(self, index: int) -> str:
        """Public method: Required by Hugging Face. Delegates to _convert_id_to_token."""
        return self._convert_id_to_token(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Required by base class. Not used often, but needed for compatibility."""
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        else:
            return [self.bos_token_id] + token_ids_0 + [self.eos_token_id] + token_ids_1 + [self.eos_token_id]

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE

    def save_pretrained(self, save_directory: str, **kwargs) -> List[str]:
        import json
        vocab_file = f"{save_directory}/vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(TRANSFORM_TOKENS, f, indent=2)
        return [vocab_file]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        import json
        vocab_file = f"{pretrained_model_name_or_path}/vocab.json"
        with open(vocab_file, 'r') as f:
            global TRANSFORM_TOKENS
            TRANSFORM_TOKENS = json.load(f)
        global ID_TO_TOKEN
        ID_TO_TOKEN = {v: k for k, v in TRANSFORM_TOKENS.items()}
        return cls()