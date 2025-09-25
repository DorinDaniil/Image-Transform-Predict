import torch
import torch.nn as nn
from transformers import PreTrainedModel, GPT2Config, GPT2Model
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional
from .tokenizer import TRANSFORM_TOKENS, VOCAB_SIZE, START_TOKEN_ID, END_TOKEN_ID, PAD_TOKEN_ID


class TransformDecoder(PreTrainedModel):
    """
    Transformer-based decoder for predicting sequences of image transformations.
    Takes an image embedding and generates a sequence of transformation tokens.
    All tokens, including "noop", are treated as regular operations.
    Special tokens [PAD], [START], [END] are handled by Hugging Face conventions.

    Input: image_embedding [batch_size, hidden_size]
    Output: sequence of token IDs from vocabulary (including "noop")
    """

    config_class = GPT2Config

    def __init__(self, config: GPT2Config, **kwargs):
        super().__init__(config)
        self.hidden_size = config.n_embd
        self.vocab_size = VOCAB_SIZE  # ← Импортируем из tokenizer.py
        self.max_length = config.n_positions

        # Initialize transformer (GPT2-style decoder)
        self.transformer = GPT2Model(config)
        del self.transformer.wte  # Убираем стандартные word embeddings
        self.transformer.wte = nn.Identity()  # Заменяем на identity — будем подавать свои inputs_embeds

        # Позиционные эмбеддинги для последовательности
        self.position_embeddings = nn.Embedding(self.max_length, self.hidden_size)

        # Линейный слой для прогнозирования следующего токена
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size)

        # Инициализация весов
        self.init_weights()
        self.post_init()

    def init_weights(self):
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(
        self,
        image_embeddings: torch.FloatTensor,          # [B, H]
        input_ids: Optional[torch.LongTensor] = None,  # [B, L] — с [START] и [END]
        attention_mask: Optional[torch.Tensor] = None, # [B, L] — маска для паддинга
        labels: Optional[torch.LongTensor] = None,     # [B, L] — сдвинутые target
        **kwargs,
    ) -> CausalLMOutputWithPast:
        batch_size = image_embeddings.size(0)
        seq_len = input_ids.size(1) if input_ids is not None else 1

        # Проверка размерности эмбеддинга
        if image_embeddings.shape[1] != self.hidden_size:
            raise ValueError(f"Expected image_embedding dim {self.hidden_size}, got {image_embeddings.shape[1]}")

        # Расширяем image_embedding до длины последовательности: [B, H] --> [B, L, H]
        hidden_states = image_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        # Добавляем позиционные эмбеддинги
        position_ids = torch.arange(seq_len, device=image_embeddings.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = hidden_states + position_embeds  # [B, L, H]

        # Пропускаем через трансформер (используя inputs_embeds, а не input_ids)
        transformer_outputs = self.transformer(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        sequence_output = transformer_outputs.last_hidden_state  # [B, L, H]
        logits = self.lm_head(sequence_output)                  # [B, L, V]

        loss = None
        if labels is not None:
            # Сдвигаем для causal language modeling: предсказываем следующий токен
            shift_logits = logits[..., :-1, :].contiguous()      # [B, L-1, V]
            shift_labels = labels[..., 1:].contiguous()          # [B, L-1]

            # Loss игнорирует -100 (pad в labels)
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        image_embeddings: torch.FloatTensor,
        max_length: int = 32,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        early_stopping: bool = True,
    ) -> torch.LongTensor:
        """
        Autoregressive generation of transformation sequence from image embedding.

        Args:
            image_embeddings: [batch_size, hidden_size] — уже в нужном пространстве
            max_length: Максимальная длина генерируемой последовательности
            temperature: Температура для сэмплинга
            top_k: Выбор из top-k наиболее вероятных токенов
            top_p: Nucleus sampling (cumulative probability threshold)
            do_sample: Если False — greedy decoding
            early_stopping: Остановить, когда все последовательности достигли [END]

        Returns:
            Generated token IDs: [batch_size, generated_length]
        """
        batch_size = image_embeddings.size(0)
        device = image_embeddings.device

        # Начинаем с [START] токена
        generated = torch.full((batch_size, 1), START_TOKEN_ID, dtype=torch.long, device=device)
        past_key_values = None

        for step in range(max_length):
            cur_len = generated.size(1)

            # Расширяем image_embedding до текущей длины последовательности
            hidden_states = image_embeddings.unsqueeze(1).expand(-1, cur_len, -1)

            # Добавляем позиционные эмбеддинги
            position_ids = torch.arange(cur_len, device=device).unsqueeze(0).expand(batch_size, -1)
            position_embeds = self.position_embeddings(position_ids)
            hidden_states = hidden_states + position_embeds

            # Forward через декодер
            outputs = self.transformer(
                inputs_embeds=hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

            # Получаем логиты только для последнего токена
            logits = self.lm_head(outputs.last_hidden_state[:, -1:, :]).squeeze(1) / temperature  # [B, V]

            if do_sample:
                # Top-k фильтрация
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                # Top-p (nucleus) фильтрация
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('Inf')

                # Сэмплинг
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            # Добавляем к сгенерированной последовательности
            generated = torch.cat([generated, next_token], dim=1)

            # Early stopping: если все последовательности достигли [END]
            if early_stopping and (next_token == END_TOKEN_ID).all():
                break

            # Обновляем past_key_values для следующей итерации
            past_key_values = outputs.past_key_values

        return generated

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        Required for beam search compatibility with Hugging Face.
        Reorders cache tensors according to beam index.
        """
        return tuple(
            tuple(p.index_select(0, beam_idx) for p in layer) for layer in past_key_values
        )