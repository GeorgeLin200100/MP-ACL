import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Embedding, Linear, Module, Sequential

from model.transformer import RMSNorm, TransformerBlock


def sample_top_p(probs: Tensor, threshold: float) -> Tensor:
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)  # (bs, vocab_size), (bs, vocab_size)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (bs, vocab_size)
    mask = cumulative_probs > threshold
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # rescale to sum to 1.0
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = torch.gather(sorted_indices, dim=-1, index=next_token)

    return next_token


class LLM(Module):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim_emb: int,
        num_layers: int,
        attn_num_heads: int,
        ffn_hidden_dim: int,
        ffn_bias: bool = False,
        emb_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.token_embedding = Embedding(vocab_size, dim_emb)
        self.emb_dropout = Dropout(emb_dropout)
        self.transformer = Sequential()
        for _ in range(num_layers):
            self.transformer.append(TransformerBlock(seq_len, dim_emb, attn_num_heads, ffn_hidden_dim, ffn_bias))
        self.norm = RMSNorm(dim_emb)
        self.projection_head = Linear(dim_emb, vocab_size)
        self.token_embedding.weight = self.projection_head.weight

    def forward(self, x: Tensor) -> Tensor:
        x = self.token_embedding(x)  # (bs, seq_len, dim_emb)
        x = self.emb_dropout(x)  # (bs, seq_len, dim_emb)
        x = self.transformer(x)  # (bs, seq_len, dim_emb)
        x = self.norm(x)  # (bs, seq_len, dim_emb)
        x = self.projection_head(x)  # (bs, seq_len, vocab_size)

        return x  # (bs, seq_len, vocab_size)

    @torch.inference_mode()
    def generate(
        self,
        inputs: Tensor,
        max_seq_len: int,
        stop_tokens: set | None = None,
        temperature: float = 0.6,
        top_p: int = 0.8,
    ) -> Tensor:
        for _ in range(max_seq_len):           
            inputs_cond = inputs if inputs.size(1) <= self.seq_len else inputs[:, -self.seq_len :]
            logits = self(inputs_cond)[:, -1, :]  # (bs, vocab_size)
            probs = F.softmax(logits / temperature, dim=-1)  # (bs, vocab_size)
            next_token = sample_top_p(probs, top_p)  # (bs, 1)
            if stop_tokens is not None and next_token.item() in stop_tokens:
                break
            inputs = torch.cat((inputs, next_token), dim=-1)

        return inputs.squeeze().cpu()
