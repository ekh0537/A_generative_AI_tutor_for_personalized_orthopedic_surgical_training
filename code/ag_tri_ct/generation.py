import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class BoundSpan:
    text: str
    evidence_ids: tuple[str, ...]
    concept_codes: tuple[str, ...]


@dataclass(frozen=True)
class TutorAnswer:
    text: str
    spans: tuple[BoundSpan, ...]
    valid: bool


class CitationBinder:
    def split(self, text: str) -> list[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]

    def bind(
        self,
        text: str,
        evidence_ids: list[str],
        concept_codes: list[str],
    ) -> TutorAnswer:
        spans = self.split(text)
        if not evidence_ids or not concept_codes:
            return TutorAnswer(text, tuple(), False)
        bound = tuple(
            BoundSpan(
                span,
                (evidence_ids[index % len(evidence_ids)],),
                (concept_codes[index % len(concept_codes)],),
            )
            for index, span in enumerate(spans)
        )
        valid = len(bound) > 0 and all(item.evidence_ids and item.concept_codes for item in bound)
        return TutorAnswer(text, bound, valid)

    def verify(
        self, answer: TutorAnswer, available_ids: set[str], available_codes: set[str]
    ) -> bool:
        if not answer.valid:
            return False
        return all(
            set(span.evidence_ids) <= available_ids and set(span.concept_codes) <= available_codes
            for span in answer.spans
        )


class GroundedDecoder(nn.Module):
    def __init__(self, vocabulary_size: int = 32768, dimension: int = 256, layers: int = 4) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, dimension)
        block = nn.TransformerDecoderLayer(dimension, 8, dimension * 4, batch_first=True)
        self.decoder = nn.TransformerDecoder(block, layers)
        self.output = nn.Linear(dimension, vocabulary_size)

    def forward(self, tokens: Tensor, memory: Tensor, causal_mask: Tensor | None = None) -> Tensor:
        hidden = self.token_embedding(tokens)
        decoded = self.decoder(hidden, memory, tgt_mask=causal_mask)
        return self.output(decoded)

    def greedy(self, prefix: Tensor, memory: Tensor, maximum_new_tokens: int) -> Tensor:
        tokens = prefix
        for _ in range(maximum_new_tokens):
            length = tokens.shape[1]
            mask = torch.triu(
                torch.full((length, length), float("-inf"), device=tokens.device), diagonal=1
            )
            logits = self.forward(tokens, memory, mask)
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens
