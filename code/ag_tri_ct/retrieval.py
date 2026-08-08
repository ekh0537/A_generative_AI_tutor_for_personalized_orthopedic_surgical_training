import math
import re
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class EvidenceChunk:
    identifier: str
    text: str
    concept_index: int
    source: str


@dataclass(frozen=True)
class RetrievalResult:
    chunk: EvidenceChunk
    score: float


class TokenIndex:
    def __init__(self, chunks: list[EvidenceChunk]) -> None:
        self.chunks = tuple(chunks)
        self.documents = [self._tokens(chunk.text) for chunk in chunks]
        self.frequencies: dict[str, int] = {}
        for document in self.documents:
            for token in set(document):
                self.frequencies[token] = self.frequencies.get(token, 0) + 1

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def score(self, query: str, index: int) -> float:
        tokens = self._tokens(query)
        document = self.documents[index]
        length = max(len(document), 1)
        score = 0.0
        for token in tokens:
            frequency = document.count(token)
            inverse = math.log((len(self.documents) + 1) / (self.frequencies.get(token, 0) + 0.5))
            score += inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / 128))
        return score


class DenseEncoder(nn.Module):
    def __init__(self, vocabulary_size: int = 32768, dimension: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, dimension)
        self.projection = nn.Linear(dimension, dimension)

    def forward(self, token_ids: Tensor, mask: Tensor) -> Tensor:
        embedded = self.embedding(token_ids)
        pooled = (embedded * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return functional.normalize(self.projection(pooled), dim=-1)


class HybridRetriever(nn.Module):
    def __init__(self, chunks: list[EvidenceChunk], dimension: int = 256) -> None:
        super().__init__()
        self.chunks = tuple(chunks)
        self.index = TokenIndex(chunks)
        self.encoder = DenseEncoder(dimension=dimension)
        self.register_buffer("document_embeddings", torch.zeros(len(chunks), dimension))

    def update_document_embeddings(self, embeddings: Tensor) -> None:
        if embeddings.shape != self.document_embeddings.shape:
            raise ValueError("document embedding shape mismatch")
        self.document_embeddings.copy_(functional.normalize(embeddings, dim=-1))

    def retrieve(
        self, query: str, query_embedding: Tensor, concept_embedding: Tensor, top_k: int = 5
    ) -> list[RetrievalResult]:
        if top_k != 5:
            raise ValueError("the evidence protocol requires top_k=5")
        key = functional.normalize(query_embedding + concept_embedding, dim=-1)
        dense = self.document_embeddings @ key
        lexical = torch.tensor(
            [self.index.score(query, index) for index in range(len(self.chunks))],
            device=dense.device,
            dtype=dense.dtype,
        )
        lexical = (lexical - lexical.mean()) / lexical.std().clamp_min(1e-6)
        combined = 0.5 * dense + 0.5 * lexical
        values, indices = torch.topk(combined, min(top_k, len(self.chunks)))
        return [
            RetrievalResult(self.chunks[index], float(value))
            for value, index in zip(values.tolist(), indices.tolist(), strict=True)
        ]
