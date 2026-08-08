from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Concept:
    index: int
    code: str
    parent: int
    region: str
    qualifier: str


class AnatomyOntology(nn.Module):
    def __init__(self, concepts: list[Concept], dimension: int = 256) -> None:
        super().__init__()
        if len(concepts) != 800:
            raise ValueError("exactly 800 concepts are required")
        self.concepts = tuple(concepts)
        self.embedding = nn.Embedding(len(concepts), dimension)
        edges = [(concept.parent, concept.index) for concept in concepts if concept.parent >= 0]
        self.register_buffer("edge_index", torch.tensor(edges, dtype=torch.long).t().contiguous())

    def forward(self, indices: Tensor) -> Tensor:
        return self.embedding(indices)

    def hierarchy_weights(self) -> Tensor:
        source, target = self.edge_index
        difference = (source - target).abs().to(dtype=torch.float32)
        return torch.reciprocal(1.0 + difference)

    def grounding_loss(self) -> Tensor:
        source, target = self.edge_index
        delta = self.embedding.weight[source] - self.embedding.weight[target]
        weights = self.hierarchy_weights()
        return (delta.square().sum(dim=-1) * weights).mean()

    def adjacency(self) -> Tensor:
        matrix = torch.zeros(800, 800, device=self.embedding.weight.device)
        source, target = self.edge_index
        matrix[source, target] = 1.0
        matrix[target, source] = 1.0
        matrix.fill_diagonal_(1.0)
        return matrix / matrix.sum(dim=-1, keepdim=True).clamp_min(1.0)


def default_concepts() -> list[Concept]:
    regions = ("proximal", "diaphyseal", "distal", "articular")
    qualifiers = ("simple", "wedge", "complex", "multifragmentary", "displaced")
    concepts: list[Concept] = []
    for index in range(800):
        parent = -1 if index == 0 else (index - 1) // 4
        region = regions[(index // len(qualifiers)) % len(regions)]
        qualifier = qualifiers[index % len(qualifiers)]
        concepts.append(Concept(index, f"AO-{index:03d}", parent, region, qualifier))
    return concepts
