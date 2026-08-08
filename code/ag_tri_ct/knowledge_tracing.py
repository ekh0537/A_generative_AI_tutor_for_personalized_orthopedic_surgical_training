from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class InteractionBatch:
    skills: Tensor
    responses: Tensor
    mask: Tensor


class TaxonomyMessagePass(nn.Module):
    def __init__(self, dimension: int = 256) -> None:
        super().__init__()
        self.self_projection = nn.Linear(dimension, dimension)
        self.neighbor_projection = nn.Linear(dimension, dimension)
        self.gate = nn.Linear(dimension * 2, dimension)

    def forward(self, states: Tensor, adjacency: Tensor) -> Tensor:
        neighbors = adjacency @ states
        own = self.self_projection(states)
        other = self.neighbor_projection(neighbors)
        gate = torch.sigmoid(self.gate(torch.cat((own, other), dim=-1)))
        return functional.layer_norm(gate * own + (1.0 - gate) * other, (states.shape[-1],))


class ColdStartTracer(nn.Module):
    def __init__(self, concepts: int = 800, dimension: int = 256, layers: int = 3) -> None:
        super().__init__()
        self.skill_embedding = nn.Embedding(concepts, dimension)
        self.response_embedding = nn.Embedding(2, dimension)
        encoder_layer = nn.TransformerEncoderLayer(dimension, 8, dimension * 4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.message_pass = TaxonomyMessagePass(dimension)
        self.readout = nn.Linear(dimension, 1)

    def forward(
        self, batch: InteractionBatch, ontology_embedding: Tensor, adjacency: Tensor
    ) -> Tensor:
        skill = self.skill_embedding(batch.skills)
        response = self.response_embedding(batch.responses.long())
        length = batch.skills.shape[1]
        causal = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=batch.skills.device),
            diagonal=1,
        )
        sequence = self.encoder(
            skill + response, mask=causal, src_key_padding_mask=~batch.mask.bool()
        )
        learner = sequence[:, -1]
        aligned = self.aligned_skill_states(ontology_embedding, adjacency)
        logits = torch.einsum("bd,cd->bc", learner, aligned) / aligned.shape[-1] ** 0.5
        return torch.sigmoid(logits)

    def aligned_skill_states(self, ontology_embedding: Tensor, adjacency: Tensor) -> Tensor:
        base = 0.5 * self.skill_embedding.weight + 0.5 * ontology_embedding
        return self.message_pass(base, adjacency)

    def alignment_loss(self, ontology_embedding: Tensor) -> Tensor:
        return functional.mse_loss(
            functional.normalize(self.skill_embedding.weight, dim=-1),
            functional.normalize(ontology_embedding, dim=-1),
        )

    def adapt(self, batch: InteractionBatch, steps: int = 5, learning_rate: float = 1e-3) -> None:
        if batch.skills.shape[1] > 10:
            raise ValueError("adaptation accepts at most 10 events")
        parameters = list(self.encoder.parameters()) + list(self.readout.parameters())
        optimizer = torch.optim.SGD(parameters, lr=learning_rate)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            skill = self.skill_embedding(batch.skills)
            response = self.response_embedding(batch.responses.long())
            hidden = self.encoder(skill + response)
            probability = torch.sigmoid(self.readout(hidden).squeeze(-1))
            loss = functional.binary_cross_entropy(probability, batch.responses.float())
            loss.backward()
            optimizer.step()
