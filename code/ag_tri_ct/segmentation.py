from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class SegmentationOutput:
    voxel_logits: Tensor
    class_logits: Tensor
    signed_distance: Tensor
    surface_points: Tensor


class Film3d(nn.Module):
    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(embedding_dim, channels * 2)

    def forward(self, features: Tensor, embedding: Tensor) -> Tensor:
        scale, shift = self.affine(embedding).chunk(2, dim=-1)
        shape = (embedding.shape[0], embedding.shape[-1])
        scale = scale.view(shape[0], -1, 1, 1, 1)
        shift = shift.view(shape[0], -1, 1, 1, 1)
        return features * (1.0 + scale) + shift


class ResidualVolumeBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.first = nn.Conv3d(channels, channels, 3, padding=1)
        self.second = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm_first = nn.InstanceNorm3d(channels, affine=True)
        self.norm_second = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, volume: Tensor) -> Tensor:
        hidden = functional.gelu(self.norm_first(self.first(volume)))
        hidden = self.norm_second(self.second(hidden))
        return functional.gelu(volume + hidden)


class ImplicitSurfaceNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 256, hidden_dim: int = 256) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, coordinates: Tensor, embedding: Tensor) -> Tensor:
        expanded = embedding[:, None, :].expand(-1, coordinates.shape[1], -1)
        return self.layers(torch.cat((coordinates, expanded), dim=-1)).squeeze(-1)


class FragmentAwareSegmenter(nn.Module):
    def __init__(self, classes: int = 800, embedding_dim: int = 256, channels: int = 32) -> None:
        super().__init__()
        self.stem = nn.Conv3d(1, channels, 3, padding=1)
        self.encoder = nn.Sequential(
            ResidualVolumeBlock(channels),
            nn.Conv3d(channels, channels * 2, 3, stride=2, padding=1),
            ResidualVolumeBlock(channels * 2),
        )
        self.film = Film3d(channels * 2, embedding_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, 2, stride=2),
            ResidualVolumeBlock(channels),
            nn.Conv3d(channels, 2, 1),
        )
        self.classifier = nn.Linear(channels * 2, classes)
        self.implicit = ImplicitSurfaceNetwork(embedding_dim)

    def forward(self, volume: Tensor, embedding: Tensor, coordinates: Tensor) -> SegmentationOutput:
        encoded = self.encoder(functional.gelu(self.stem(volume)))
        conditioned = self.film(encoded, embedding)
        voxel_logits = self.decoder(conditioned)
        pooled = conditioned.mean(dim=(-3, -2, -1))
        class_logits = self.classifier(pooled)
        signed_distance = self.implicit(coordinates, embedding)
        weights = torch.softmax(-signed_distance.abs(), dim=-1)
        surface_points = coordinates * weights.unsqueeze(-1)
        return SegmentationOutput(voxel_logits, class_logits, signed_distance, surface_points)

    def infer_class(self, volume: Tensor, candidate_embeddings: Tensor) -> Tensor:
        stemmed = functional.gelu(self.stem(volume))
        encoded = self.encoder(stemmed)
        pooled = encoded.mean(dim=(-3, -2, -1))
        logits = self.classifier(pooled)
        indices = logits.argmax(dim=-1)
        return candidate_embeddings[indices]
