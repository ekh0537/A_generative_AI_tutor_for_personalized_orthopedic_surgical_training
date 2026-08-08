from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class Mesh:
    vertices: Tensor
    faces: Tensor
    labels: Tensor


def sample_surface_points(signed_distance: Tensor, coordinates: Tensor, count: int) -> Tensor:
    if signed_distance.shape != coordinates.shape[:-1]:
        raise ValueError("distance and coordinate shapes are incompatible")
    weights = torch.softmax(-signed_distance.abs(), dim=-1)
    indices = torch.multinomial(weights, count, replacement=True)
    expanded = indices.unsqueeze(-1).expand(-1, -1, 3)
    return torch.gather(coordinates, 1, expanded)


def chamfer_distance(first: Tensor, second: Tensor) -> Tensor:
    distances = torch.cdist(first, second)
    return distances.min(dim=-1).values.mean() + distances.min(dim=-2).values.mean()


def hausdorff95(first: Tensor, second: Tensor) -> Tensor:
    distances = torch.cdist(first, second)
    forward = torch.quantile(distances.min(dim=-1).values, 0.95, dim=-1)
    backward = torch.quantile(distances.min(dim=-2).values, 0.95, dim=-1)
    return torch.maximum(forward, backward).mean()


def edge_incidence(faces: Tensor) -> dict[tuple[int, int], int]:
    incidence: dict[tuple[int, int], int] = {}
    for face in faces.tolist():
        edges = ((face[0], face[1]), (face[1], face[2]), (face[2], face[0]))
        for left, right in edges:
            edge = (min(left, right), max(left, right))
            incidence[edge] = incidence.get(edge, 0) + 1
    return incidence


def is_watertight(mesh: Mesh) -> bool:
    if mesh.faces.numel() == 0 or mesh.vertices.numel() == 0:
        return False
    return all(count == 2 for count in edge_incidence(mesh.faces).values())
