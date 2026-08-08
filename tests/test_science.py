import numpy as np
import torch
from ag_tri_ct.configuration import ExperimentConfig
from ag_tri_ct.generation import CitationBinder
from ag_tri_ct.knowledge_tracing import ColdStartTracer, InteractionBatch
from ag_tri_ct.losses import soft_dice_loss
from ag_tri_ct.mesh import Mesh, chamfer_distance, is_watertight
from ag_tri_ct.metrics import CompositeMetrics, binary_auc, dice_score
from ag_tri_ct.ontology import AnatomyOntology, default_concepts
from ag_tri_ct.segmentation import FragmentAwareSegmenter
from ag_tri_ct.statistics import benjamini_hochberg, holm_bonferroni


def test_configuration_contract() -> None:
    config = ExperimentConfig()
    config.validate()
    assert config.effective_batch_size == 128


def test_ontology_shape_and_grounding() -> None:
    ontology = AnatomyOntology(default_concepts())
    assert ontology.embedding.weight.shape == (800, 256)
    assert ontology.adjacency().shape == (800, 800)
    assert ontology.grounding_loss().isfinite()


def test_segmentation_shapes() -> None:
    model = FragmentAwareSegmenter(channels=4)
    volume = torch.randn(1, 1, 8, 8, 8)
    embedding = torch.randn(1, 256)
    coordinates = torch.randn(1, 32, 3)
    output = model(volume, embedding, coordinates)
    assert output.voxel_logits.shape == (1, 2, 8, 8, 8)
    assert output.class_logits.shape == (1, 800)
    assert output.signed_distance.shape == (1, 32)


def test_dice_identity() -> None:
    target = torch.ones(2, 4, 4, 4, dtype=torch.long)
    prediction = torch.ones_like(target)
    assert float(dice_score(prediction, target)) == 1.0


def test_dice_loss_orders_predictions() -> None:
    target = torch.ones(1, 2, 2, 2, dtype=torch.long)
    good = torch.stack((torch.full_like(target, -5), torch.full_like(target, 5)), dim=1).float()
    bad = -good
    assert soft_dice_loss(good, target) < soft_dice_loss(bad, target)


def test_auc_known_ordering() -> None:
    scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
    labels = torch.tensor([0, 1, 0, 1])
    assert binary_auc(scores, labels) == 1.0


def test_composite_definition() -> None:
    metrics = CompositeMetrics(0.957, 0.886, 0.728, 0.054)
    assert 0.8 < metrics.composite() < 0.95


def test_bound_spans_require_both_sources() -> None:
    binder = CitationBinder()
    answer = binder.bind("First claim. Second claim.", ["e1"], ["AO-001"])
    assert binder.verify(answer, {"e1"}, {"AO-001"})


def test_tracer_output() -> None:
    tracer = ColdStartTracer(dimension=32, layers=1)
    ontology = AnatomyOntology(default_concepts(), dimension=32)
    batch = InteractionBatch(
        torch.tensor([[1, 2, 3]]), torch.tensor([[1, 0, 1]]), torch.ones(1, 3, dtype=torch.bool)
    )
    output = tracer(batch, ontology.embedding.weight, ontology.adjacency())
    assert output.shape == (1, 800)
    assert torch.all((output >= 0) & (output <= 1))


def test_mesh_metrics() -> None:
    points = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    assert chamfer_distance(points, points) == 0
    mesh = Mesh(
        torch.randn(4, 3),
        torch.tensor([[0, 1, 2], [0, 3, 1], [1, 3, 2], [0, 2, 3]]),
        torch.zeros(4, dtype=torch.long),
    )
    assert is_watertight(mesh)


def test_multiple_comparison_procedures() -> None:
    values = np.array([0.001, 0.01, 0.2], dtype=np.float64)
    assert holm_bonferroni(values).tolist() == [True, True, False]
    assert benjamini_hochberg(values).tolist() == [True, True, False]
