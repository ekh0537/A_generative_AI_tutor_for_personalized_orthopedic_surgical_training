from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .configuration import ExperimentConfig
from .generation import CitationBinder, TutorAnswer
from .knowledge_tracing import ColdStartTracer, InteractionBatch
from .ontology import AnatomyOntology, default_concepts
from .retrieval import EvidenceChunk, HybridRetriever
from .segmentation import FragmentAwareSegmenter, SegmentationOutput


@dataclass(frozen=True)
class TutorOutput:
    segmentation: SegmentationOutput
    answer: TutorAnswer
    mastery: Tensor
    concept_indices: Tensor


class AnatomyGroundedTutor(nn.Module):
    def __init__(self, chunks: list[EvidenceChunk], config: ExperimentConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExperimentConfig()
        self.config.validate()
        self.ontology = AnatomyOntology(default_concepts(), self.config.embedding_dim)
        self.segmenter = FragmentAwareSegmenter(
            self.config.ontology_size, self.config.embedding_dim
        )
        self.retriever = HybridRetriever(chunks, self.config.embedding_dim)
        self.tracer = ColdStartTracer(self.config.ontology_size, self.config.embedding_dim)
        self.binder = CitationBinder()

    def forward(
        self,
        volume: Tensor,
        coordinates: Tensor,
        query: str,
        query_embedding: Tensor,
        interactions: InteractionBatch,
    ) -> TutorOutput:
        initial = torch.zeros(volume.shape[0], dtype=torch.long, device=volume.device)
        initial_embedding = self.ontology(initial)
        segmentation = self.segmenter(volume, initial_embedding, coordinates)
        concepts = segmentation.class_logits.argmax(dim=-1)
        concept_embedding = self.ontology(concepts)
        segmentation = self.segmenter(volume, concept_embedding, coordinates)
        results = self.retriever.retrieve(
            query, query_embedding, concept_embedding[0], self.config.retrieval_top_k
        )
        evidence_ids = [result.chunk.identifier for result in results]
        codes = [self.ontology.concepts[int(index)].code for index in concepts]
        answer_text = " ".join(result.chunk.text for result in results)
        answer = self.binder.bind(answer_text, evidence_ids, codes)
        mastery = self.tracer(
            interactions, self.ontology.embedding.weight, self.ontology.adjacency()
        )
        return TutorOutput(segmentation, answer, mastery, concepts)
