from dataclasses import dataclass
from typing import List


@dataclass
class UtteranceMetadata:
    # Text data
    utterance_id: str
    phones: List[str]
    words: List[str]
    word_phone_offsets: List[int]
    # Phone scores
    phone_accuracy_scores: List[float]
    # Word scores
    word_accuracy_scores: List[float]
    word_stress_scores: List[float]
    word_total_scores: List[float]
    # Sentence scores
    sentence_accuracy_score: float
    sentence_completeness_score: float
    sentence_fluency_score: float
    sentence_prosodic_score: float
    sentence_total_score: float
