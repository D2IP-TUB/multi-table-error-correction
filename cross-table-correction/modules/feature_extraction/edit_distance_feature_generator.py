from typing import List

from rapidfuzz.distance import Levenshtein

from core.candidate import Candidate
from core.cell import Cell


def get_edit_distance_features_batch(cell: Cell, candidates: List[Candidate]) -> None:
    cell_value = cell.value
    if not cell_value:
        for _, candidate_obj in candidates.items():
            candidate_obj.features["levenshtein_similarity"] = 0.0
        return

    try:
        similarity_func = Levenshtein.normalized_similarity
    except ImportError:

        def similarity_func(s1, s2):
            return 1 - Levenshtein.normalized_distance(s1, s2)

    seen = {}
    for _, candidate_obj in candidates.items():
        correction = candidate_obj.correction_value
        if correction not in seen:
            if not correction:
                seen[correction] = 0.0
            else:
                seen[correction] = similarity_func(cell_value, correction)

        candidate_obj.features["levenshtein_similarity"] = seen[correction]
