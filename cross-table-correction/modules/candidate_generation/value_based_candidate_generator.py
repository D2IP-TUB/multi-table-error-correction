import difflib
import json
import logging
import unicodedata
from collections import defaultdict
from typing import Dict, List, Optional

from core.candidate import Candidate
from core.cell import Cell
from core.table import Table
from modules.candidate_generation.candidate_generator import CandidateGenerator


class ValueBasedCorrector(CandidateGenerator):
    """
    value corrector .
    """

    def __init__(self, config):
        super().__init__(config)
        self.value_encodings = getattr(
            config.correction, "value_encodings", ["identity", "unicode"]
        )
        self.max_value_length = getattr(config.correction, "max_value_length", 50)
        self.min_correction_occurrence = getattr(
            config.correction, "min_correction_occurrence", 2
        )

        self.labeled_samples = []
        self._transformation_cache = {}  # Cache for transformation computations
        self._encoding_cache = {}  # Cache for value encodings
        self._models = None  # Single set of models shared across all error values
        self._models_built = False  # Flag to track if models have been built

    def update_from_labeled_samples(self, samples: List[Cell]):
        """Store and pre-process labeled samples for model building"""
        self.labeled_samples.extend(samples)

        # Pre-compute transformations for all sample pairs
        logging.info(f"Pre-computing transformations for {len(samples)} samples...")
        for sample_cell in samples:
            if hasattr(sample_cell, "ground_truth") and self._is_valid_training_pair(
                sample_cell.value, sample_cell.ground_truth
            ):
                old_value = sample_cell.value
                new_value = sample_cell.ground_truth

                # Cache transformation
                pair_key = (old_value, new_value)
                if pair_key not in self._transformation_cache:
                    self._transformation_cache[pair_key] = (
                        self._generate_transformations(old_value, new_value)
                    )

                # Cache encodings
                for encoding in self.value_encodings:
                    enc_key = (old_value, encoding)
                    if enc_key not in self._encoding_cache:
                        self._encoding_cache[enc_key] = self._encode_value(
                            old_value, encoding
                        )
        
        # Invalidate models cache when new samples are added
        self._models_built = False
        self._models = None

    def _build_models(self) -> List[Dict]:
        """
        Build a single set of models from all labeled samples.
        These models are shared across all error values.
        Models are built once and cached for efficiency.
        """
        if self._models_built and self._models is not None:
            return self._models

        models = [{}, {}, {}, {}]  # remover, adder, replacer, swapper

        # Build models from all cached transformations
        logging.info(f"Building value-based correction models from {len(self.labeled_samples)} samples...")
        for sample_cell in self.labeled_samples:
            if hasattr(sample_cell, "ground_truth") and self._is_valid_training_pair(
                sample_cell.value, sample_cell.ground_truth
            ):
                old_value = sample_cell.value
                new_value = sample_cell.ground_truth

                # Get cached transformation
                pair_key = (old_value, new_value)
                if pair_key in self._transformation_cache:
                    transformations = self._transformation_cache[pair_key]
                    self._update_models(models, old_value, new_value, transformations)
        
        # Cache the built models
        self._models = models
        self._models_built = True
        logging.info(f"Models built successfully with {len(self._transformation_cache)} transformations cached")
        return models

    def _update_models(
        self, models: List[Dict], old_value: str, new_value: str, transformations: Dict
    ):
        """Efficiently update models using cached encodings"""

        for encoding in self.value_encodings:
            # Get cached encoding
            enc_key = (old_value, encoding)
            if enc_key in self._encoding_cache:
                encoded_old = self._encoding_cache[enc_key]
            else:
                encoded_old = self._encode_value(old_value, encoding)
                self._encoding_cache[enc_key] = encoded_old

            # Update transformation models
            if transformations["remover"]:
                self._add_to_model(
                    models[0], encoded_old, json.dumps(transformations["remover"])
                )
            if transformations["adder"]:
                self._add_to_model(
                    models[1], encoded_old, json.dumps(transformations["adder"])
                )
            if transformations["replacer"]:
                self._add_to_model(
                    models[2], encoded_old, json.dumps(transformations["replacer"])
                )

            # Update swapper model
            self._add_to_model(models[3], encoded_old, new_value)

    def generate_candidates(self, cell: Cell, table: Table) -> Dict[str, Candidate]:
        """
        Generate candidates using shared models.
        """
        if not self.labeled_samples:
            return {}

        # Use shared model (built once from all samples)
        models = self._build_models()
        old_value = cell.value

        # Batch process all model+encoding combinations
        correction_probabilities = defaultdict(dict)
        corrections_source_model = defaultdict(dict)

        for model_idx, model_name in enumerate(
            ["remover", "adder", "replacer", "swapper"]
        ):
            model = models[model_idx]

            for encoding in self.value_encodings:
                # Use cached encoding or compute it
                enc_key = (old_value, encoding)
                if enc_key in self._encoding_cache:
                    encoded_value = self._encoding_cache[enc_key]
                else:
                    encoded_value = self._encode_value(old_value, encoding)
                    self._encoding_cache[enc_key] = encoded_value

                feature_name = f"{model_name}_{encoding}"

                if encoded_value not in model:
                    continue

                sum_scores = sum(model[encoded_value].values())
                if sum_scores == 0:
                    continue

                if model_name in ["remover", "adder", "replacer"]:
                    # Batch process transformations
                    for transformation_str, score in model[encoded_value].items():
                        probability = score / sum_scores
                        if probability < self.min_probability:
                            continue

                        new_value = self._apply_single_transformation(
                            old_value, transformation_str, model_name
                        )

                        if new_value and new_value != old_value:
                            correction_probabilities[new_value][feature_name] = (
                                probability
                            )
                            if new_value not in corrections_source_model:
                                corrections_source_model[new_value] = {
                                    "value_based": []
                                }
                            corrections_source_model[new_value]["value_based"].append(
                                (model_name, encoding)
                            )

                elif model_name == "swapper":
                    for new_value, score in model[encoded_value].items():
                        probability = score / sum_scores
                        if probability >= self.min_probability:
                            correction_probabilities[new_value][feature_name] = (
                                probability
                            )
                            if new_value not in corrections_source_model:
                                corrections_source_model[new_value] = {
                                    "value_based": []
                                }
                            corrections_source_model[new_value]["value_based"].append(
                                (model_name, encoding)
                            )

        # Efficiently create candidates
        candidates = {}
        for correction, model_probs in correction_probabilities.items():
            features = {
                **{
                    f"{model}_{enc}": 0.0
                    for model in ["remover", "adder", "replacer", "swapper"]
                    for enc in self.value_encodings
                },
            }

            features.update(model_probs)
            candidate = Candidate(
                correction,
                features,
                candidates_source_model=corrections_source_model[correction],
            )
            
            # Add to pool and store reference
            pool_key = self._add_candidate_to_pool(
                table.table_id, cell.column_idx, correction, candidate
            )
            candidates[correction] = pool_key

        return candidates

    def _is_valid_training_pair(self, old_value: str, new_value: str) -> bool:
        """Check if training pair is valid.
        
        Filters out:
        - Pairs where values are missing or identical
        - Values exceeding max length
        - 'N/A' values (case-insensitive)
        - Values starting with digits (likely IDs/codes)
        """
        # Check for None or empty values
        if not old_value or not new_value:
            return False
        
        # Check values are different
        if old_value == new_value:
            return False
        
        # Check length constraints
        if len(old_value) > self.max_value_length or len(new_value) > self.max_value_length:
            return False
        
        # Filter out N/A values
        if old_value.lower() == "n/a" or old_value == "":
            return False
        
        # Filter out values starting with digits (likely IDs/codes)
        if old_value and old_value[0].isdigit():
            return False
        
        return True

    def _generate_transformations(
        self, old_value: str, new_value: str
    ) -> Dict[str, Dict]:
        """Generate transformation dictionaries using difflib"""
        transformations = {"remover": {}, "adder": {}, "replacer": {}}

        s = difflib.SequenceMatcher(None, old_value, new_value)
        for tag, i1, i2, j1, j2 in s.get_opcodes():
            index_range = json.dumps([i1, i2])

            if tag == "delete":
                transformations["remover"][index_range] = ""
            elif tag == "insert":
                transformations["adder"][index_range] = new_value[j1:j2]
            elif tag == "replace":
                transformations["replacer"][index_range] = new_value[j1:j2]

        return transformations

    def _encode_value(self, value: str, encoding: str) -> str:
        """Encode value using specified encoding method"""
        if encoding == "identity":
            return json.dumps(list(value))
        elif encoding == "unicode":
            return json.dumps([unicodedata.category(c) for c in value])
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    def _apply_single_transformation(
        self, old_value: str, transformation_str: str, model_name: str
    ) -> Optional[str]:
        """Apply a single transformation to generate new value"""
        try:
            transformation = json.loads(transformation_str)
            index_character_dictionary = {i: c for i, c in enumerate(old_value)}

            for change_range_string in transformation:
                change_range = json.loads(change_range_string)

                if model_name in ["remover", "replacer"]:
                    for i in range(change_range[0], change_range[1]):
                        index_character_dictionary[i] = ""

                if model_name in ["adder", "replacer"]:
                    ov = (
                        ""
                        if change_range[0] not in index_character_dictionary
                        else index_character_dictionary[change_range[0]]
                    )
                    index_character_dictionary[change_range[0]] = (
                        transformation[change_range_string] + ov
                    )

            return "".join(
                index_character_dictionary[i]
                for i in range(len(index_character_dictionary))
            )

        except (json.JSONDecodeError, KeyError, IndexError):
            return None

    def get_strategy_name(self) -> str:
        return "value_based"
