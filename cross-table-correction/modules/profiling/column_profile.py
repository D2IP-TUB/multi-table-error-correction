"""
ColumnProfile dataclass
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class ColumnProfile:
    """
    Profile information for a column based on clean cells only.
    """

    table_id: str
    column_idx: int
    column_name: Optional[str] = None

    # === CARDINALITIES===
    uniqueness: float = 0.0  # unique_clean_values / total_clean_cells

    # === VALUE DISTRIBUTIONS ===
    value_histogram: Dict[str, int] = (
        None  # actual values -> frequency (only if uniqueness < 1)
    )

    # Numeric statistics (computed by casting string values)
    min_value: Optional[float] = None  # Minimum numeric value
    max_value: Optional[float] = None  # Maximum numeric value
    q1: Optional[float] = None  # First quartile
    q2: Optional[float] = None  # Median (second quartile)
    q3: Optional[float] = None  # Third quartile
    mean_value: Optional[float] = None  # Mean for numeric columns
    std_value: Optional[float] = None  # Standard deviation for numeric columns

    # === CONTEXTUAL DEPENDENCIES ===
    functional_dependencies: list = None  # FDs where this column is RHS
    fd_confidence_scores: Dict[tuple, float] = None  # (LHS_cols, RHS_col) -> confidence

    # === SEMANTIC PROFILES ===
    # For embedding-based similarity
    clean_value_embeddings: Dict[str, any] = None  # value -> embedding (if applicable)
    is_embeddable: bool = False  # whether values can be meaningfully embedded

    # For n-gram similarity (character level only)
    clean_char_ngrams: set = None  # character n-grams of clean values

    # === DATA TYPES, PATTERNS, AND DOMAINS ===
    basic_type: str = "varchar"  # Basic type (e.g., 'int', 'float', 'varchar')
    inferred_data_type: str = "varchar"  # DBMS-specific type

    # Pattern analysis
    mask_histogram: Dict[str, int] = None  # Unicode pattern masks -> frequency

    # Length statistics
    length_min: int = 0
    length_max: int = 0
    length_median: float = 0.0
    length_mu: float = 0.0  # Mean length
    length_sigma: float = 0.0  # Standard deviation of length

    # Numeric format analysis
    max_digits: int = 0  # Maximum number of digits
    max_decimals: int = 0  # Maximum number of decimal places
    has_negatives: bool = False  # Whether negative numbers are present
    has_scientific: bool = False  # Whether scientific notation is present

    def __post_init__(self):
        """Initialize empty dictionaries/sets if None"""
        if self.mask_histogram is None:
            self.mask_histogram = {}
        if self.value_histogram is None:
            self.value_histogram = {}
        if self.functional_dependencies is None:
            self.functional_dependencies = []
        if self.fd_confidence_scores is None:
            self.fd_confidence_scores = {}
        if self.clean_value_embeddings is None:
            self.clean_value_embeddings = {}
        if self.clean_char_ngrams is None:
            self.clean_char_ngrams = set()

    # === UTILITY METHODS ===

    def get_uniqueness(self) -> float:
        """Get uniqueness ratio (distinct values / total clean cells)"""
        return self.uniqueness

    def is_numeric(self) -> bool:
        """Check if column contains numeric data"""
        return self.inferred_data_type in [
            "int",
            "tinyint",
            "smallint",
            "bigint",
            "decimal",
            "float",
        ]

    def get_numeric_range(self) -> Tuple[Optional[float], Optional[float]]:
        """Get numeric range (min, max)"""
        return (self.min_value, self.max_value)

    def get_quartiles(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Get quartiles (Q1, Q2/median, Q3)"""
        return (self.q1, self.q2, self.q3)

    def get_pattern_diversity(self) -> int:
        """Get number of distinct patterns observed"""
        return len(self.mask_histogram)

    def get_dominant_pattern(self) -> Optional[str]:
        """Get most frequent pattern"""
        if not self.mask_histogram:
            return None
        return max(self.mask_histogram.items(), key=lambda x: x[1])[0]

    def get_length_range(self) -> Tuple[int, int]:
        """Get length range (min, max)"""
        return (self.length_min, self.length_max)

    def is_fixed_length(self) -> bool:
        """Check if all values have the same length"""
        return self.length_min == self.length_max

    def mask_prob(self, mask: str) -> float:
        """
        Get probability of a specific mask pattern.

        Args:
            mask: Pattern mask string

        Returns:
            Probability (0.0 if mask not found)
        """
        if not self.mask_histogram:
            return 0.0

        total_patterns = sum(self.mask_histogram.values())
        if total_patterns == 0:
            return 0.0

        count = self.mask_histogram.get(mask, 0)
        return count / total_patterns

    def value_prob(self, value: str) -> float:
        """
        Get probability of a specific value in categorical columns.

        Args:
            value: Value to check

        Returns:
            Probability (0.0 if value not found or column is 100% unique)
        """
        if not self.value_histogram or self.uniqueness == 1.0:
            return 0.0

        total_values = sum(self.value_histogram.values())
        if total_values == 0:
            return 0.0

        count = self.value_histogram.get(value, 0)
        return count / total_values

    def has_functional_dependencies(self) -> bool:
        """Check if this column participates in any functional dependencies"""
        return bool(self.functional_dependencies)

    def get_embedding(self, value: str) -> Optional[any]:
        """
        Get embedding for a specific value if available.

        Args:
            value: Value to get embedding for

        Returns:
            Embedding array or None if not available
        """
        if not self.clean_value_embeddings:
            return None
        return self.clean_value_embeddings.get(value)

    def has_embeddings(self) -> bool:
        """Check if this column has semantic embeddings available"""
        return self.is_embeddable and bool(self.clean_value_embeddings)

    def get_char_ngram_overlap(self, value: str, n: int = 2) -> float:
        """
        Get character n-gram overlap with clean values.

        Args:
            value: Value to check
            n: N-gram size

        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not self.clean_char_ngrams:
            return 0.0

        # Generate n-grams for the value
        value_ngrams = set()
        for i in range(len(value) - n + 1):
            value_ngrams.add(value[i : i + n])

        if not value_ngrams:
            return 0.0

        overlap = len(value_ngrams & self.clean_char_ngrams)
        return overlap / len(value_ngrams)

    def get_mask_percentile(self, mask: str) -> float:
        """
        Get percentile rank of mask probability among all masks.

        Args:
            mask: Pattern mask string

        Returns:
            Percentile (0.0 to 1.0)
        """
        if not self.mask_histogram:
            return 0.0

        mask_prob = self.mask_prob(mask)

        # Count how many masks have lower probability
        lower_count = 0
        total_count = len(self.mask_histogram)

        for other_mask, other_count in self.mask_histogram.items():
            total_patterns = sum(self.mask_histogram.values())
            other_prob = other_count / total_patterns if total_patterns > 0 else 0
            if other_prob < mask_prob:
                lower_count += 1

        return lower_count / total_count if total_count > 0 else 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "table_id": self.table_id,
            "column_idx": self.column_idx,
            "column_name": self.column_name,
            "uniqueness": self.uniqueness,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "q1": self.q1,
            "q2": self.q2,
            "q3": self.q3,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "basic_type": self.basic_type,
            "inferred_data_type": self.inferred_data_type,
            "mask_histogram": self.mask_histogram,
            "length_min": self.length_min,
            "length_max": self.length_max,
            "length_median": self.length_median,
            "length_mu": self.length_mu,
            "length_sigma": self.length_sigma,
            "max_digits": self.max_digits,
            "max_decimals": self.max_decimals,
            "has_negatives": self.has_negatives,
            "has_scientific": self.has_scientific,
            "value_histogram": self.value_histogram,
            "functional_dependencies": self.functional_dependencies,
            "fd_confidence_scores": self.fd_confidence_scores,
            "clean_value_embeddings": self.clean_value_embeddings,  # Note: may need special handling for numpy arrays
            "clean_char_ngrams": list(
                self.clean_char_ngrams
            ),  # Convert set to list for JSON
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ColumnProfile":
        """Create instance from dictionary"""
        return cls(**data)

    def __str__(self) -> str:
        return (
            f"ColumnProfile(table={self.table_id}, col={self.column_idx}, "
            f"uniqueness={self.uniqueness:.3f}, "
            f"patterns={len(self.mask_histogram)}, "
            f"type={self.basic_type},  "
        )
