"""
Global candidate pool manager to deduplicate candidates across cells.

Each candidate is identified by (table_id, column_idx, value) and stored once.
Cells store references to candidates instead of full objects.
"""

import logging
from typing import Dict, Tuple, Optional

from core.candidate import Candidate


class CandidatePool:
    """Global pool for storing unique candidates by (table_id, column_idx, value)"""
    
    _instance = None  # Singleton instance
    
    def __init__(self):
        """Initialize the candidate pool"""
        self._pool: Dict[Tuple[str, int, str], Candidate] = {}
        self._stats = {
            "added": 0,
            "reused": 0,
            "total_unique": 0,
        }
    
    @classmethod
    def get_instance(cls) -> "CandidatePool":
        """Get or create singleton instance"""
        if cls._instance is None:
            cls._instance = CandidatePool()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the singleton (for testing or starting fresh)"""
        cls._instance = None
    
    def add_candidate(
        self, 
        table_id: str, 
        column_idx: int, 
        value: str, 
        candidate: Candidate
    ) -> Tuple[str, int, str]:
        """
        Add a candidate to the pool or reuse existing one.
        
        Args:
            table_id: ID of the table
            column_idx: Column index in the table
            value: The candidate correction value
            candidate: The Candidate object
            
        Returns:
            The candidate key (table_id, column_idx, value)
        """
        key = (table_id, column_idx, value)
        
        if key in self._pool:
            self._stats["reused"] += 1
        else:
            self._pool[key] = candidate
            self._stats["added"] += 1
            self._stats["total_unique"] = len(self._pool)
        
        return key
    
    def get_candidate(self, key: Tuple[str, int, str]) -> Optional[Candidate]:
        """Get a candidate from the pool by key"""
        return self._pool.get(key)
    
    def get_candidate_value(self, key: Tuple[str, int, str]) -> str:
        """Get the value from the candidate key"""
        return key[2]  # value is the third element
    
    def size(self) -> int:
        """Get the current number of unique candidates in the pool"""
        return len(self._pool)
    
    def get_stats(self) -> dict:
        """Get statistics about the pool"""
        return {
            **self._stats,
            "pool_size": len(self._pool),
        }
    
    def log_stats(self):
        """Log pool statistics"""
        stats = self.get_stats()
        logging.info(
            f"Candidate Pool Statistics: "
            f"Added={stats['added']}, "
            f"Reused={stats['reused']}, "
            f"Unique={stats['total_unique']}"
        )
    
    def clear(self):
        """Clear the pool"""
        self._pool.clear()
        self._stats = {"added": 0, "reused": 0, "total_unique": 0}
