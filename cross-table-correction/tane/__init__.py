"""
TANE - Functional Dependency Discovery Algorithm
"""

try:
    from .tane import TANE, read_db
    __all__ = ['TANE', 'read_db']
except ImportError as e:
    raise ImportError(f"Failed to import TANE components. Please ensure 'fca==3.2' is installed. Error: {e}")
