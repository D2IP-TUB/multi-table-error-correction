"""
Simple BERT-based embedding module with caching.
"""

import hashlib
import logging
import os
import warnings
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Suppress transformer warnings about gamma/beta parameter names
warnings.filterwarnings("ignore", message=".*gamma.*weight.*")
warnings.filterwarnings("ignore", message=".*beta.*bias.*")

# Global model cache
_TOKENIZER = None
_MODEL = None

# Global embedding cache
_EMBEDDING_CACHE = {}


def get_embedder(model_name="bert-base-uncased"):
    """
    Get BERT embedder.
    
    Args:
        model_name: Name of BERT model to use
        
    Returns:
        BertEmbedder instance
    """
    if os.environ.get('DISABLE_EMBEDDINGS', '0') == '1':
        logger.info("Embeddings disabled")
        return None
    
    return BertEmbedder(model_name)


class BertEmbedder:
    """Simple BERT embedder with model and embedding caching."""
    
    def __init__(self, model_name="bert-base-uncased"):
        self.model_name = model_name
        self._load_model()
        # Get embedding dimension from model config
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"BertEmbedder initialized with {model_name}, embedding_dim={self.embedding_dim}")
    
    def _load_model(self):
        """Load BERT model and tokenizer once."""
        global _TOKENIZER, _MODEL
        
        if _MODEL is not None:
            self.tokenizer = _TOKENIZER
            self.model = _MODEL
            return
        
        try:
            from transformers import AutoModel, AutoTokenizer
            
            cache_dir = os.path.expanduser('~/.cache/huggingface')
            logger.info(f"Loading BERT model: {self.model_name}")
            
            _TOKENIZER = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cache_dir)
            _MODEL = AutoModel.from_pretrained(self.model_name, cache_dir=cache_dir)
            _MODEL.eval()
            _MODEL = _MODEL.cpu()
            
            self.tokenizer = _TOKENIZER
            self.model = _MODEL
            logger.info("BERT model loaded and cached")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to BERT embeddings using mean pooling.
        Results are cached.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        import torch

        # Check cache for each text
        texts_to_compute = []
        text_indices = []
        cached_embeddings = {}
        
        for i, text in enumerate(texts):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            if text_hash in _EMBEDDING_CACHE:
                cached_embeddings[i] = _EMBEDDING_CACHE[text_hash]
            else:
                texts_to_compute.append(text)
                text_indices.append((i, text_hash))
        
        # If all cached, return immediately
        if not texts_to_compute:
            result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for idx, embedding in cached_embeddings.items():
                result[idx] = embedding
            return result
        
        # Compute missing embeddings
        try:
            inputs = self.tokenizer(
                texts_to_compute,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            inputs = {k: v.cpu() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Mean pooling over token dimension
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy().astype(np.float32)
            
            # Cache and build result
            result = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            for idx, (orig_idx, text_hash) in enumerate(text_indices):
                embedding = embeddings[idx]
                _EMBEDDING_CACHE[text_hash] = embedding
                result[orig_idx] = embedding
            
            # Add cached embeddings
            for idx, embedding in cached_embeddings.items():
                result[idx] = embedding
            
            logger.debug(f"Encoded {len(texts)} texts (computed {len(texts_to_compute)}, cached {len(cached_embeddings)})")
            return result
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
