"""Feature extraction modules for SecondLife AI Pricing.

Includes:
- CLIP image feature extraction (openai/clip-vit-base-patch32)
- Multi-view appliance interaction embedding aggregation
- Vietnamese text processing & normalization (PhoBERT-compatible)
- PhoBERT text embedding extraction (vinai/phobert-base, 768-d)
"""

from src.features.extract_clip_embeddings import ClipFeatureExtractor
from src.features.aggregate_image_embeddings import ImageEmbeddingAggregator
from src.features.vietnamese_text_processor import VietnameseTextProcessor, SegmentationEngine
from src.features.extract_phobert_embeddings import PhoBertFeatureExtractor

__all__ = [
    "ClipFeatureExtractor",
    "ImageEmbeddingAggregator",
    "VietnameseTextProcessor",
    "SegmentationEngine",
    "PhoBertFeatureExtractor",
]
