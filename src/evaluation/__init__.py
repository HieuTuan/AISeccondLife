"""Evaluation package for SecondLife AI Pricing.

Includes:
- Zero-shot CLIP categorization evaluation
- CLIP and PhoBERT performance benchmarking engine
"""

from src.evaluation.benchmark_features import (
    FeatureBenchmarkEngine,
    SystemHardwareInfo,
    BenchmarkMetricPoint,
    run_benchmark_pipeline,
)

__all__ = [
    "FeatureBenchmarkEngine",
    "SystemHardwareInfo",
    "BenchmarkMetricPoint",
    "run_benchmark_pipeline",
]
