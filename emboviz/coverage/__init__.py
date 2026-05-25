"""Coverage analysis — does the training set actually contain the kind of
data needed for the model to learn the failing axis?

Currently text-based (parses dataset task descriptions); a future
vision-based analyzer will use scene embeddings.
"""

from emboviz.coverage.text_analyzer import analyze_dataset_coverage
from emboviz.coverage.gap_detector import CoverageGap, CoverageReport, detect_gaps

__all__ = ["analyze_dataset_coverage", "detect_gaps", "CoverageGap", "CoverageReport"]
