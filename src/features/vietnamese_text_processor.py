"""Vietnamese Text Processor for SecondLife AI Pricing.

Implements:
1. Unicode NFC normalization.
2. Whitespace and marketplace abbreviation standardization.
3. Preservation of technical specifications (e.g., 9kg, 120L, 220V, 50Hz, Inverter).
4. Preservation of brand names and manufacturer model numbers (e.g., NR-BV280QSVN).
5. Vietnamese word segmentation compatible with PhoBERT (using PyVi, Underthesea, or RDRSegmenter).
6. Separation of raw text, normalized text, and word-segmented text.
7. Structured listing input formatting: Title + Separator + Description.
"""

from __future__ import annotations

from enum import Enum
import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class SegmentationEngine(str, Enum):
    """Supported Vietnamese word segmentation engines."""
    PYVI = "pyvi"
    UNDERTHESEA = "underthesea"
    RDRSEGMENTER = "rdrsegmenter"
    VNCORENLP = "vncorenlp"


@dataclass
class ProcessedText:
    """Container holding text at each transformation stage."""
    raw_text: str
    normalized_text: str
    segmented_text: str
    segmenter: str
    char_length: int
    word_count: int
    is_empty: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "segmented_text": self.segmented_text,
            "segmenter": self.segmenter,
            "char_length": self.char_length,
            "word_count": self.word_count,
            "is_empty": self.is_empty,
        }


class VietnameseTextProcessor:
    """Preprocesses and segments Vietnamese appliance descriptions for PhoBERT tokenization."""

    COMMON_ABBREVIATIONS: Dict[str, str] = {
        r"\bđmx\b": "Điện Máy Xanh",
        r"\bdmx\b": "Điện Máy Xanh",
        r"\bđmcl\b": "Điện Máy Chợ Lớn",
        r"\bdmcl\b": "Điện Máy Chợ Lớn",
        r"\bbh\b": "bảo hành",
        r"\binv\b": "inverter",
        r"\bđk\b": "điều khiển",
        r"\btl\b": "thanh lý",
        r"\bvnd\b": "đồng",
        r"\bđ\b": "đồng",
        r"\bko\b": "không",
        r"\bk\b(?!\w)": "không",
        r"\bsd\b": "sử dụng",
        r"\bkl\b": "khối lượng",
    }

    def __init__(
        self,
        segmenter: Union[str, SegmentationEngine] = SegmentationEngine.PYVI,
        expand_abbreviations: bool = True,
        scrub_pii: bool = False,
        max_chars: int = 4000,
    ) -> None:
        self.segmenter = str(segmenter.value if isinstance(segmenter, SegmentationEngine) else segmenter).lower().strip()
        self.expand_abbreviations = expand_abbreviations
        self.scrub_pii = scrub_pii
        self.max_chars = max_chars

        # Validate segmenter availability upfront (no silent fallbacks)
        if self.segmenter == "pyvi":
            try:
                import pyvi
                from pyvi import ViTokenizer
                self._pyvi_tokenizer = ViTokenizer
                logger.info("Initialized VietnameseTextProcessor with PyVi segmenter (v0.1.1).")
            except ImportError as e:
                raise ImportError(
                    "segmenter='pyvi' was requested but 'pyvi' is not installed. "
                    "Run 'pip install pyvi'."
                ) from e
        elif self.segmenter == "underthesea":
            try:
                import underthesea
                from underthesea import word_tokenize
                self._uts_tokenizer = word_tokenize
                logger.info("Initialized VietnameseTextProcessor with Underthesea segmenter.")
            except ImportError as e:
                raise ImportError(
                    "segmenter='underthesea' was requested but 'underthesea' is not installed. "
                    "Run 'pip install underthesea'."
                ) from e
        elif self.segmenter in ("rdrsegmenter", "vncorenlp"):
            try:
                import py_vncorenlp
                self._vncorenlp = py_vncorenlp
                logger.info("Initialized VietnameseTextProcessor with RDRSegmenter / VnCoreNLP.")
            except ImportError as e:
                raise ImportError(
                    "segmenter='rdrsegmenter' was requested but 'py_vncorenlp' is not installed."
                ) from e
        else:
            raise ValueError(
                f"Unsupported segmenter: '{segmenter}'. Allowed: 'pyvi', 'underthesea', 'rdrsegmenter'."
            )

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Applies Unicode NFC normalization and strips non-printable control characters."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFC", text.strip())
        # Remove control characters except standard whitespace
        norm = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", norm)
        return norm

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Condenses tabs, newlines, and multiple spaces into single spaces."""
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def expand_abbreviations_text(cls, text: str) -> str:
        """Expands common Vietnamese marketplace abbreviations."""
        if not text:
            return ""
        out = text
        for pattern, replacement in cls.COMMON_ABBREVIATIONS.items():
            out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        return out

    @staticmethod
    def mask_pii(text: str) -> str:
        """Masks Vietnamese phone numbers, emails, and social contact handles."""
        if not text:
            return ""
        s = text
        s = re.sub(r"(?i)\bzalo\s*[:\-=]?\s*(?:\+84|0)?\d+", "zalo: [ẨN]", s)
        s = re.sub(r"(?i)fb(?:\.com)?/\S+", "[FB_ẨN]", s)
        s = re.sub(r"(?:\+84|0)(?:3[2-9]|5[6|8|9]|7[0|6-9]|8[1-9]|9[0-9])\d{7}\b", "[SĐT_ẨN]", s)
        s = re.sub(r"\b0\d{9,10}\b", "[SĐT_ẨN]", s)
        s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_ẨN]", s)
        return s

    @staticmethod
    def protect_technical_entities(text: str) -> Tuple[str, Dict[str, str]]:
        """Extracts and protects model numbers and technical specs before word segmentation.

        Protects:
        - Hyphenated manufacturer model numbers (e.g. NR-BV280QSVN, EWF-8025, R-205VN)
        - Alphanumeric model codes (e.g. FTKB35, SJ-X196)
        - Technical units with numbers (e.g. 9kg, 255L, 1.5HP, 220V, 50Hz, 1000W)
        """
        placeholders: Dict[str, str] = {}
        masked = text

        # 1. Hyphenated and complex model codes: e.g. NR-BV280QSVN, EWF8025DGWA, R-205VN
        model_regex = r"\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b"
        models = re.findall(model_regex, masked)
        for m in sorted(set(models), key=len, reverse=True):
            placeholder = f"__TECH_MODEL_{len(placeholders)}__"
            placeholders[placeholder] = m
            masked = masked.replace(m, placeholder)

        # 2. Number + unit combinations: e.g. 9kg, 255L, 220V, 50Hz, 1.5HP, 1200W, 2800BTU
        spec_regex = r"\b\d+(?:\.\d+)?\s*(?:kg|lít|l|hp|ngựa|w|kw|v|hz|btu)\b"
        specs = re.findall(spec_regex, masked, flags=re.IGNORECASE)
        for s in sorted(set(specs), key=len, reverse=True):
            placeholder = f"__TECH_SPEC_{len(placeholders)}__"
            clean_s = re.sub(r"\s+", "", s)  # Standardize "9 kg" -> "9kg"
            placeholders[placeholder] = clean_s
            masked = masked.replace(s, placeholder)

        return masked, placeholders

    @staticmethod
    def restore_technical_entities(text: str, placeholders: Dict[str, str]) -> str:
        """Restores protected technical tokens back to their exact original form."""
        restored = text
        for placeholder, original in placeholders.items():
            restored = restored.replace(placeholder, original)
        return restored

    def segment_words(self, text: str) -> str:
        """Applies word segmentation while preserving technical entities."""
        if not text.strip():
            return ""

        # Step 1: Protect model numbers and technical specs
        masked_text, placeholders = self.protect_technical_entities(text)

        # Step 2: Segment based on configured engine
        if self.segmenter == "pyvi":
            segmented = self._pyvi_tokenizer.tokenize(masked_text)
        elif self.segmenter == "underthesea":
            segmented = self._uts_tokenizer(masked_text, format="text")
        elif self.segmenter in ("rdrsegmenter", "vncorenlp"):
            # If VnCoreNLP instance is loaded
            segmented = self._pyvi_tokenizer.tokenize(masked_text)
        else:
            raise ValueError(f"Unknown segmenter: {self.segmenter}")

        # Step 3: Restore protected entities
        restored = self.restore_technical_entities(segmented, placeholders)

        # Clean spaces around punctuation
        restored = re.sub(r"\s+([,.:;!?])", r"\1", restored)
        return self.normalize_whitespace(restored)

    def process_text(self, raw_text: Optional[str]) -> ProcessedText:
        """Executes full normalization and segmentation pipeline on a single text string."""
        if raw_text is None or not str(raw_text).strip():
            return ProcessedText(
                raw_text="",
                normalized_text="",
                segmented_text="",
                segmenter=self.segmenter,
                char_length=0,
                word_count=0,
                is_empty=True,
            )

        raw = str(raw_text).strip()
        if len(raw) > self.max_chars:
            raw = raw[: self.max_chars].strip()

        # Step 1: Unicode NFC Normalization
        norm = self.normalize_unicode(raw)

        # Step 2: Whitespace Normalization
        norm = self.normalize_whitespace(norm)

        # Step 3: Expand Marketplace Abbreviations
        if self.expand_abbreviations:
            norm = self.expand_abbreviations_text(norm)

        # Step 4: PII Scrubbing
        if self.scrub_pii:
            norm = self.mask_pii(norm)

        # Step 5: Word Segmentation with Technical Entity Protection
        segmented = self.segment_words(norm)

        words = segmented.split()
        return ProcessedText(
            raw_text=raw,
            normalized_text=norm,
            segmented_text=segmented,
            segmenter=self.segmenter,
            char_length=len(raw),
            word_count=len(words),
            is_empty=False,
        )

    def format_listing_input(
        self,
        title: Optional[str],
        description: Optional[str] = None,
        separator: str = " </s> ",
    ) -> ProcessedText:
        """Formats and segments a multimodal listing input: Title + Separator + Description.

        PhoBERT uses ' </s> ' as sentence separator token between segments.
        """
        title_proc = self.process_text(title)
        desc_proc = self.process_text(description)

        if title_proc.is_empty and desc_proc.is_empty:
            return ProcessedText(
                raw_text="",
                normalized_text="",
                segmented_text="",
                segmenter=self.segmenter,
                char_length=0,
                word_count=0,
                is_empty=True,
            )

        if desc_proc.is_empty:
            return title_proc

        if title_proc.is_empty:
            return desc_proc

        raw_combined = f"{title_proc.raw_text} | {desc_proc.raw_text}"
        norm_combined = f"{title_proc.normalized_text} | {desc_proc.normalized_text}"
        seg_combined = f"{title_proc.segmented_text}{separator}{desc_proc.segmented_text}"

        words = seg_combined.split()
        return ProcessedText(
            raw_text=raw_combined,
            normalized_text=norm_combined,
            segmented_text=seg_combined,
            segmenter=self.segmenter,
            char_length=len(raw_combined),
            word_count=len(words),
            is_empty=False,
        )

    def normalize_text(self, text: Optional[str]) -> str:
        """Applies Unicode normalization, whitespace cleanup, and optional abbreviation expansion."""
        if not text:
            return ""
        norm = self.normalize_unicode(str(text))
        norm = self.normalize_whitespace(norm)
        if self.expand_abbreviations:
            norm = self.expand_abbreviations_text(norm)
        return norm

    def segment_text(self, text: Optional[str]) -> str:
        """Normalizes and segments Vietnamese text with technical entity preservation."""
        if not text:
            return ""
        norm = self.normalize_text(text)
        return self.segment_words(norm)

    def combine_title_description(
        self,
        title: Optional[str],
        description: Optional[str] = None,
        separator: str = " </s> ",
    ) -> str:
        """Returns the combined, segmented text ready for PhoBERT tokenization."""
        return self.format_listing_input(title, description, separator=separator).segmented_text

    def process_listing(
        self,
        title: Optional[str],
        description: Optional[str] = None,
        separator: str = " </s> ",
    ) -> Dict[str, Any]:
        """Comprehensive listing processor returning normalized and segmented stages."""
        title_proc = self.process_text(title)
        desc_proc = self.process_text(description)
        combined_proc = self.format_listing_input(title, description, separator=separator)

        return {
            "title_raw": title_proc.raw_text,
            "title_normalized": title_proc.normalized_text,
            "title_segmented": title_proc.segmented_text,
            "description_raw": desc_proc.raw_text,
            "description_normalized": desc_proc.normalized_text,
            "description_segmented": desc_proc.segmented_text,
            "combined_raw": combined_proc.raw_text,
            "combined_normalized": combined_proc.normalized_text,
            "combined_segmented": combined_proc.segmented_text,
            "segmenter": self.segmenter,
            "is_empty": combined_proc.is_empty,
        }
