"""PhoBERT Vietnamese Text Feature Extraction Pipeline for SecondLife AI Pricing.

Extracts contextual 768-dimensional sentence embeddings using pretrained
vinai/phobert-base with attention-masked mean pooling and optional L2 normalization.
Processes titles and descriptions, tracks truncation events, preserves row ordering,
and persists embeddings as NumPy arrays and mapping metadata as CSV.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.features.vietnamese_text_processor import ProcessedText, VietnameseTextProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PhoBertFeatureExtractor:
    """Extracts 768-dimensional embeddings from Vietnamese appliance descriptions."""

    def __init__(
        self,
        model_name: str = "vinai/phobert-base",
        segmenter: str = "pyvi",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 256,
        pooling_strategy: str = "mean",
        normalize: bool = True,
        normalize_l2: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.pooling_strategy = pooling_strategy.lower().strip()
        self.normalize = normalize_l2 if normalize_l2 is not None else normalize

        if self.pooling_strategy not in ("mean", "cls"):
            raise ValueError(f"Unsupported pooling_strategy: '{pooling_strategy}'. Allowed: 'mean', 'cls'.")

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Initializing VietnameseTextProcessor with segmenter='{segmenter}'...")
        self.text_processor = VietnameseTextProcessor(segmenter=segmenter, expand_abbreviations=True)

        logger.info(f"Loading PhoBERT model and tokenizer '{model_name}' on device '{self.device}'...")
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.load_duration_seconds = time.time() - t0
        logger.info(f"PhoBERT model loaded in {self.load_duration_seconds:.2f}s.")

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Computes SHA-256 hash of a file for lineage audit."""
        p = Path(file_path)
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def pool_hidden_states(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Applies configured pooling strategy across token hidden representations.

        Strategies:
        1. 'mean': Excludes padding tokens using the attention mask:
           v = sum(h_i * mask_i) / sum(mask_i)
        2. 'cls': Takes the <s> / CLS token representation at index 0.
        """
        if self.pooling_strategy == "cls":
            return last_hidden_state[:, 0, :]

        # Mean pooling with explicit attention mask exclusion
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        return sum_embeddings / sum_mask

    def extract_from_texts(
        self,
        segmented_texts: List[str],
    ) -> Tuple[np.ndarray, List[bool]]:
        """Extracts 768-d embeddings for a batch of pre-segmented Vietnamese strings.

        Returns:
            Tuple of:
            - embeddings: NumPy array of shape (N, 768), float32.
            - is_truncated_flags: List of booleans indicating if each text exceeded max_length.
        """
        if not segmented_texts:
            return np.empty((0, 768), dtype=np.float32), []

        # Tokenize with truncation and padding
        encoded = self.tokenizer(
            segmented_texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Detect truncation by comparing against untruncated token length
        is_truncated_flags = []
        for text in segmented_texts:
            untruncated_tokens = self.tokenizer.encode(text, truncation=False)
            is_truncated_flags.append(len(untruncated_tokens) > self.max_length)

        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state  # [B, S, 768]
            pooled = self.pool_hidden_states(last_hidden_state, attention_mask)

            if self.normalize:
                pooled = pooled / pooled.norm(dim=-1, keepdim=True)

        embeddings = pooled.cpu().numpy().astype(np.float32)
        return embeddings, is_truncated_flags

    def extract_single(
        self,
        title: Optional[str],
        description: Optional[str] = None,
        separator: str = " </s> ",
    ) -> np.ndarray:
        """Extracts 768-d embedding for a single listing title and description.

        Returns:
            1D NumPy array of shape (768,), dtype float32.
        """
        proc = self.text_processor.format_listing_input(title, description, separator=separator)
        embs, _ = self.extract_from_texts([proc.segmented_text])
        return embs[0]

    def extract_from_dataframe(
        self,
        df: pd.DataFrame,
        title_col: str = "title",
        description_col: str = "description",
        separator: str = " </s> ",
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
        """Processes tabular listing texts, runs batched inference, returns embeddings and mapping."""
        bs = batch_size if batch_size is not None else self.batch_size
        start_time = time.time()

        processed_records: List[ProcessedText] = []
        for _, row in df.iterrows():
            title = row.get(title_col, None)
            desc = row.get(description_col, None)
            proc = self.text_processor.format_listing_input(title, desc, separator=separator)
            processed_records.append(proc)

        segmented_texts = [p.segmented_text for p in processed_records]
        num_samples = len(segmented_texts)

        # Batched inference
        all_embeddings: List[np.ndarray] = []
        all_truncated_flags: List[bool] = []

        num_batches = (num_samples + bs - 1) // bs
        logger.info(f"Extracting PhoBERT embeddings across {num_batches} batches (batch_size={bs})...")

        for b in range(num_batches):
            batch_texts = segmented_texts[b * bs : (b + 1) * bs]
            batch_embeds, batch_trunc = self.extract_from_texts(batch_texts)
            all_embeddings.append(batch_embeds)
            all_truncated_flags.extend(batch_trunc)

        duration = time.time() - start_time
        final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.empty((0, 768), dtype=np.float32)

        # Verify array invariants
        assert final_embeddings.shape == (num_samples, 768), f"Unexpected shape {final_embeddings.shape}"
        assert not np.isnan(final_embeddings).any(), "NaN detected in extracted PhoBERT embeddings!"
        assert not np.isinf(final_embeddings).any(), "Inf detected in extracted PhoBERT embeddings!"

        if self.normalize:
            norms = np.linalg.norm(final_embeddings, axis=1)
            assert np.all(np.abs(norms - 1.0) < 1e-4), "Embeddings not unit normalized!"

        # Build mapping DataFrame preserving original row metadata
        mapping_rows = []
        for idx, (p_text, is_trunc) in enumerate(zip(processed_records, all_truncated_flags)):
            orig_row = df.iloc[idx].to_dict()
            record_entry = {
                "embedding_index": idx,
                "is_truncated": is_trunc,
                "token_count": p_text.word_count,
                "segmented_input": p_text.segmented_text,
                "normalized_text": p_text.normalized_text,
            }
            # Carry over original metadata columns
            for col in df.columns:
                if col not in ("embedding_index", "is_truncated"):
                    record_entry[col] = orig_row.get(col, None)
            mapping_rows.append(record_entry)

        df_mapping = pd.DataFrame(mapping_rows)

        metadata = {
            "model_name": self.model_name,
            "model_revision": getattr(self.model.config, "_commit_hash", "main"),
            "device": self.device,
            "segmenter": self.text_processor.segmenter,
            "pooling_strategy": self.pooling_strategy,
            "normalized_l2": self.normalize,
            "max_sequence_length": self.max_length,
            "batch_size": bs,
            "total_samples": num_samples,
            "truncated_samples": int(sum(all_truncated_flags)),
            "truncation_rate": round(float(sum(all_truncated_flags) / max(1, num_samples)), 4),
            "embedding_shape": list(final_embeddings.shape),
            "embedding_dtype": str(final_embeddings.dtype),
            "inference_duration_seconds": round(duration, 3),
            "throughput_samples_per_second": round(num_samples / max(0.001, duration), 2),
            "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        return final_embeddings, df_mapping, metadata

    def extract_from_file_and_save(
        self,
        input_csv: str | Path,
        output_dir: str | Path = "data/embeddings",
        prefix: str = "phobert_demo",
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Loads input CSV, extracts embeddings, and saves .npy, .csv mapping, and .json metadata."""
        in_p = Path(input_csv)
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        if not in_p.exists():
            raise FileNotFoundError(f"Input file not found: {in_p}")

        file_hash = self.compute_sha256(in_p)
        df_input = pd.read_csv(in_p)
        logger.info(f"Loaded {len(df_input)} rows from {in_p.name} (SHA-256: {file_hash[:12]}...)")

        embeddings, df_mapping, meta = self.extract_from_dataframe(
            df=df_input,
            batch_size=batch_size,
        )

        out_npy = out_d / f"{prefix}_embeddings.npy"
        np.save(out_npy, embeddings)
        logger.info(f"Saved PhoBERT embeddings to {out_npy}")

        out_csv = out_d / f"{prefix}_mapping.csv"
        df_mapping.to_csv(out_csv, index=False, encoding="utf-8")
        logger.info(f"Saved PhoBERT mapping to {out_csv}")

        meta["source_file"] = str(in_p.resolve())
        meta["source_sha256"] = file_hash
        meta["output_embeddings_file"] = str(out_npy.resolve())
        meta["output_mapping_file"] = str(out_csv.resolve())

        meta_file = out_d / f"{prefix}_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved extraction metadata to {meta_file}")

        return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 768-d PhoBERT Vietnamese text embeddings.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/fixtures/vietnamese_appliance_descriptions.csv",
        help="Path to input CSV containing title and description",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Directory to save embeddings and mapping",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="phobert_demo",
        help="Prefix for output filenames",
    )
    parser.add_argument(
        "--segmenter",
        type=str,
        default="pyvi",
        choices=["pyvi", "underthesea"],
        help="Vietnamese word segmenter",
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Token pooling strategy",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device (cuda or cpu)",
    )
    args = parser.parse_args()

    extractor = PhoBertFeatureExtractor(
        segmenter=args.segmenter,
        pooling_strategy=args.pooling,
        device=args.device,
        batch_size=args.batch_size,
    )
    meta = extractor.extract_from_file_and_save(
        input_csv=args.input,
        output_dir=args.output_dir,
        prefix=args.prefix,
        batch_size=args.batch_size,
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
