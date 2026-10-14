"""CLIP Image Feature Extraction Pipeline for SecondLife AI Pricing.

Extracts projected 512-dimensional, L2-normalized image embeddings using
openai/clip-vit-base-patch32 with batched torch.inference_mode().
Verifies image integrity, preserves deterministic row ordering, tracks source
provenance back to interaction IDs, and exports embeddings as NumPy arrays and
mapping metadata as CSV.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ClipFeatureExtractor:
    """Extracts 512-dimensional L2-normalized embeddings using pretrained CLIP."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading CLIP model '{model_name}' on device '{self.device}'...")
        t0 = time.time()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.load_duration_seconds = time.time() - t0
        logger.info(f"CLIP model loaded in {self.load_duration_seconds:.2f}s.")

    @staticmethod
    def _extract_tensor(output: Any) -> torch.Tensor:
        """Extracts feature tensor from model output across transformers versions."""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def compute_sha256(file_path: Path | str) -> str:
        """Computes SHA-256 hash of a file for lineage tracking."""
        p = Path(file_path)
        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def extract_from_images(
        self,
        images: List[Image.Image],
    ) -> np.ndarray:
        """Extracts L2-normalized embeddings for a list of PIL Images."""
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            raw_out = self.model.get_image_features(**inputs)
            tensor_features = self._extract_tensor(raw_out)
            # L2 normalization
            normalized = tensor_features / tensor_features.norm(dim=-1, keepdim=True)

        return normalized.cpu().numpy().astype(np.float32)

    def extract_from_manifest(
        self,
        manifest_path: str | Path,
        output_dir: str | Path = "data/embeddings",
        image_root: Optional[str | Path] = None,
        batch_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Loads dataset manifest, runs batched inference, saves .npy and .csv mapping.

        Returns:
            Dictionary containing extraction metadata and file locations.
        """
        bs = batch_size if batch_size is not None else self.batch_size
        manifest_p = Path(manifest_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not manifest_p.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_p}")

        manifest_hash = self.compute_sha256(manifest_p)
        df_manifest = pd.read_csv(manifest_p)
        total_records = len(df_manifest)
        logger.info(f"Loaded manifest with {total_records} records (SHA-256: {manifest_hash[:12]}...)")

        # Step 1: Verify Image Integrity and Resolve Paths
        verified_items: List[Tuple[int, pd.Series, Path]] = []
        failed_items: List[Dict[str, Any]] = []

        root_p = Path(image_root) if image_root else Path(".")

        for idx, row in df_manifest.iterrows():
            cand_paths = []
            if "local_path" in row and pd.notna(row["local_path"]):
                cand_paths.append(Path(str(row["local_path"])))
                cand_paths.append(root_p / str(row["local_path"]))
            if "repo_path" in row and pd.notna(row["repo_path"]):
                cand_paths.append(root_p / str(row["repo_path"]))
                cand_paths.append(Path(str(row["repo_path"])))
                cand_paths.append(Path("data/clip_sample") / str(row["repo_path"]))

            resolved_path: Optional[Path] = None
            for cp in cand_paths:
                if cp.exists() and cp.is_file():
                    resolved_path = cp
                    break

            if resolved_path is None:
                failed_items.append({
                    "row_index": idx,
                    "image_id": row.get("image_id", f"ROW_{idx}"),
                    "interaction_id": row.get("interaction_id", None),
                    "error": "File does not exist on disk",
                })
                continue

            # Verify PIL readability
            try:
                with Image.open(resolved_path) as test_img:
                    test_img.verify()
                verified_items.append((idx, row, resolved_path))
            except Exception as e:
                failed_items.append({
                    "row_index": idx,
                    "image_id": row.get("image_id", f"ROW_{idx}"),
                    "interaction_id": row.get("interaction_id", None),
                    "error": f"Corrupted or unreadable image: {e}",
                })

        logger.info(
            f"Image verification: {len(verified_items)} valid, {len(failed_items)} failed out of {total_records}."
        )

        if not verified_items:
            raise RuntimeError(f"No valid, accessible images found from manifest {manifest_p}")

        # Step 2: Batched Inference
        start_time = time.time()
        embedding_list: List[np.ndarray] = []
        mapping_records: List[Dict[str, Any]] = []

        num_batches = (len(verified_items) + bs - 1) // bs
        logger.info(f"Extracting embeddings across {num_batches} batches (batch_size={bs})...")

        for b_idx in range(num_batches):
            batch_chunk = verified_items[b_idx * bs : (b_idx + 1) * bs]
            batch_images: List[Image.Image] = []
            batch_meta: List[Tuple[int, pd.Series, Path]] = []

            for row_idx, row_series, img_path in batch_chunk:
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_meta.append((row_idx, row_series, img_path))
                except Exception as e:
                    logger.error(f"Failed to load image {img_path}: {e}")
                    failed_items.append({
                        "row_index": row_idx,
                        "image_id": row_series.get("image_id", f"ROW_{row_idx}"),
                        "interaction_id": row_series.get("interaction_id", None),
                        "error": str(e),
                    })

            if not batch_images:
                continue

            batch_embeds = self.extract_from_images(batch_images)
            embedding_list.append(batch_embeds)

            for local_i, (orig_idx, row_data, img_path) in enumerate(batch_meta):
                global_idx = len(mapping_records)
                mapping_records.append({
                    "embedding_index": global_idx,
                    "image_id": row_data.get("image_id", f"IMG_{global_idx:04d}"),
                    "interaction_id": row_data.get("interaction_id", None),
                    "sku": row_data.get("sku", None),
                    "category": row_data.get("category", None),
                    "brand": row_data.get("brand", None),
                    "quality_label": row_data.get("quality_label", None),
                    "view": row_data.get("view", None),
                    "timestamp": row_data.get("timestamp", None),
                    "local_path": str(img_path.as_posix()),
                    "original_manifest_row": orig_idx,
                })

        duration = time.time() - start_time
        all_embeddings = np.vstack(embedding_list)
        logger.info(
            f"Extraction complete in {duration:.2f}s ({len(all_embeddings)} images, "
            f"shape={all_embeddings.shape}, dtype={all_embeddings.dtype})"
        )

        # Step 3: Validate Norms and Shapes
        norms = np.linalg.norm(all_embeddings, axis=1)
        norm_diff = np.abs(norms - 1.0)
        assert np.all(norm_diff < 1e-4), f"Embeddings not unit normalized! Max diff: {np.max(norm_diff)}"
        assert not np.isnan(all_embeddings).any(), "NaN values detected in extracted embeddings!"
        assert not np.isinf(all_embeddings).any(), "Inf values detected in extracted embeddings!"

        # Step 4: Persist Artifacts
        npy_path = out_dir / "clip_image_embeddings.npy"
        np.save(npy_path, all_embeddings)
        logger.info(f"Saved image embeddings to {npy_path}")

        df_mapping = pd.DataFrame(mapping_records)
        csv_path = out_dir / "clip_image_mapping.csv"
        df_mapping.to_csv(csv_path, index=False, encoding="utf-8")
        logger.info(f"Saved image mapping to {csv_path}")

        # Step 5: Metadata Manifest
        metadata = {
            "model_name": self.model_name,
            "model_revision": getattr(self.model.config, "_commit_hash", "main"),
            "device": self.device,
            "processor_config": {
                "image_size": 224,
                "crop_size": 224,
                "do_resize": True,
                "do_center_crop": True,
                "do_normalize": True,
            },
            "manifest_path": str(manifest_p.resolve()),
            "manifest_sha256": manifest_hash,
            "total_manifest_records": total_records,
            "successful_images": len(all_embeddings),
            "failed_images": len(failed_items),
            "embedding_shape": list(all_embeddings.shape),
            "embedding_dtype": str(all_embeddings.dtype),
            "batch_size": bs,
            "extraction_duration_seconds": round(duration, 3),
            "throughput_images_per_second": round(len(all_embeddings) / max(0.001, duration), 2),
            "output_embeddings_file": str(npy_path.resolve()),
            "output_mapping_file": str(csv_path.resolve()),
            "failed_samples": failed_items[:10],
            "extracted_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        meta_path = out_dir / "clip_image_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 512-d CLIP image embeddings from manifest.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="reports/clip_dataset_manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Directory to save embeddings and mapping",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default=".",
        help="Root directory for relative image paths",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda or cpu)",
    )
    args = parser.parse_args()

    extractor = ClipFeatureExtractor(device=args.device, batch_size=args.batch_size)
    meta = extractor.extract_from_manifest(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        image_root=args.image_root,
        batch_size=args.batch_size,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
