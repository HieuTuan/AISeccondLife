"""
TudoBonus Image Dataset Preparation for Pretrained CLIP Evaluation.
Part of SecondLife AI Pricing Project (Vietnamese Second-Hand Appliance Pricing).

Author: Senior Machine Learning Engineer & Computer Vision Engineer
Objective: Construct a verified, balanced multimodal sample of 300-500 images
           covering accessible appliance categories for zero-shot CLIP benchmarking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    raise ImportError(
        "huggingface_hub is not installed. Please install huggingface_hub>=0.23.0"
    )

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("PrepareClipDataset")


class TudoBonusClipDatasetBuilder:
    """Selects and downloads a balanced evaluation dataset from TudoBonus for CLIP."""

    def __init__(
        self,
        metadata_path: str = "data/raw/metadata.csv",
        repo_id: str = "TudoBonus/tudobonus",
        output_dir: str = "data/clip_sample",
        manifest_path: str = "reports/clip_dataset_manifest.csv",
        target_per_category: int = 25,
        random_state: int = 42,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.manifest_path = Path(manifest_path)
        self.target_per_category = target_per_category
        self.random_state = random_state

        self.api = HfApi()
        self.repo_files: Set[str] = set()
        self.metadata_df: Optional[pd.DataFrame] = None
        self.sampled_manifest_df: Optional[pd.DataFrame] = None
        self.summary_stats: Dict[str, Any] = {}

    def inspect_and_filter_metadata(self) -> pd.DataFrame:
        """Loads metadata and intersects with accessible image paths in repository."""
        logger.info(f"Loading metadata from {self.metadata_path}...")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.metadata_df = pd.read_csv(self.metadata_path)
        logger.info(f"Loaded {len(self.metadata_df):,} total metadata records.")

        logger.info(f"Querying repository files from Hugging Face: {self.repo_id}...")
        all_files = self.api.list_repo_files(repo_id=self.repo_id, repo_type="dataset")
        self.repo_files = {f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))}
        logger.info(f"Discovered {len(self.repo_files):,} accessible images in Hugging Face repository.")

        accessible_mask = self.metadata_df["path"].isin(self.repo_files)
        df_accessible = self.metadata_df[accessible_mask].copy()
        logger.info(
            f"Cross-referenced accessible metadata rows: {len(df_accessible):,} "
            f"across {df_accessible['interaction_id'].nunique():,} unique interactions."
        )
        return df_accessible

    def sample_balanced_dataset(self, df_accessible: pd.DataFrame) -> pd.DataFrame:
        """
        Samples a balanced subset across all accessible categories.
        Ensures high interaction diversity (prefers 1 image per interaction),
        while taking multi-views for rare categories to maintain balance.
        """
        logger.info(
            f"Sampling balanced dataset (target: ~{self.target_per_category} images per category, "
            f"random_state={self.random_state})..."
        )

        all_categories = sorted(self.metadata_df["product_type"].unique())
        accessible_categories = sorted(df_accessible["product_type"].unique())
        unrepresented_categories = sorted(list(set(all_categories) - set(accessible_categories)))

        logger.info(f"Accessible categories ({len(accessible_categories)}): {accessible_categories}")
        logger.info(f"Unrepresented categories ({len(unrepresented_categories)}): {unrepresented_categories}")

        selected_indices: List[int] = []

        for cat in accessible_categories:
            cat_df = df_accessible[df_accessible["product_type"] == cat]
            unique_ints = cat_df["interaction_id"].unique()

            if cat == "Washer Dryer":
                # Only 10 records across 2 interactions in entire sample release; include all
                selected_indices.extend(cat_df.index.tolist())
            elif cat == "Dryer":
                # 9 interactions; select 2 views per interaction = 18 images
                for i_id in unique_ints:
                    sub = cat_df[cat_df["interaction_id"] == i_id].sort_values("view")
                    selected_indices.extend(sub.index[:2].tolist())
            elif cat == "Beer Fridge":
                # 11 interactions; select 2 views per interaction = 22 images
                for i_id in unique_ints:
                    sub = cat_df[cat_df["interaction_id"] == i_id].sort_values("view")
                    selected_indices.extend(sub.index[:2].tolist())
            elif cat == "Food Processor":
                # 19 interactions; select 1 per interaction (19) + 6 secondary views = 25 images
                for i_id in unique_ints:
                    sub = cat_df[cat_df["interaction_id"] == i_id].sort_values("view")
                    selected_indices.append(sub.index[0])
                remaining = cat_df.loc[~cat_df.index.isin(selected_indices)]
                selected_indices.extend(remaining.index[:6].tolist())
            elif cat == "Wine Cooler":
                # 21 interactions; select 1 per interaction (21) + 4 secondary views = 25 images
                for i_id in unique_ints:
                    sub = cat_df[cat_df["interaction_id"] == i_id].sort_values("view")
                    selected_indices.append(sub.index[0])
                remaining = cat_df.loc[~cat_df.index.isin(selected_indices)]
                selected_indices.extend(remaining.index[:4].tolist())
            else:
                # Categories with >= 25 interactions: sample exactly 25 distinct interactions, 1 view each
                n_sample = min(self.target_per_category, len(unique_ints))
                sampled_ints = (
                    pd.Series(unique_ints)
                    .sample(n_sample, random_state=self.random_state)
                    .tolist()
                )
                for i_id in sampled_ints:
                    sub = cat_df[cat_df["interaction_id"] == i_id].sort_values("view")
                    # Prefer view 0, else first available view
                    v0 = sub[sub["view"] == 0]
                    if len(v0) > 0:
                        selected_indices.append(v0.index[0])
                    else:
                        selected_indices.append(sub.index[0])

        sample_df = self.metadata_df.loc[selected_indices].copy().reset_index(drop=True)
        # Assign clean image_id
        sample_df["image_id"] = [f"IMG_{i+1:04d}" for i in range(len(sample_df))]
        sample_df["category"] = sample_df["product_type"]

        logger.info(
            f"Sampled {len(sample_df)} total records across "
            f"{sample_df['interaction_id'].nunique()} unique interactions and "
            f"{sample_df['category'].nunique()} categories."
        )
        self.sampled_manifest_df = sample_df
        return sample_df

    def _download_and_verify_single(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Downloads a single image and validates properties."""
        repo_path = str(row["path"])
        dest_file = self.output_dir / repo_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Re-use existing sample if present
        existing_sample = Path("data/raw/samples") / repo_path
        if existing_sample.exists() and not dest_file.exists():
            shutil.copy2(existing_sample, dest_file)

        # Download from HF if not on disk
        if not dest_file.exists() or dest_file.stat().st_size == 0:
            try:
                cached_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=repo_path,
                    repo_type="dataset",
                )
                shutil.copy2(cached_path, dest_file)
            except Exception as exc:
                logger.warning(f"Download failed for {repo_path}: {exc}")
                return None

        # Verify image properties via PIL
        try:
            with Image.open(dest_file) as img:
                img_format = img.format
                img_mode = img.mode
                width, height = img.size
                file_size_kb = round(dest_file.stat().st_size / 1024, 2)
                is_valid = bool(width > 0 and height > 0 and img_format in ["JPEG", "PNG", "WEBP"])
        except Exception as err:
            logger.warning(f"Corrupted image file {dest_file}: {err}")
            return None

        return {
            "image_id": row["image_id"],
            "interaction_id": int(row["interaction_id"]),
            "sku": str(row["sku"]),
            "category": str(row["category"]),
            "quality_label": str(row["quality_label"]),
            "brand": str(row["brand"]),
            "view": int(row["view"]),
            "timestamp": int(row["timestamp"]),
            "repo_path": repo_path,
            "local_path": str(dest_file).replace("\\", "/"),
            "width": width,
            "height": height,
            "format": img_format,
            "mode": img_mode,
            "file_size_kb": file_size_kb,
            "verified_valid": is_valid,
        }

    def download_and_verify_images(self, max_workers: int = 10) -> pd.DataFrame:
        """
        Concurrently downloads selected images using ThreadPoolExecutor,
        validates image integrity via PIL, and computes resolution, format, and file size.
        """
        if self.sampled_manifest_df is None:
            raise ValueError("Must run sample_balanced_dataset before downloading.")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_records: List[Dict[str, Any]] = []

        logger.info(
            f"Starting concurrent download and verification of {len(self.sampled_manifest_df)} images "
            f"with {max_workers} worker threads..."
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        rows = [row for _, row in self.sampled_manifest_df.iterrows()]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._download_and_verify_single, r) for r in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading & Verifying"):
                res = future.result()
                if res is not None:
                    manifest_records.append(res)

        verified_df = pd.DataFrame(manifest_records).sort_values("image_id").reset_index(drop=True)
        failed_count = len(self.sampled_manifest_df) - len(verified_df)
        logger.info(
            f"Successfully downloaded and verified {len(verified_df)} images "
            f"({failed_count} failures)."
        )
        return verified_df

    def export_manifest(self, verified_df: pd.DataFrame) -> Path:
        """Saves verified manifest CSV and computes summary statistics."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        verified_df.to_csv(self.manifest_path, index=False, encoding="utf-8")
        logger.info(f"Manifest written to: {self.manifest_path}")

        cat_dist = verified_df["category"].value_counts().to_dict()
        all_cats = sorted(self.metadata_df["product_type"].unique())
        unrep_cats = sorted(list(set(all_cats) - set(cat_dist.keys())))

        self.summary_stats = {
            "total_verified_images": len(verified_df),
            "total_unique_interactions": verified_df["interaction_id"].nunique(),
            "total_unique_skus": verified_df["sku"].nunique(),
            "categories_covered_count": len(cat_dist),
            "total_appliance_categories": len(all_cats),
            "category_distribution": cat_dist,
            "unrepresented_categories": unrep_cats,
            "mean_file_size_kb": round(verified_df["file_size_kb"].mean(), 2),
            "total_dataset_size_mb": round(verified_df["file_size_kb"].sum() / 1024, 2),
            "resolution_summary": {
                "min_width": int(verified_df["width"].min()),
                "max_width": int(verified_df["width"].max()),
                "min_height": int(verified_df["height"].min()),
                "max_height": int(verified_df["height"].max()),
            },
        }

        logger.info(f"Summary: {json.dumps(self.summary_stats, indent=2)}")
        return self.manifest_path

    def run(self) -> Tuple[Path, Dict[str, Any]]:
        """Executes full pipeline: filter -> sample -> download & verify -> manifest export."""
        logger.info("=== Starting CLIP Dataset Preparation Pipeline ===")
        df_accessible = self.inspect_and_filter_metadata()
        sample_df = self.sample_balanced_dataset(df_accessible)
        verified_df = self.download_and_verify_images()
        manifest_file = self.export_manifest(verified_df)
        logger.info("=== CLIP Dataset Preparation Completed Successfully ===")
        return manifest_file, self.summary_stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare balanced TudoBonus image sample for CLIP zero-shot evaluation."
    )
    parser.add_argument("--metadata-path", type=str, default="data/raw/metadata.csv")
    parser.add_argument("--repo-id", type=str, default="TudoBonus/tudobonus")
    parser.add_argument("--output-dir", type=str, default="data/clip_sample")
    parser.add_argument("--manifest-path", type=str, default="reports/clip_dataset_manifest.csv")
    parser.add_argument("--target-per-category", type=int, default=25)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()
    builder = TudoBonusClipDatasetBuilder(
        metadata_path=args.metadata_path,
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        manifest_path=args.manifest_path,
        target_per_category=args.target_per_category,
        random_state=args.random_state,
    )
    builder.run()


if __name__ == "__main__":
    main()
