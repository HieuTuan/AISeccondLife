"""Real Vietnamese Listing Ingestion and Dataset Versioning Pipeline.

Imports raw CSV/Parquet listings from authorized partners or user submissions,
performs schema validation, text & category normalization, IQR outlier detection,
deduplication, image verification, and partitions records into:
1. Valid (production-ready multimodal or tabular records)
2. Incomplete (valid text/price but missing optional fields like images or age)
3. Suspicious (statistical price outliers or semantic duplicates)
4. Unauthorized (unlicensed or blocked data sources)

Persists partition artifacts in a reproducible, versioned directory without overwriting raw inputs.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.data.pricing_schema import (
    ApplianceCategory,
    ConditionTier,
    LicenseStatus,
    ListingSource,
    PricingListing,
    ValidationPolicy,
)
from src.data.pricing_validation import PricingValidator, ValidationReport
from src.data.pricing_preprocessing import (
    CategoryNormalizer,
    PriceNormalizer,
    VietnameseTextNormalizer,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class RealListingsImporter:
    """Imports, standardizes, validates, and versions real Vietnamese appliance listings."""

    def __init__(
        self,
        policy: Optional[ValidationPolicy] = None,
        output_dir: str | Path = "data/processed",
        version: str = "1.0.0",
    ) -> None:
        self.policy = policy if policy is not None else ValidationPolicy()
        self.output_dir = Path(output_dir)
        self.version = version
        self.validator = PricingValidator(policy=self.policy)

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        """Computes SHA-256 hash of a file for provenance verification."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def load_raw_dataset(self, file_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Loads CSV or Parquet raw dataset and computes provenance metadata."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

        file_hash = self.compute_file_hash(p)
        file_size_bytes = p.stat().st_size

        if p.suffix.lower() == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix.lower() in (".csv", ".txt"):
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported file format: {p.suffix}")

        provenance = {
            "source_file": str(p.resolve()),
            "source_filename": p.name,
            "source_sha256": file_hash,
            "source_size_bytes": file_size_bytes,
            "loaded_records": len(df),
            "ingested_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        logger.info(f"Loaded {len(df)} records from {p.name} (SHA-256: {file_hash[:12]}...)")
        return df, provenance

    def normalize_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes titles, categories, prices, and scrubs PII prior to strict validation."""
        df_norm = df.copy()

        # 1. Text normalization & PII scrubbing for title and description
        if "title" in df_norm.columns:
            df_norm["title"] = df_norm["title"].apply(
                lambda t: VietnameseTextNormalizer.scrub_pii(
                    VietnameseTextNormalizer.normalize_text(t, expand_abbreviations=False)
                )
            )
        if "description" in df_norm.columns:
            df_norm["description"] = df_norm["description"].apply(
                lambda d: VietnameseTextNormalizer.scrub_pii(
                    VietnameseTextNormalizer.normalize_text(d, expand_abbreviations=False)
                )
                if pd.notna(d) else None
            )

        # 2. Category normalization
        if "category" in df_norm.columns:
            def _norm_cat(c: Any) -> Optional[str]:
                enum_val = CategoryNormalizer.normalize(str(c))
                return enum_val.value if enum_val else None

            df_norm["category"] = df_norm["category"].apply(_norm_cat)

        # 3. Numeric price parsing
        if "listing_price_vnd" in df_norm.columns:
            df_norm["listing_price_vnd"] = df_norm["listing_price_vnd"].apply(
                PriceNormalizer.parse_vnd_price
            )

        # 4. Brand uppercase normalization
        if "brand" in df_norm.columns:
            df_norm["brand"] = df_norm["brand"].apply(
                lambda b: str(b).strip().upper() if pd.notna(b) and str(b).strip() else ""
            )

        return df_norm

    def partition_records(
        self,
        df_audited: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Separates audited DataFrame into valid, incomplete, suspicious, unauthorized, and synthetic partitions.

        Guarantees that:
        - Unauthorized listings are blocked.
        - Synthetic benchmark examples are strictly isolated from real training partitions.
        - Suspicious outliers and duplicates are quarantined.
        - Only authentic, authorized listings enter the valid/incomplete partitions.
        """
        # Partition 1: Unauthorized (License status is unlicensed_blocked)
        unauth_mask = (
            df_audited["license_status"].astype(str).str.lower() == "unlicensed_blocked"
        ) | (df_audited["validation_error"].str.contains("unlicensed_blocked", case=False, na=False))
        df_unauthorized = df_audited[unauth_mask].copy()

        # Non-unauthorized candidates
        remaining_authorized = df_audited[~unauth_mask].copy()

        # Partition 2: Synthetic Isolation (Synthetic test data quarantined from real training data)
        synth_mask = (
            remaining_authorized["source"].astype(str).str.lower() == "synthetic_benchmark"
        ) | (
            remaining_authorized["license_status"].astype(str).str.lower() == "synthetic_test_only"
        )
        df_synthetic = remaining_authorized[synth_mask].copy()

        # Real training candidates only
        real_candidates = remaining_authorized[~synth_mask].copy()

        # Partition 3: Suspicious (Price outlier or semantic duplicate)
        suspicious_mask = (real_candidates["is_price_outlier"] == True) | (real_candidates["is_duplicate"] == True)
        df_suspicious = real_candidates[suspicious_mask].copy()

        # Non-suspicious real candidates
        clean_candidates = real_candidates[~suspicious_mask].copy()

        # Partition 4: Incomplete (Valid schema & price, but missing optional fields: no images, missing age, missing model)
        def _is_incomplete(row: pd.Series) -> bool:
            if not row.get("is_valid", False):
                return True
            # Check empty image paths
            paths = row.get("image_paths", [])
            try:
                parsed = PricingListing.parse_image_paths(paths)
                has_imgs = len(parsed) > 0
            except Exception:
                has_imgs = False
            has_age = pd.notna(row.get("age_years")) and row.get("age_years") is not None
            has_model = pd.notna(row.get("model_number")) and str(row.get("model_number")).strip() != ""
            # If valid but lacks images or age or model, mark incomplete
            return not (has_imgs and has_age and has_model)

        incomplete_mask = clean_candidates.apply(_is_incomplete, axis=1)
        df_incomplete = clean_candidates[incomplete_mask & clean_candidates["is_valid"]].copy()

        # Partition 5: Valid (Fully populated, compliant multimodal real records)
        df_valid = clean_candidates[~incomplete_mask & clean_candidates["is_valid"]].copy()

        return {
            "valid": df_valid,
            "incomplete": df_incomplete,
            "suspicious": df_suspicious,
            "unauthorized": df_unauthorized,
            "synthetic": df_synthetic,
        }

    def import_and_process(
        self,
        raw_file_path: str | Path,
        image_root: Optional[str | Path] = None,
        verify_images: bool = False,
    ) -> Dict[str, Any]:
        """End-to-end ingestion pipeline: load, normalize, validate, partition, and version."""
        raw_path = Path(raw_file_path)
        df_raw, provenance = self.load_raw_dataset(raw_path)

        # Step 1: Pre-validation Normalization
        df_norm = self.normalize_fields(df_raw)

        # Step 2: Audit and Validation
        report, df_audited = self.validator.validate_dataframe(
            df_norm,
            image_root=image_root,
            verify_images_exist=verify_images,
        )

        # Step 3: Partition Records
        partitions = self.partition_records(df_audited)

        # Step 4: Persist in Versioned Output Directory
        version_dir = self.output_dir / f"v{self.version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        partition_counts = {}
        saved_files = {}

        for part_name, part_df in partitions.items():
            partition_counts[part_name] = len(part_df)
            out_csv = version_dir / f"{part_name}.csv"
            part_df.to_csv(out_csv, index=False, encoding="utf-8")
            saved_entry: Dict[str, Any] = {
                "csv": str(out_csv.resolve()),
                "records": len(part_df),
            }
            try:
                out_parquet = version_dir / f"{part_name}.parquet"
                part_df.to_parquet(out_parquet, index=False)
                saved_entry["parquet"] = str(out_parquet.resolve())
            except Exception as e:
                logger.warning(f"Parquet export skipped for {part_name}: {e}")
            saved_files[part_name] = saved_entry

        # Step 5: Save Ingestion Manifest
        manifest_data = {
            "pipeline_version": self.version,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "provenance": provenance,
            "validation_report": report.to_dict(),
            "partition_counts": partition_counts,
            "saved_artifacts": saved_files,
            "policy": self.policy.model_dump(),
            "is_acquisition_blocked": report.is_acquisition_blocked or len(partitions["unauthorized"]) > 0,
        }

        manifest_file = version_dir / "ingestion_manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Ingestion complete: Valid={partition_counts['valid']}, "
            f"Incomplete={partition_counts['incomplete']}, "
            f"Suspicious={partition_counts['suspicious']}, "
            f"Unauthorized={partition_counts['unauthorized']}. Manifest: {manifest_file.name}"
        )
        return manifest_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Import and validate real Vietnamese appliance listings.")
    parser.add_argument("--input", type=str, required=True, help="Path to raw CSV or Parquet dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--version", type=str, default="1.0.0", help="Dataset version tag")
    parser.add_argument("--image-root", type=str, default=None, help="Root directory for checking images")
    parser.add_argument("--verify-images", action="store_true", help="Verify physical image files on disk")
    args = parser.parse_args()

    importer = RealListingsImporter(output_dir=args.output_dir, version=args.version)
    manifest = importer.import_and_process(
        args.input,
        image_root=args.image_root,
        verify_images=args.verify_images,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
