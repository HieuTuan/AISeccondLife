"""Data Validation Engine for SecondLife AI Pricing in Vietnam.

Performs schema validation, price boundary auditing, per-category IQR outlier detection,
exact and semantic deduplication, image path verification, and dataset provenance reporting.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from pydantic import ValidationError

from src.data.pricing_schema import (
    ApplianceCategory,
    ConditionTier,
    LicenseStatus,
    ListingSource,
    MAX_PRICE_VND,
    MIN_PRICE_VND,
    PricingListing,
    ValidationPolicy,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Summary of data validation and quality audit findings."""
    total_records: int = 0
    valid_records: int = 0
    invalid_schema_records: int = 0
    missing_value_summary: Dict[str, int] = field(default_factory=dict)
    price_boundary_violations: int = 0
    price_outliers_count: int = 0
    category_outlier_summary: Dict[str, int] = field(default_factory=dict)
    exact_duplicate_count: int = 0
    semantic_duplicate_count: int = 0
    missing_image_files_count: int = 0
    unlicensed_blocked_count: int = 0
    is_acquisition_blocked: bool = False
    validation_status: str = "PASS"  # PASS, PARTIAL, BLOCKED, FAIL
    error_samples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        """Renders validation findings as a markdown table report."""
        lines = [
            "# SecondLife Pricing Dataset Quality Audit Report",
            "",
            f"**Validation Status:** `{self.validation_status}`",
            f"**Acquisition Blocked:** `{self.is_acquisition_blocked}`",
            f"**Total Records Evaluated:** {self.total_records:,}",
            f"**Valid Records:** {self.valid_records:,} ({(self.valid_records / max(1, self.total_records)):.1%})",
            "",
            "## Quality Scorecard",
            "",
            "| Check Dimension | Violations Found | Status |",
            "|:---|:---:|:---:|",
            f"| Schema & Format Conformance | {self.invalid_schema_records:,} | {'PASS' if self.invalid_schema_records == 0 else 'WARNING'} |",
            f"| Price Boundary ([50K - 150M VND]) | {self.price_boundary_violations:,} | {'PASS' if self.price_boundary_violations == 0 else 'WARNING'} |",
            f"| Statistical Price Outliers (IQR) | {self.price_outliers_count:,} | {'INFO' if self.price_outliers_count == 0 else 'FLAGGED'} |",
            f"| Exact & Source Duplicates | {self.exact_duplicate_count:,} | {'PASS' if self.exact_duplicate_count == 0 else 'WARNING'} |",
            f"| Semantic & Title Duplicates | {self.semantic_duplicate_count:,} | {'PASS' if self.semantic_duplicate_count == 0 else 'INFO'} |",
            f"| Missing Referenced Image Files | {self.missing_image_files_count:,} | {'PASS' if self.missing_image_files_count == 0 else 'FAIL'} |",
            f"| Unlicensed / Scraping Blocked | {self.unlicensed_blocked_count:,} | {'PASS' if self.unlicensed_blocked_count == 0 else 'BLOCKED'} |",
            "",
            "## Missing Values Breakdown",
            "",
            "| Column | Missing Count | Missing Rate |",
            "|:---|:---:|:---:|",
        ]
        for col, cnt in self.missing_value_summary.items():
            rate = cnt / max(1, self.total_records)
            lines.append(f"| `{col}` | {cnt:,} | {rate:.1%} |")

        if self.category_outlier_summary:
            lines.extend([
                "",
                "## Price Outliers by Category",
                "",
                "| Category | Outlier Count |",
                "|:---|:---:|",
            ])
            for cat, cnt in self.category_outlier_summary.items():
                lines.append(f"| {cat} | {cnt:,} |")

        return "\n".join(lines)


class PricingValidator:
    """Validates and audits used appliance pricing datasets with configurable policies."""

    def __init__(
        self,
        policy: Optional[ValidationPolicy] = None,
        min_price_vnd: Optional[float] = None,
        max_price_vnd: Optional[float] = None,
        iqr_multiplier: Optional[float] = None,
    ) -> None:
        if policy is not None:
            self.policy = policy.model_copy()
        else:
            self.policy = ValidationPolicy()
        if min_price_vnd is not None:
            self.policy.min_price_vnd = min_price_vnd
        if max_price_vnd is not None:
            self.policy.max_price_vnd = max_price_vnd
        if iqr_multiplier is not None:
            self.policy.iqr_multiplier = iqr_multiplier

        self.min_price_vnd = self.policy.min_price_vnd
        self.max_price_vnd = self.policy.max_price_vnd
        self.iqr_multiplier = self.policy.iqr_multiplier

    def validate_dataframe(
        self,
        df: pd.DataFrame,
        image_root: Optional[str | Path] = None,
        verify_images_exist: bool = False,
    ) -> Tuple[ValidationReport, pd.DataFrame]:
        """Runs complete quality audit over a pricing DataFrame.

        Returns:
            Tuple of (ValidationReport, augmented DataFrame with validation flags).
        """
        PricingListing.set_validation_policy(self.policy)
        try:
            return self._execute_validation(df, image_root=image_root, verify_images_exist=verify_images_exist)
        finally:
            PricingListing.reset_validation_policy()

    def _execute_validation(
        self,
        df: pd.DataFrame,
        image_root: Optional[str | Path] = None,
        verify_images_exist: bool = False,
    ) -> Tuple[ValidationReport, pd.DataFrame]:
        report = ValidationReport(total_records=len(df))
        if df.empty:
            report.validation_status = "EMPTY"
            return report, df.copy()

        df_out = df.copy()
        df_out["is_valid"] = True
        df_out["validation_error"] = ""
        df_out["is_price_outlier"] = False
        df_out["outlier_reason"] = ""
        df_out["is_duplicate"] = False

        # 1. Missing Values Analysis
        for col in df.columns:
            missing_cnt = int(df[col].isna().sum() + (df[col] == "").sum())
            report.missing_value_summary[col] = missing_cnt

        # 2. Schema and Pydantic Model Conformance
        valid_indices = []
        for idx, row in df.iterrows():
            try:
                row_dict = row.to_dict()
                # Check for unlicensed status upfront
                if str(row_dict.get("license_status", "")).lower() == "unlicensed_blocked":
                    report.unlicensed_blocked_count += 1
                    df_out.loc[idx, "is_valid"] = False
                    df_out.loc[idx, "validation_error"] = "License status is 'unlicensed_blocked'"
                    continue

                # Run Pydantic validation
                PricingListing.from_series(row)
                valid_indices.append(idx)
            except ValidationError as ve:
                report.invalid_schema_records += 1
                df_out.loc[idx, "is_valid"] = False
                err_msg = "; ".join([e["msg"] for e in ve.errors()])
                df_out.loc[idx, "validation_error"] = err_msg
                if len(report.error_samples) < 10:
                    report.error_samples.append({
                        "row_index": int(idx),
                        "listing_id": row.get("listing_id", "N/A"),
                        "error": err_msg,
                    })
            except Exception as e:
                report.invalid_schema_records += 1
                df_out.loc[idx, "is_valid"] = False
                df_out.loc[idx, "validation_error"] = str(e)

        # 3. Price Boundary Check
        if "listing_price_vnd" in df.columns:
            prices = pd.to_numeric(df["listing_price_vnd"], errors="coerce")
            bound_mask = (prices < self.min_price_vnd) | (prices > self.max_price_vnd) | prices.isna()
            bound_count = int(bound_mask.sum())
            report.price_boundary_violations = bound_count

        # 4. Per-Category Outlier Detection (IQR Method)
        if "category" in df.columns and "listing_price_vnd" in df.columns:
            for cat, group in df.groupby("category"):
                prices = pd.to_numeric(group["listing_price_vnd"], errors="coerce").dropna()
                if len(prices) >= 5:
                    q1 = prices.quantile(0.25)
                    q3 = prices.quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = max(self.min_price_vnd, q1 - self.iqr_multiplier * iqr)
                    upper_bound = q3 + self.iqr_multiplier * iqr

                    outlier_mask = (group["listing_price_vnd"] < lower_bound) | (
                        group["listing_price_vnd"] > upper_bound
                    )
                    outlier_indices = group[outlier_mask].index
                    df_out.loc[outlier_indices, "is_price_outlier"] = True
                    for o_idx in outlier_indices:
                        p_val = df_out.loc[o_idx, "listing_price_vnd"]
                        reason = f"Price {p_val:,.0f} outside [{lower_bound:,.0f}, {upper_bound:,.0f}]"
                        df_out.loc[o_idx, "outlier_reason"] = reason

                    cat_outliers = len(outlier_indices)
                    if cat_outliers > 0:
                        report.category_outlier_summary[str(cat)] = cat_outliers
                        report.price_outliers_count += cat_outliers

        # 5. Exact & Semantic Deduplication
        if "listing_id" in df.columns:
            exact_dups = df.duplicated(subset=["listing_id"], keep="first")
            report.exact_duplicate_count += int(exact_dups.sum())
            df_out.loc[exact_dups, "is_duplicate"] = True

        if "source_record_id" in df.columns:
            src_dups = df["source_record_id"].notna() & df.duplicated(
                subset=["source_record_id"], keep="first"
            )
            report.exact_duplicate_count += int(src_dups.sum())
            df_out.loc[src_dups, "is_duplicate"] = True

        # Semantic duplication: Same normalized title + brand + category + price
        sem_cols = [c for c in ["title", "brand", "category", "listing_price_vnd"] if c in df.columns]
        if len(sem_cols) == 4:
            sem_dups = df.duplicated(subset=sem_cols, keep="first")
            report.semantic_duplicate_count = int(sem_dups.sum())
            df_out.loc[sem_dups, "is_duplicate"] = True

        # 6. Physical Image Relationship Verification (if requested)
        if verify_images_exist and image_root is not None and "image_paths" in df.columns:
            root_path = Path(image_root)
            missing_paths_total = 0

            for idx, row in df.iterrows():
                paths_raw = row["image_paths"]
                try:
                    parsed_paths = PricingListing.parse_image_paths(paths_raw)
                except Exception:
                    parsed_paths = []

                if not parsed_paths:
                    missing_paths_total += 1
                    df_out.loc[idx, "is_valid"] = False
                    df_out.loc[idx, "validation_error"] += "; No valid image paths"
                    continue

                row_has_missing = False
                for p_str in parsed_paths:
                    p = Path(p_str)
                    full_p = p if p.is_absolute() else root_path / p
                    if not full_p.exists():
                        row_has_missing = True
                        break
                    else:
                        # Quick PIL check
                        try:
                            with Image.open(full_p) as img:
                                if img.size[0] <= 0 or img.size[1] <= 0:
                                    row_has_missing = True
                                    break
                        except Exception:
                            row_has_missing = True
                            break

                if row_has_missing:
                    missing_paths_total += 1
                    df_out.loc[idx, "is_valid"] = False
                    df_out.loc[idx, "validation_error"] += "; Image missing or unreadable on disk"

            report.missing_image_files_count = missing_paths_total

        # Compute Final Score
        report.valid_records = int(df_out["is_valid"].sum())

        if report.unlicensed_blocked_count > 0:
            report.is_acquisition_blocked = True
            report.validation_status = "BLOCKED"
        elif report.valid_records == 0:
            report.validation_status = "FAIL"
        elif report.valid_records < len(df) * 0.8:
            report.validation_status = "PARTIAL"
        else:
            report.validation_status = "PASS"

        logger.info(
            f"Validation complete: {report.valid_records}/{report.total_records} valid. Status: {report.validation_status}"
        )
        return report, df_out


def load_and_validate_dataset(
    file_path: str | Path,
    image_root: Optional[str | Path] = None,
    verify_images: bool = False,
) -> Tuple[ValidationReport, pd.DataFrame]:
    """Reads CSV or Parquet file and runs complete data audit."""
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {p}")

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix.lower() in (".csv", ".txt"):
        df = pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file format: {p.suffix}")

    validator = PricingValidator()
    return validator.validate_dataframe(df, image_root=image_root, verify_images_exist=verify_images)
