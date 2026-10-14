"""Canonical Data Schema for SecondLife AI Pricing in Vietnam.

Defines Pydantic v2 models, enumeration classes, and configurable validation policies
for pre-owned household appliance pricing datasets.
"""

from __future__ import annotations

import json
import math
import re
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator


class ApplianceCategory(str, Enum):
    """The 16 canonical household appliance categories."""
    AIR_CONDITIONER = "Air Conditioner"
    BEER_FRIDGE = "Beer Fridge"
    DISHWASHER = "Dishwasher"
    DRYER = "Dryer"
    FOOD_PROCESSOR = "Food Processor"
    FREEZER = "Freezer"
    MICROWAVE = "Microwave"
    OVEN = "Oven"
    REFRIGERATOR = "Refrigerator"
    STAND_MIXER = "Stand Mixer"
    STOVE = "Stove"
    WASHER_DRYER = "Washer Dryer"
    WASHING_MACHINE = "Washing Machine"
    WATER_DISPENSER = "Water Dispenser"
    WATER_PURIFIER = "Water Purifier"
    WINE_COOLER = "Wine Cooler"


class ConditionTier(str, Enum):
    """Standardized 5-tier physical and functional condition scale."""
    LIKE_NEW = "Like New"       # ~99% - Almost unblemished, fully working
    EXCELLENT = "Excellent"     # ~90-95% - Minimal wear, fully working
    GOOD = "Good"               # ~80-89% - Normal wear, fully working
    FAIR = "Fair"               # ~70-79% - Visible scuffs/wear, functioning
    POOR = "Poor"               # <70% - Significant defects or needs repair


class ListingSource(str, Enum):
    """Origin source of the listing record."""
    SECONDLIFE_PARTNER = "secondlife_partner"
    CHOTOT_PARTNER = "chotot_partner"
    FACEBOOK_PARTNER = "facebook_partner"
    USER_SUBMISSION = "user_submission"
    SYNTHETIC_BENCHMARK = "synthetic_benchmark"


class LicenseStatus(str, Enum):
    """Legal licensing status for AI training rights."""
    AUTHORIZED_PARTNER = "authorized_partner"
    PUBLIC_DOMAIN_CC0 = "public_domain_cc0"
    SYNTHETIC_TEST_ONLY = "synthetic_test_only"
    UNLICENSED_BLOCKED = "unlicensed_blocked"


# Validation Default Constants
MIN_PRICE_VND = 50_000.0          # 50,000 VND (~$2.00)
MAX_PRICE_VND = 150_000_000.0     # 150,000,000 VND (~$6,000)
MIN_AGE_YEARS = 0.0
MAX_AGE_YEARS = 30.0


class ValidationPolicy(BaseModel):
    """Configurable policy governing price boundaries, image requirements, and outlier rules."""
    min_price_vnd: float = MIN_PRICE_VND
    max_price_vnd: float = MAX_PRICE_VND
    require_images: bool = False  # Allows records without images for text/tabular experiments
    min_age_years: float = MIN_AGE_YEARS
    max_age_years: float = MAX_AGE_YEARS
    iqr_multiplier: float = 1.5
    allow_unlicensed: bool = False


class PricingListing(BaseModel):
    """Canonical multimodal listing record for used appliance price estimation."""

    _policy: ClassVar[ValidationPolicy] = ValidationPolicy()

    listing_id: str = Field(
        ...,
        description="Unique identifier for the listing (UUID or alphanumeric).",
        min_length=3,
        max_length=64,
    )
    title: str = Field(
        ...,
        description="Vietnamese listing title.",
        min_length=5,
        max_length=255,
    )
    description: Optional[str] = Field(
        default=None,
        description="Detailed Vietnamese description of appliance features and condition.",
        max_length=5000,
    )
    category: ApplianceCategory = Field(
        ...,
        description="Canonical appliance category enum.",
    )
    brand: str = Field(
        ...,
        description="Appliance brand name (normalized uppercase).",
        min_length=2,
        max_length=64,
    )
    model_number: Optional[str] = Field(
        default=None,
        description="Manufacturer model number or SKU identifier.",
        max_length=64,
    )
    condition: ConditionTier = Field(
        ...,
        description="Assessed physical condition tier.",
    )
    age_years: Optional[float] = Field(
        default=None,
        description="Estimated appliance operational age in years.",
    )
    image_paths: List[str] = Field(
        default_factory=list,
        description="List of relative file paths to verified images of this appliance. May be empty for text/tabular experiments.",
    )
    # Price lifecycle fields
    listing_price_vnd: float = Field(
        ...,
        description="Active seller asking price in Vietnamese Dong (VND).",
    )
    initial_listing_price_vnd: Optional[float] = Field(
        default=None,
        description="Original asking price before any AI suggestions were shown to seller.",
    )
    ai_suggested_price_vnd: Optional[float] = Field(
        default=None,
        description="Price suggested by SecondLife AI model (if applicable).",
    )
    user_adjusted_price_vnd: Optional[float] = Field(
        default=None,
        description="Price adjusted by the user after reviewing AI recommendation.",
    )
    transaction_price_vnd: Optional[float] = Field(
        default=None,
        description="Final realized sale/transaction price upon deal completion.",
    )
    product_group_id: Optional[str] = Field(
        default=None,
        description="Product cluster/SKU ID used to prevent duplicate leakage across splits.",
        max_length=128,
    )
    physical_item_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the physical appliance unit (distinguishing individual pre-owned units from manufacturer model numbers).",
        max_length=128,
    )
    listed_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 formatted datetime string.",
    )
    source: ListingSource = Field(
        ...,
        description="Source channel of the dataset record.",
    )
    source_record_id: Optional[str] = Field(
        default=None,
        description="Original identifier from the upstream source for lineage audit.",
        max_length=128,
    )
    license_status: LicenseStatus = Field(
        ...,
        description="Licensing authorization status.",
    )
    permission_evidence: Optional[str] = Field(
        default=None,
        description="Documented reference to user consent record ID, B2B partner contract ID, or public license URI proving legal authorization.",
        max_length=255,
    )

    @classmethod
    def set_validation_policy(cls, policy: ValidationPolicy) -> None:
        """Sets active validation policy for all instances."""
        cls._policy = policy

    @classmethod
    def reset_validation_policy(cls) -> None:
        """Resets validation policy to default."""
        cls._policy = ValidationPolicy()

    @field_validator("brand")
    @classmethod
    def normalize_brand(cls, v: str) -> str:
        cleaned = v.strip().upper()
        if len(cleaned) < 2:
            raise ValueError("Brand name must be at least 2 characters.")
        return cleaned

    @field_validator("listing_price_vnd")
    @classmethod
    def validate_price(cls, v: float) -> float:
        if v < cls._policy.min_price_vnd:
            raise ValueError(
                f"Listing price {v:,.0f} VND is below minimum valid threshold of {cls._policy.min_price_vnd:,.0f} VND."
            )
        if v > cls._policy.max_price_vnd:
            raise ValueError(
                f"Listing price {v:,.0f} VND exceeds maximum threshold of {cls._policy.max_price_vnd:,.0f} VND."
            )
        return float(v)

    @field_validator("age_years")
    @classmethod
    def validate_age(cls, v: Optional[float]) -> Optional[float]:
        if v is not None:
            if v < cls._policy.min_age_years or v > cls._policy.max_age_years:
                raise ValueError(
                    f"Age {v} years is outside valid bounds [{cls._policy.min_age_years}, {cls._policy.max_age_years}]."
                )
        return v

    @field_validator("image_paths", mode="before")
    @classmethod
    def parse_image_paths(cls, v: Any) -> List[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped == "" or v_stripped == "nan":
                raise ValueError("image_paths must be a non-empty list of path strings or JSON array string.")
            if v_stripped == "[]":
                return []
            # Handle JSON array string
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    parsed = json.loads(v_stripped)
                    if isinstance(parsed, list):
                        return [str(p).strip() for p in parsed if str(p).strip()]
                except Exception:
                    pass
            # Handle comma or semicolon separated string
            tokens = [p.strip() for p in re.split(r"[,;]", v_stripped) if p.strip()]
            if tokens:
                return tokens
            raise ValueError("image_paths must be a non-empty list of path strings or JSON array string.")
        elif isinstance(v, (list, tuple)):
            return [str(p).strip() for p in v if str(p).strip()]
        raise ValueError("image_paths must be a list of path strings or JSON array string.")

    @model_validator(mode="after")
    def validate_lifecycle_and_compliance(self) -> PricingListing:
        # Populate initial listing price if omitted
        if self.initial_listing_price_vnd is None:
            self.initial_listing_price_vnd = self.listing_price_vnd

        # Check image requirement policy
        if self._policy.require_images and len(self.image_paths) == 0:
            raise ValueError("image_paths cannot be empty under strict multimodal policy.")

        # Check licensing compliance
        if not self._policy.allow_unlicensed and self.license_status == LicenseStatus.UNLICENSED_BLOCKED:
            raise ValueError(
                f"Listing {self.listing_id} has license_status='unlicensed_blocked' and cannot be used."
            )
        return self

    def to_flat_dict(self) -> Dict[str, Any]:
        """Converts model to a flat dictionary suitable for Pandas DataFrame export."""
        d = self.model_dump()
        d["category"] = self.category.value
        d["condition"] = self.condition.value
        d["source"] = self.source.value
        d["license_status"] = self.license_status.value
        d["image_paths"] = json.dumps(self.image_paths, ensure_ascii=False)
        return d

    @field_validator(
        "age_years",
        "initial_listing_price_vnd",
        "ai_suggested_price_vnd",
        "user_adjusted_price_vnd",
        "transaction_price_vnd",
        mode="before",
    )
    @classmethod
    def clean_optional_floats(cls, v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("nan", "none", ""):
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return float(v)

    @field_validator(
        "description",
        "model_number",
        "product_group_id",
        "physical_item_id",
        "listed_at",
        "source_record_id",
        "permission_evidence",
        mode="before",
    )
    @classmethod
    def clean_optional_strings(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        s = str(v).strip()
        if s.lower() in ("nan", "none", ""):
            return None
        return s

    @classmethod
    def from_series(cls, s: pd.Series) -> PricingListing:
        """Constructs and validates a PricingListing from a Pandas Series."""
        row_dict = {}
        for k, v in s.to_dict().items():
            if pd.isna(v):
                row_dict[k] = None
            else:
                row_dict[k] = v
        return cls(**row_dict)


def create_empty_schema_dataframe() -> pd.DataFrame:
    """Returns an empty DataFrame typed according to the canonical schema."""
    columns = [
        "listing_id",
        "title",
        "description",
        "category",
        "brand",
        "model_number",
        "condition",
        "age_years",
        "image_paths",
        "listing_price_vnd",
        "initial_listing_price_vnd",
        "ai_suggested_price_vnd",
        "user_adjusted_price_vnd",
        "transaction_price_vnd",
        "product_group_id",
        "physical_item_id",
        "listed_at",
        "source",
        "source_record_id",
        "license_status",
        "permission_evidence",
    ]
    return pd.DataFrame(columns=columns)
