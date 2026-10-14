"""Data Preprocessing, Text Normalization, Category Mapping, and Leakage-Free Splitting.

Implements:
1. Vietnamese text cleaning and unicode NFC normalization.
2. Multilingual appliance category mapping to canonical 16 classes.
3. String price parsing and conversion into clean VND floats.
4. Group-aware train/validation/test splitting preventing duplicate product leakage.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.pricing_schema import ApplianceCategory, ConditionTier

logger = logging.getLogger(__name__)

# Category Keyword Mapping Dictionary
CATEGORY_MAPPINGS: Dict[str, ApplianceCategory] = {
    # Air Conditioner
    "điều hòa": ApplianceCategory.AIR_CONDITIONER,
    "máy lạnh": ApplianceCategory.AIR_CONDITIONER,
    "dieu hoa": ApplianceCategory.AIR_CONDITIONER,
    "may lanh": ApplianceCategory.AIR_CONDITIONER,
    "air conditioner": ApplianceCategory.AIR_CONDITIONER,
    "ar condicionado": ApplianceCategory.AIR_CONDITIONER,
    "ar-condicionado": ApplianceCategory.AIR_CONDITIONER,
    # Refrigerator
    "tủ lạnh": ApplianceCategory.REFRIGERATOR,
    "tu lanh": ApplianceCategory.REFRIGERATOR,
    "refrigerator": ApplianceCategory.REFRIGERATOR,
    "fridge": ApplianceCategory.REFRIGERATOR,
    "geladeira": ApplianceCategory.REFRIGERATOR,
    "refrigerador": ApplianceCategory.REFRIGERATOR,
    # Washing Machine
    "máy giặt": ApplianceCategory.WASHING_MACHINE,
    "may giat": ApplianceCategory.WASHING_MACHINE,
    "washing machine": ApplianceCategory.WASHING_MACHINE,
    "washer": ApplianceCategory.WASHING_MACHINE,
    "lavadora": ApplianceCategory.WASHING_MACHINE,
    "lavadora de roupas": ApplianceCategory.WASHING_MACHINE,
    # Dryer
    "máy sấy": ApplianceCategory.DRYER,
    "máy sấy quần áo": ApplianceCategory.DRYER,
    "may say": ApplianceCategory.DRYER,
    "clothes dryer": ApplianceCategory.DRYER,
    "dryer": ApplianceCategory.DRYER,
    "secadora": ApplianceCategory.DRYER,
    # Washer Dryer Combo
    "máy giặt sấy": ApplianceCategory.WASHER_DRYER,
    "may giat say": ApplianceCategory.WASHER_DRYER,
    "giặt sấy kết hợp": ApplianceCategory.WASHER_DRYER,
    "washer dryer": ApplianceCategory.WASHER_DRYER,
    "lava e seca": ApplianceCategory.WASHER_DRYER,
    # Microwave
    "lò vi sóng": ApplianceCategory.MICROWAVE,
    "lo vi song": ApplianceCategory.MICROWAVE,
    "microwave": ApplianceCategory.MICROWAVE,
    "micro-ondas": ApplianceCategory.MICROWAVE,
    # Oven
    "lò nướng": ApplianceCategory.OVEN,
    "lo nuong": ApplianceCategory.OVEN,
    "oven": ApplianceCategory.OVEN,
    "forno": ApplianceCategory.OVEN,
    # Stove
    "bếp": ApplianceCategory.STOVE,
    "bếp gas": ApplianceCategory.STOVE,
    "bếp từ": ApplianceCategory.STOVE,
    "bếp hồng ngoại": ApplianceCategory.STOVE,
    "stove": ApplianceCategory.STOVE,
    "cooking range": ApplianceCategory.STOVE,
    "fogão": ApplianceCategory.STOVE,
    "fogao": ApplianceCategory.STOVE,
    # Dishwasher
    "máy rửa bát": ApplianceCategory.DISHWASHER,
    "máy rửa chén": ApplianceCategory.DISHWASHER,
    "may rua bat": ApplianceCategory.DISHWASHER,
    "may rua chen": ApplianceCategory.DISHWASHER,
    "dishwasher": ApplianceCategory.DISHWASHER,
    "lava-louças": ApplianceCategory.DISHWASHER,
    "lava-loucas": ApplianceCategory.DISHWASHER,
    # Water Purifier
    "máy lọc nước": ApplianceCategory.WATER_PURIFIER,
    "may loc nuoc": ApplianceCategory.WATER_PURIFIER,
    "water purifier": ApplianceCategory.WATER_PURIFIER,
    "purificador de água": ApplianceCategory.WATER_PURIFIER,
    "purificador": ApplianceCategory.WATER_PURIFIER,
    # Water Dispenser
    "cây nước nóng lạnh": ApplianceCategory.WATER_DISPENSER,
    "bình nước nóng lạnh": ApplianceCategory.WATER_DISPENSER,
    "máy nước nóng lạnh": ApplianceCategory.WATER_DISPENSER,
    "water dispenser": ApplianceCategory.WATER_DISPENSER,
    "bebedouro": ApplianceCategory.WATER_DISPENSER,
    # Freezer
    "tủ đông": ApplianceCategory.FREEZER,
    "tu dong": ApplianceCategory.FREEZER,
    "freezer": ApplianceCategory.FREEZER,
    # Wine Cooler
    "tủ rượu": ApplianceCategory.WINE_COOLER,
    "tủ bảo quản rượu": ApplianceCategory.WINE_COOLER,
    "tu ruou": ApplianceCategory.WINE_COOLER,
    "wine cooler": ApplianceCategory.WINE_COOLER,
    "adega": ApplianceCategory.WINE_COOLER,
    # Beer Fridge
    "tủ ướp bia": ApplianceCategory.BEER_FRIDGE,
    "tu uop bia": ApplianceCategory.BEER_FRIDGE,
    "beer fridge": ApplianceCategory.BEER_FRIDGE,
    "cervejeira": ApplianceCategory.BEER_FRIDGE,
    # Food Processor
    "máy xay sinh tố": ApplianceCategory.FOOD_PROCESSOR,
    "máy xay đa năng": ApplianceCategory.FOOD_PROCESSOR,
    "máy chế biến thực phẩm": ApplianceCategory.FOOD_PROCESSOR,
    "food processor": ApplianceCategory.FOOD_PROCESSOR,
    "processador": ApplianceCategory.FOOD_PROCESSOR,
    # Stand Mixer
    "máy đánh trứng": ApplianceCategory.STAND_MIXER,
    "máy nhào bột": ApplianceCategory.STAND_MIXER,
    "máy trộn bột": ApplianceCategory.STAND_MIXER,
    "stand mixer": ApplianceCategory.STAND_MIXER,
    "batedeira": ApplianceCategory.STAND_MIXER,
}


class CategoryNormalizer:
    """Normalizes raw multilingual or informal category names to canonical ApplianceCategory."""

    @classmethod
    def normalize(cls, raw_category: Optional[str]) -> Optional[ApplianceCategory]:
        if not raw_category or not str(raw_category).strip():
            return None

        clean_text = str(raw_category).strip().lower()
        clean_text = unicodedata.normalize("NFC", clean_text)

        # 1. Exact match in predefined mapping
        if clean_text in CATEGORY_MAPPINGS:
            return CATEGORY_MAPPINGS[clean_text]

        # 2. Match exact canonical values (case-insensitive)
        for cat in ApplianceCategory:
            if clean_text == cat.value.lower():
                return cat

        # 3. Substring matching in priority order
        for key, val in CATEGORY_MAPPINGS.items():
            if re.search(rf"\b{re.escape(key)}\b", clean_text):
                return val

        return None


class VietnameseTextNormalizer:
    """Cleans and standardizes Vietnamese textual listing titles and descriptions."""

    COMMON_ABBREVIATIONS: Dict[str, str] = {
        r"\bbh\b": "bảo hành",
        r"\binv\b": "inverter",
        r"(?<=\d)tr\b": " triệu",
        r"\btr\b": "triệu",
        r"\bđ\b": "đồng",
        r"\bvnd\b": "đồng",
        r"\bl\b": "lít",
    }

    @classmethod
    def normalize_text(cls, text: Optional[str], expand_abbreviations: bool = False) -> str:
        if text is None or not isinstance(text, str):
            return ""

        # Step 1: Unicode NFC Normalization (crucial for Vietnamese combined diacritics)
        norm_text = unicodedata.normalize("NFC", text.strip())

        # Step 2: Expand common marketplace abbreviations if requested
        if expand_abbreviations:
            for pattern, repl in cls.COMMON_ABBREVIATIONS.items():
                norm_text = re.sub(pattern, repl, norm_text, flags=re.IGNORECASE)

        # Step 3: Remove extraneous control characters, preserve standard punctuation
        norm_text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", norm_text)

        # Step 4: Condense multiple spaces and repeated punctuation marks
        norm_text = re.sub(r"\s+", " ", norm_text)
        norm_text = re.sub(r"([!?.,])\1+", r"\1", norm_text)

        return norm_text.strip()

    @classmethod
    def scrub_pii(cls, text: Optional[str]) -> str:
        """Sanitizes text by removing personally identifiable information (PII).

        Masks:
        - Vietnamese phone numbers (e.g. 0912345678, +84912345678, 03x, 07x, 08x)
        - Email addresses
        - Zalo/Social contact handles
        """
        if text is None or not isinstance(text, str):
            return ""

        s = text
        # Step 1: Mask social contact prefixes (e.g. zalo: 0901234567, fb.com/xyz)
        s = re.sub(r"(?i)\bzalo\s*[:\-=]?\s*(?:\+84|0)?\d+", "zalo: [ẨN]", s)
        s = re.sub(r"(?i)fb(?:\.com)?/\S+", "[FB_ẨN]", s)

        # Step 2: Mask Vietnamese phone numbers
        s = re.sub(r"(?:\+84|0)(?:3[2-9]|5[6|8|9]|7[0|6-9]|8[1-9]|9[0-9])\d{7}\b", "[SĐT_ẨN]", s)
        s = re.sub(r"\b0\d{9,10}\b", "[SĐT_ẨN]", s)

        # Step 3: Mask email addresses
        s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_ẨN]", s)

        return s.strip()


class PriceNormalizer:
    """Parses Vietnamese marketplace price strings into numeric VND values."""

    @classmethod
    def parse_vnd_price(cls, price_input: Union[str, int, float]) -> Optional[float]:
        if price_input is None:
            return None

        if isinstance(price_input, (int, float)):
            if np.isnan(price_input) or np.isinf(price_input) or price_input <= 0:
                return None
            return float(price_input)

        raw = str(price_input).strip().lower()
        if not raw:
            return None

        # 1. Colloquial format with fractional suffix: e.g. "3tr5", "3tr500", "2tr8", "1tr25"
        m_tr_suffix = re.search(r"(\d+)\s*(?:triệu|tr)\s*(\d+)", raw)
        if m_tr_suffix:
            whole = float(m_tr_suffix.group(1))
            rem_str = m_tr_suffix.group(2)
            if len(rem_str) == 1:
                fraction = float(rem_str) / 10.0
            elif len(rem_str) == 2:
                fraction = float(rem_str) / 100.0
            elif len(rem_str) == 3:
                fraction = float(rem_str) / 1000.0
            else:
                fraction = float(rem_str) / (10 ** len(rem_str))
            return (whole + fraction) * 1_000_000

        # 2. Decimal format: e.g. "3.5 triệu", "3,5 tr", "4.2tr"
        m_dec = re.search(r"(\d+)[,.](\d+)\s*(?:triệu|tr)", raw)
        if m_dec:
            return float(f"{m_dec.group(1)}.{m_dec.group(2)}") * 1_000_000

        # 3. Whole triệu format: e.g. "3 triệu", "4tr"
        m_whole = re.search(r"(\d+)\s*(?:triệu|tr)", raw)
        if m_whole:
            return float(m_whole.group(1)) * 1_000_000

        # 4. Handle "k" suffix: e.g. "500k", "1500k"
        m_k = re.search(r"^(\d+)\s*k$", raw)
        if m_k:
            return float(m_k.group(1)) * 1_000

        # 5. Formatted standard numeric strings: e.g. "3.500.000 đ", "3,500,000 VND"
        clean_num = re.sub(r"[^\d]", "", raw)
        if clean_num:
            try:
                val = float(clean_num)
                return val if val > 0 else None
            except ValueError:
                return None

        return None


class DataSplitter:
    """Partitions datasets into train/validation/test sets while preventing duplicate physical item leakage."""

    @staticmethod
    def generate_product_group_key(row: pd.Series) -> str:
        """Constructs an identifier representing the specific physical appliance instance.

        Explicitly distinguishes unique physical item instances from manufacturer model numbers
        to prevent unrelated physical products from being falsely grouped together just because
        they share category, brand, or model.
        """
        # 1. Prefer explicit product_group_id if already present
        explicit_group = str(row.get("product_group_id", "")).strip()
        if explicit_group and explicit_group.lower() not in ("nan", "none", ""):
            return explicit_group

        # 2. Prefer physical_item_id if present (identifies the specific physical appliance unit)
        physical_id = str(row.get("physical_item_id", "")).strip()
        if physical_id and physical_id.lower() not in ("nan", "none", ""):
            return f"item_{physical_id}"

        # 3. Prefer source_record_id if present (identifies upstream physical listing/interaction)
        src_id = str(row.get("source_record_id", "")).strip()
        if src_id and src_id.lower() not in ("nan", "none", ""):
            return f"src_{src_id}"

        # 4. Fallback to unique listing_id so distinct physical products remain independent
        listing_id = str(row.get("listing_id", "")).strip()
        if listing_id and listing_id.lower() not in ("nan", "none", ""):
            return f"listing_{listing_id}"

        # 5. Deterministic hash of row contents for anonymous rows
        row_str = f"{row.get('title', '')}_{row.get('category', '')}_{row.get('listing_price_vnd', '')}"
        return f"group_{hashlib.md5(row_str.encode('utf-8')).hexdigest()[:16]}"

    @classmethod
    def assign_physical_item_groups(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Identifies duplicate listings of the same physical item and assigns a shared group key.

        Duplicate detection rules for identical physical units:
        - Shared explicit product_group_id, physical_item_id, or source_record_id.
        - Shared image file paths or image file content hashes.
        - Exact duplicate semantic content (same seller/source + identical title + brand + category + price).

        Unrelated physical items that simply share category, brand, or manufacturer model number
        are strictly given separate group keys to prevent artificial monolithic clustering.
        """
        df_out = df.copy()
        if "product_group_id" not in df_out.columns:
            df_out["product_group_id"] = None

        assigned_groups: Dict[Any, str] = {}
        for idx, row in df_out.iterrows():
            exp_group = str(row.get("product_group_id", "")).strip()
            if exp_group and exp_group.lower() not in ("nan", "none", ""):
                assigned_groups[idx] = exp_group
                continue
            phys_id = str(row.get("physical_item_id", "")).strip()
            if phys_id and phys_id.lower() not in ("nan", "none", ""):
                assigned_groups[idx] = f"item_{phys_id}"
                continue
            src_id = str(row.get("source_record_id", "")).strip()
            if src_id and src_id.lower() not in ("nan", "none", ""):
                assigned_groups[idx] = f"src_{src_id}"
                continue

        # Link image duplicates: listings with identical non-empty image paths
        image_to_group: Dict[str, str] = {}
        for idx, row in df_out.iterrows():
            if idx in assigned_groups:
                grp = assigned_groups[idx]
            else:
                grp = f"listing_{row.get('listing_id', idx)}"

            raw_imgs = str(row.get("image_paths", "")).strip()
            if raw_imgs and raw_imgs not in ("[]", "nan", "none", ""):
                if raw_imgs in image_to_group:
                    grp = image_to_group[raw_imgs]
                else:
                    image_to_group[raw_imgs] = grp
            assigned_groups[idx] = grp

        for idx, grp in assigned_groups.items():
            df_out.loc[idx, "product_group_id"] = grp

        # Fill any remaining with generate_product_group_key
        unassigned_mask = df_out["product_group_id"].isna() | (df_out["product_group_id"] == "")
        if unassigned_mask.any():
            df_out.loc[unassigned_mask, "product_group_id"] = df_out[unassigned_mask].apply(
                cls.generate_product_group_key, axis=1
            )
        return df_out

    @classmethod
    def split_train_val_test(
        cls,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Performs group-aware disjoint partition across train, validation, and test splits.

        Guarantees that identical product groups or multi-view listings never leak across splits.
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Split ratios must sum to 1.0"
        if df.empty:
            return df.copy(), df.copy(), df.copy()

        df_work = df.copy()
        if "product_group_id" not in df_work.columns or df_work["product_group_id"].isna().any():
            df_work = cls.assign_physical_item_groups(df_work)

        groups = df_work["product_group_id"].values

        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        gss_outer = GroupShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=random_state)
        train_idx, temp_idx = next(gss_outer.split(df_work, groups=groups))

        df_train = df_work.iloc[train_idx].copy()
        df_temp = df_work.iloc[temp_idx].copy()

        # Second split: Val vs Test
        if temp_ratio > 0 and not df_temp.empty:
            relative_test_ratio = test_ratio / temp_ratio
            if test_ratio <= 0.0 or relative_test_ratio <= 0.0:
                df_val = df_temp.copy()
                df_test = pd.DataFrame(columns=df_work.columns)
            elif val_ratio <= 0.0:
                df_val = pd.DataFrame(columns=df_work.columns)
                df_test = df_temp.copy()
            else:
                temp_groups = df_temp["product_group_id"].values
                gss_inner = GroupShuffleSplit(
                    n_splits=1, test_size=relative_test_ratio, random_state=random_state
                )
                val_sub_idx, test_sub_idx = next(gss_inner.split(df_temp, groups=temp_groups))

                df_val = df_temp.iloc[val_sub_idx].copy()
                df_test = df_temp.iloc[test_sub_idx].copy()
        else:
            df_val = pd.DataFrame(columns=df_work.columns)
            df_test = pd.DataFrame(columns=df_work.columns)

        # Verify no leakage between splits
        train_groups = set(df_train["product_group_id"].unique())
        val_groups = set(df_val["product_group_id"].unique())
        test_groups = set(df_test["product_group_id"].unique())

        assert train_groups.isdisjoint(val_groups), "Leakage detected between train and val groups!"
        assert train_groups.isdisjoint(test_groups), "Leakage detected between train and test groups!"
        assert val_groups.isdisjoint(test_groups), "Leakage detected between val and test groups!"

        logger.info(
            f"Group split complete: Train={len(df_train)} ({len(train_groups)} groups), "
            f"Val={len(df_val)} ({len(val_groups)} groups), "
            f"Test={len(df_test)} ({len(test_groups)} groups)"
        )
        return df_train, df_val, df_test
