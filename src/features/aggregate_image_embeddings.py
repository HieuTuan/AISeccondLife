"""Multi-View Image Embedding Aggregator for SecondLife AI Pricing.

Aggregates multiple photograph embeddings belonging to the same physical appliance
interaction into a single canonical 512-dimensional multimodal representation:
1. Groups image embeddings by interaction_id.
2. Applies mean pooling across all views of each interaction.
3. Applies L2 normalization: v = v / ||v||_2.
4. Preserves deterministic row ordering and tracks provenance metadata.
5. Saves interaction embeddings as NumPy array and mapping metadata as CSV.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ImageEmbeddingAggregator:
    """Aggregates multiple image embeddings per interaction into a normalized interaction vector."""

    @staticmethod
    def aggregate_embeddings(
        image_embeddings: np.ndarray,
        image_mapping_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Performs mean pooling and L2 normalization per interaction_id.

        Args:
            image_embeddings: Float32 array of shape (N, D).
            image_mapping_df: DataFrame containing at least 'embedding_index' and 'interaction_id'.

        Returns:
            Tuple of:
            - interaction_embeddings: Float32 array of shape (M, D).
            - interaction_mapping_df: DataFrame with M rows mapping to interaction metadata.
        """
        if len(image_embeddings) != len(image_mapping_df):
            raise ValueError(
                f"Length mismatch: {len(image_embeddings)} embeddings vs {len(image_mapping_df)} mapping rows"
            )

        if "interaction_id" not in image_mapping_df.columns:
            raise KeyError("image_mapping_df must contain 'interaction_id' column")

        # Deterministic ordering: Preserve order of first appearance in the mapping
        seen_interactions = []
        interaction_to_indices: Dict[Any, List[int]] = {}

        for idx, row in image_mapping_df.iterrows():
            inter_id = row["interaction_id"]
            emb_idx = int(row.get("embedding_index", idx))
            if inter_id not in interaction_to_indices:
                seen_interactions.append(inter_id)
                interaction_to_indices[inter_id] = []
            interaction_to_indices[inter_id].append(emb_idx)

        num_interactions = len(seen_interactions)
        feature_dim = image_embeddings.shape[1]
        logger.info(
            f"Aggregating {len(image_embeddings)} image embeddings across {num_interactions} unique interactions..."
        )

        agg_embeddings = np.zeros((num_interactions, feature_dim), dtype=np.float32)
        mapping_records = []

        for out_idx, inter_id in enumerate(seen_interactions):
            indices = interaction_to_indices[inter_id]
            view_embeds = image_embeddings[indices]  # Shape: (num_views, 512)

            # Mean pooling across views
            mean_vec = np.mean(view_embeds, axis=0)

            # L2 Normalization
            norm = float(np.linalg.norm(mean_vec))
            if norm > 1e-12:
                norm_vec = mean_vec / norm
            else:
                norm_vec = mean_vec

            agg_embeddings[out_idx] = norm_vec

            # Metadata from representative row (first view)
            first_row = image_mapping_df.iloc[indices[0]]
            image_ids = [str(image_mapping_df.iloc[i].get("image_id", i)) for i in indices]

            mapping_records.append({
                "interaction_embedding_index": out_idx,
                "interaction_id": inter_id,
                "sku": first_row.get("sku", None),
                "category": first_row.get("category", None),
                "brand": first_row.get("brand", None),
                "quality_label": first_row.get("quality_label", None),
                "image_count": len(indices),
                "source_image_indices": json.dumps(indices),
                "source_image_ids": json.dumps(image_ids),
            })

        df_agg_mapping = pd.DataFrame(mapping_records)

        # Integrity Validation
        norms = np.linalg.norm(agg_embeddings, axis=1)
        norm_diff = np.abs(norms - 1.0)
        assert np.all(norm_diff < 1e-4), f"Aggregated vectors not unit normalized! Max diff: {np.max(norm_diff)}"
        assert not np.isnan(agg_embeddings).any(), "NaN values detected in aggregated embeddings!"
        assert not np.isinf(agg_embeddings).any(), "Inf values detected in aggregated embeddings!"

        logger.info(
            f"Aggregation complete: shape={agg_embeddings.shape}, dtype={agg_embeddings.dtype}, "
            f"all unit norms verified."
        )
        return agg_embeddings, df_agg_mapping

    def aggregate_and_save(
        self,
        embeddings_path: str | Path = "data/embeddings/clip_image_embeddings.npy",
        mapping_path: str | Path = "data/embeddings/clip_image_mapping.csv",
        output_dir: str | Path = "data/embeddings",
    ) -> Dict[str, Any]:
        """Loads image embeddings and mapping, performs aggregation, and saves artifacts."""
        emb_p = Path(embeddings_path)
        map_p = Path(mapping_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        if not emb_p.exists():
            raise FileNotFoundError(f"Image embeddings not found: {emb_p}")
        if not map_p.exists():
            raise FileNotFoundError(f"Image mapping not found: {map_p}")

        image_embeddings = np.load(emb_p)
        df_image_mapping = pd.read_csv(map_p)

        agg_embeddings, df_agg_mapping = self.aggregate_embeddings(
            image_embeddings=image_embeddings,
            image_mapping_df=df_image_mapping,
        )

        out_npy = out_dir / "clip_interaction_embeddings.npy"
        np.save(out_npy, agg_embeddings)
        logger.info(f"Saved interaction embeddings to {out_npy}")

        out_csv = out_dir / "clip_interaction_mapping.csv"
        df_agg_mapping.to_csv(out_csv, index=False, encoding="utf-8")
        logger.info(f"Saved interaction mapping to {out_csv}")

        metadata = {
            "source_image_embeddings_file": str(emb_p.resolve()),
            "source_image_mapping_file": str(map_p.resolve()),
            "total_input_images": len(image_embeddings),
            "total_unique_interactions": len(agg_embeddings),
            "interaction_embedding_shape": list(agg_embeddings.shape),
            "multi_view_interactions_count": int((df_agg_mapping["image_count"] > 1).sum()),
            "single_view_interactions_count": int((df_agg_mapping["image_count"] == 1).sum()),
            "max_views_per_interaction": int(df_agg_mapping["image_count"].max()),
            "output_embeddings_file": str(out_npy.resolve()),
            "output_mapping_file": str(out_csv.resolve()),
            "aggregated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        meta_file = out_dir / "clip_interaction_metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate image embeddings into interaction vectors.")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/embeddings/clip_image_embeddings.npy",
        help="Path to image embeddings .npy file",
    )
    parser.add_argument(
        "--mapping",
        type=str,
        default="data/embeddings/clip_image_mapping.csv",
        help="Path to image mapping .csv file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/embeddings",
        help="Directory to save interaction artifacts",
    )
    args = parser.parse_args()

    aggregator = ImageEmbeddingAggregator()
    meta = aggregator.aggregate_and_save(
        embeddings_path=args.embeddings,
        mapping_path=args.mapping,
        output_dir=args.output_dir,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
