"""Zero-Shot CLIP Evaluation Pipeline for TudoBonus Household Appliances.

Evaluates openai/clip-vit-base-patch32 on verified sample images from TudoBonus.
Computes image-level and interaction-level classification metrics:
- Accuracy, Macro F1, Weighted F1
- Per-category Precision, Recall, F1-Score, Support
- Confusion Matrix
- Error analysis (top misclassifications)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import torch
from transformers import CLIPModel, CLIPProcessor

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default 16 TudoBonus Categories and their descriptive prompt labels
CATEGORY_DESCRIPTIONS: Dict[str, List[str]] = {
    "Air Conditioner": ["air conditioner", "split air conditioner unit"],
    "Beer Fridge": ["beer fridge", "beverage cooler refrigerator"],
    "Dishwasher": ["dishwasher", "kitchen dishwasher appliance"],
    "Dryer": ["clothes dryer", "laundry dryer machine"],
    "Food Processor": ["food processor", "electric food processor kitchen appliance"],
    "Freezer": ["deep freezer", "chest freezer refrigerator"],
    "Microwave": ["microwave oven", "countertop microwave appliance"],
    "Oven": ["kitchen oven", "built-in electric oven"],
    "Refrigerator": ["refrigerator", "fridge household appliance"],
    "Stand Mixer": ["stand mixer", "kitchen stand mixer food blender"],
    "Stove": ["kitchen stove", "cooking range gas stove"],
    "Washer Dryer": ["all-in-one washer dryer combo machine"],
    "Washing Machine": ["washing machine", "laundry washing machine"],
    "Water Dispenser": ["water dispenser", "water cooler dispenser with bottle"],
    "Water Purifier": ["water purifier filter", "countertop water purification system"],
    "Wine Cooler": ["wine cooler refrigerator", "wine cellar fridge"],
}

PROMPT_TEMPLATES: List[str] = [
    "a photo of a {label}",
    "a photo of a household {label}",
    "a clean picture of an appliance, {label}",
]


class ClipZeroShotEvaluator:
    """Evaluates pretrained CLIP model on household appliance images."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.batch_size = batch_size

        logger.info(f"Loading CLIP model '{model_name}' on device: {self.device}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("CLIP model loaded successfully.")

    @staticmethod
    def _extract_features(output: Any) -> torch.Tensor:
        """Extracts feature tensor from model output across transformers versions."""
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def build_text_embeddings(
        self, categories: List[str]
    ) -> Tuple[torch.Tensor, List[str]]:
        """Computes normalized text embeddings for candidate categories using prompt ensembling."""
        category_embeddings = []

        with torch.no_grad():
            for cat in categories:
                synonyms = CATEGORY_DESCRIPTIONS.get(cat, [cat.lower()])
                prompts = []
                for syn in synonyms:
                    for tmpl in PROMPT_TEMPLATES:
                        prompts.append(tmpl.format(label=syn))

                inputs = self.processor(
                    text=prompts, return_tensors="pt", padding=True
                ).to(self.device)
                text_out = self.model.get_text_features(**inputs)
                text_features = self._extract_features(text_out)
                # Normalize features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                # Mean pool prompt ensemble
                ensemble_embed = text_features.mean(dim=0)
                ensemble_embed = ensemble_embed / ensemble_embed.norm(dim=-1, keepdim=True)
                category_embeddings.append(ensemble_embed)

        embeddings_tensor = torch.stack(category_embeddings, dim=0)  # [num_classes, embed_dim]
        return embeddings_tensor, categories

    def evaluate_dataset(
        self,
        manifest_path: str | Path,
        image_root: str | Path = "data/clip_sample",
        candidate_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Loads manifest, runs batched zero-shot classification, and computes metrics."""
        manifest_path = Path(manifest_path)
        image_root = Path(image_root)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        df = pd.read_csv(manifest_path)
        logger.info(f"Loaded manifest with {len(df)} records from {manifest_path}")

        # Filter valid records
        valid_records = []
        for idx, row in df.iterrows():
            if "local_path" in row and Path(str(row["local_path"])).exists():
                local_path = Path(str(row["local_path"]))
            elif "repo_path" in row and (image_root / str(row["repo_path"])).exists():
                local_path = image_root / str(row["repo_path"])
            elif "relative_path" in row and (image_root / str(row["relative_path"])).exists():
                local_path = image_root / str(row["relative_path"])
            else:
                logger.warning(f"File missing on disk for row {idx}: {row.get('local_path', row.get('repo_path'))}")
                continue
            valid_records.append((idx, row, local_path))

        if not valid_records:
            raise RuntimeError(f"No valid images found on disk at {image_root}")

        logger.info(f"Found {len(valid_records)} verified accessible images on disk.")

        # Determine target categories
        if candidate_categories is None:
            # Use all 16 canonical categories for realistic open-world zero-shot evaluation
            categories = sorted(list(CATEGORY_DESCRIPTIONS.keys()))
        else:
            categories = sorted(candidate_categories)

        cat2idx = {c: i for i, c in enumerate(categories)}
        text_embeddings, _ = self.build_text_embeddings(categories)  # [K, D]

        # Batched image inference
        all_preds = []
        all_targets = []
        all_confidences = []
        all_probs_list = []
        record_metadata = []

        num_batches = (len(valid_records) + self.batch_size - 1) // self.batch_size
        logger.info(f"Running inference across {num_batches} batches (batch_size={self.batch_size})...")

        for b in range(num_batches):
            batch_items = valid_records[b * self.batch_size : (b + 1) * self.batch_size]
            pil_images = []
            valid_batch_meta = []

            for orig_idx, row, img_path in batch_items:
                try:
                    img = Image.open(img_path).convert("RGB")
                    pil_images.append(img)
                    valid_batch_meta.append((row, img_path))
                except Exception as e:
                    logger.error(f"Error opening image {img_path}: {e}")

            if not pil_images:
                continue

            with torch.no_grad():
                inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
                image_out = self.model.get_image_features(**inputs)
                image_features = self._extract_features(image_out)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)  # [B, D]

                # Similarity logits
                logits = (image_features @ text_embeddings.T) * 100.0  # CLIP logit scale ~100
                probs = torch.softmax(logits, dim=-1).cpu().numpy()  # [B, K]

            for i, (row, img_path) in enumerate(valid_batch_meta):
                prob_vec = probs[i]
                pred_idx = int(np.argmax(prob_vec))
                pred_label = categories[pred_idx]
                pred_conf = float(prob_vec[pred_idx])

                true_cat = str(row["category"])
                true_idx = cat2idx.get(true_cat, -1)

                all_preds.append(pred_label)
                all_targets.append(true_cat)
                all_confidences.append(pred_conf)
                all_probs_list.append(prob_vec)
                record_metadata.append({
                    "image_id": row.get("image_id", ""),
                    "interaction_id": row.get("interaction_id", ""),
                    "sku": row.get("sku", ""),
                    "category": true_cat,
                    "predicted_category": pred_label,
                    "confidence": pred_conf,
                    "is_correct": bool(pred_label == true_cat),
                    "relative_path": str(row.get("relative_path", row.get("repo_path", row.get("local_path", "")))),
                    "quality_label": row.get("quality_label", ""),
                })

        # Calculate Image-Level Metrics
        present_categories = sorted(list(set(all_targets)))
        acc = accuracy_score(all_targets, all_preds)
        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)

        clf_rep = classification_report(
            all_targets,
            all_preds,
            labels=present_categories,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(all_targets, all_preds, labels=present_categories)
        cm_dict = {
            "labels": present_categories,
            "matrix": cm.tolist(),
        }

        # Calculate Interaction-Level Metrics
        # Group by interaction_id, mean-pool probability vectors
        interaction_groups: Dict[str, List[np.ndarray]] = {}
        interaction_trues: Dict[str, str] = {}
        for meta, prob_vec in zip(record_metadata, all_probs_list):
            iid = str(meta["interaction_id"])
            if iid not in interaction_groups:
                interaction_groups[iid] = []
                interaction_trues[iid] = meta["category"]
            interaction_groups[iid].append(prob_vec)

        interaction_preds = []
        interaction_targets = []
        for iid, p_list in interaction_groups.items():
            pooled_prob = np.mean(p_list, axis=0)
            p_cat = categories[int(np.argmax(pooled_prob))]
            interaction_preds.append(p_cat)
            interaction_targets.append(interaction_trues[iid])

        interaction_acc = accuracy_score(interaction_targets, interaction_preds)
        interaction_macro_f1 = f1_score(
            interaction_targets, interaction_preds, average="macro", zero_division=0
        )

        # Identify Top Misclassifications
        incorrect_samples = [m for m in record_metadata if not m["is_correct"]]
        incorrect_samples.sort(key=lambda x: x["confidence"], reverse=True)

        results = {
            "model_name": self.model_name,
            "device": self.device,
            "num_evaluated_images": len(record_metadata),
            "num_evaluated_interactions": len(interaction_groups),
            "num_categories_evaluated": len(present_categories),
            "candidate_categories": categories,
            "present_categories": present_categories,
            "image_level_metrics": {
                "accuracy": float(acc),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "classification_report": clf_rep,
            },
            "interaction_level_metrics": {
                "accuracy": float(interaction_acc),
                "macro_f1": float(interaction_macro_f1),
            },
            "confusion_matrix": cm_dict,
            "top_misclassifications": incorrect_samples[:20],
        }

        # Save predictions dataframe
        pred_df = pd.DataFrame(record_metadata)
        return results, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CLIP Zero-Shot on TudoBonus sample")
    parser.add_argument(
        "--manifest",
        type=str,
        default="reports/clip_dataset_manifest.csv",
        help="Path to sample manifest CSV",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        default="data/clip_sample",
        help="Directory containing downloaded sample images",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        default="reports/clip_zero_shot_metrics.json",
        help="Path to save evaluation metrics JSON",
    )
    parser.add_argument(
        "--output-preds",
        type=str,
        default="reports/clip_predictions.csv",
        help="Path to save prediction results CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size",
    )
    args = parser.parse_args()

    evaluator = ClipZeroShotEvaluator(batch_size=args.batch_size)
    results, pred_df = evaluator.evaluate_dataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
    )

    # Save outputs
    out_metrics_path = Path(args.output_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved evaluation metrics to {out_metrics_path}")

    out_preds_path = Path(args.output_preds)
    out_preds_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_preds_path, index=False, encoding="utf-8")
    logger.info(f"Saved predictions to {out_preds_path}")

    # Log summary
    img_metrics = results["image_level_metrics"]
    inter_metrics = results["interaction_level_metrics"]
    logger.info("=" * 60)
    logger.info("CLIP ZERO-SHOT EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {results['model_name']}")
    logger.info(f"Device: {results['device']}")
    logger.info(f"Evaluated Images: {results['num_evaluated_images']}")
    logger.info(f"Evaluated Interactions: {results['num_evaluated_interactions']}")
    logger.info(f"Image Accuracy: {img_metrics['accuracy']:.4f}")
    logger.info(f"Image Macro F1: {img_metrics['macro_f1']:.4f}")
    logger.info(f"Image Weighted F1: {img_metrics['weighted_f1']:.4f}")
    logger.info(f"Interaction Accuracy: {inter_metrics['accuracy']:.4f}")
    logger.info(f"Interaction Macro F1: {inter_metrics['macro_f1']:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
