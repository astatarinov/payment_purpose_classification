import numpy as np
from loguru import logger
from types import MappingProxyType
import torch
import torch.nn.functional as F

from loans_classification.model_service import ModelService
from loans_classification.config import CLASS_THRESHOLDS, is_leasing


class ClassificationPipeline:
    def __init__(self, model_service: ModelService, class_thresholds: MappingProxyType = CLASS_THRESHOLDS):
        self.model_service = model_service
        self.class_thresholds = class_thresholds
        self.class_names = list(self.class_thresholds.keys())
        self.best_thresholds = np.array(list(self.class_thresholds.values()))

    def classify(self, features: torch.tensor, texts: list[str]) -> list:
        try:
            model = self.model_service.get_multiclass_model()
            results = self._predict_with_thresholds(model, features, texts)
            return results
        except Exception as e:
            logger.error("Error during classification process: {}", str(e))
            raise e

    def _predict_with_thresholds(self, model, data, texts: list[str]):
        model.eval()
        with torch.no_grad():
            logits = model(data)
            probabilities = F.softmax(logits, dim=1).cpu().numpy()

        results = []
        for idx, sample_probas in enumerate(probabilities):
            text = texts[idx]
            if is_leasing(text):
                result = {
                    "label": "leasing",
                    "confident": True,
                    "proba": 1.0
                }
                results.append(result)
                logger.info(f"Sample {idx}: Forced class 'leasing' by rule-based match")
                continue

            try:
                other_idx = self.class_names.index("other")
            except ValueError:
                other_idx = None

            candidates = [
                (self.class_names[i], sample_probas[i])
                for i in range(len(self.class_names))
                if self.class_names[i] != "other" and sample_probas[i] >= self.best_thresholds[i]
            ]
            if candidates:
                best_candidate = max(candidates, key=lambda x: x[1])
                candidate_label, candidate_prob = best_candidate
            else:
                candidate_label, candidate_prob = None, -1.0

            other_prob = sample_probas[other_idx] if other_idx is not None else -1.0

            if candidate_label is not None and candidate_prob > other_prob:
                best_label = candidate_label
                best_proba = candidate_prob
                confident = True
            else:
                best_label = self.class_names[other_idx] if other_idx is not None else None
                best_proba = other_prob
                confident = True

            result = {
                "label": best_label,
                "confident": confident,
                "proba": best_proba
            }
            results.append(result)
            logger.info(
                f"Sample {idx}: Predicted class: {best_label}, proba: {best_proba:.4f}, confidence: {confident}"
            )
        return results
