import torch
from loguru import logger
import torch.nn as nn
import torch.nn.functional as F


class PaymentClassifier(nn.Module):
    def __init__(self, input_dim=1024, hidden_dims=(512, 256),
                 num_classes=9, dropout=0.3):
        super().__init__()
        self.norm_in = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm_in(x)
        x = F.gelu(self.fc1(x))
        x = self.dropout(x)
        x = F.gelu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)


class ModelService:
    def __init__(self, model_paths: dict):
        self.model_paths = model_paths
        self.models = {}

    @staticmethod
    def load_model(path: str):
        model = PaymentClassifier()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Model loaded from path: {path}")
        return model

    def load_all_models(self):
        for model_name, path in self.model_paths.items():
            logger.info(f"Loading model: {model_name}")
            try:
                self.models[model_name] = self.load_model(path)
                logger.info("All models loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load model {model_name} from {path}: {e}")

    def unload_all_models(self):
        logger.info("Unloading all models...")
        self.models.clear()
        logger.info("All models unloaded.")

    def get_multiclass_model(self):
        model = self.models.get('multiclass')
        if model is None:
            logger.error("Multiclass model requested but not loaded.")
            raise RuntimeError("Multiclass model not loaded. Call load_all_models() first.")
        return model
