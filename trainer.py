import os
from yaml import safe_load
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, fbeta_score, classification_report
from loguru import logger

BASE_PATH = os.getcwd()

with open(f"{BASE_PATH}/model_config.yaml") as f:
    MODEL_CONFIG = safe_load(f)


class PaymentTrainer:
    def __init__(self, model, train_loader, valid_loader):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(MODEL_CONFIG["trainer"]["lr"]),
                                    weight_decay=float(MODEL_CONFIG["trainer"]["weight_decay"]))
        self.device = MODEL_CONFIG["trainer"]["device"]
        self.num_epochs = int(MODEL_CONFIG["trainer"]["num_epochs"])

        self.train_loss_history = []
        self.valid_loss_history = []
        self.train_fbeta_history = []
        self.valid_fbeta_history = []
        self.best_valid_fbeta = 0.0
        self.best_model_state = None

    def train(self):
        with mlflow.start_run():
            for epoch in range(self.num_epochs):
                self.model.train()
                train_loss = 0.0
                train_preds = []
                train_true = []
                for batch_embeddings, batch_labels in self.train_loader:
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_labels = batch_labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(batch_embeddings)
                    loss = self.criterion(outputs, batch_labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item() * batch_embeddings.size(0)
                    preds = outputs.argmax(dim=1)
                    train_preds.extend(preds.cpu().numpy())
                    train_true.extend(batch_labels.cpu().numpy())

                train_loss /= len(self.train_loader.dataset)
                train_fbeta = fbeta_score(train_true, train_preds, beta=0.5, average='macro')

                self.model.eval()
                valid_loss = 0.0
                valid_preds = []
                valid_true = []
                with torch.no_grad():
                    for batch_embeddings, batch_labels in self.valid_loader:
                        batch_embeddings = batch_embeddings.to(self.device)
                        batch_labels = batch_labels.to(self.device)

                        outputs = self.model(batch_embeddings)
                        loss = self.criterion(outputs, batch_labels)
                        valid_loss += loss.item() * batch_embeddings.size(0)

                        preds = outputs.argmax(dim=1)
                        valid_preds.extend(preds.cpu().numpy())
                        valid_true.extend(batch_labels.cpu().numpy())

                valid_loss /= len(self.valid_loader.dataset)
                valid_fbeta = fbeta_score(valid_true, valid_preds, beta=0.5, average='macro')

                if valid_fbeta > self.best_valid_fbeta:
                    self.best_valid_fbeta = valid_fbeta
                    self.best_model_state = self.model.state_dict()
                    mlflow.log_metric("best_valid_fbeta", self.best_valid_fbeta, step=epoch)

                self.train_loss_history.append(train_loss)
                self.valid_loss_history.append(valid_loss)
                self.train_fbeta_history.append(train_fbeta)
                self.valid_fbeta_history.append(valid_fbeta)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("valid_loss", valid_loss, step=epoch)
                mlflow.log_metric("train_fbeta", train_fbeta, step=epoch)
                mlflow.log_metric("valid_fbeta", valid_fbeta, step=epoch)

                print(f"Epoch {epoch + 1}/{self.num_epochs} | "
                      f"Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f} | "
                      f"Train F-beta: {train_fbeta:.4f} | Valid F-beta: {valid_fbeta:.4f}")

            epochs_range = range(1, self.num_epochs + 1)
            plt.figure(figsize=(10, 4))
            plt.plot(epochs_range, self.train_loss_history, label='Train Loss')
            plt.plot(epochs_range, self.valid_loss_history, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training & Validation Loss')
            plt.legend()
            plt.grid(True)
            plt.show()

            plt.figure(figsize=(10, 4))
            plt.plot(epochs_range, self.train_fbeta_history, label='Train F-beta (β=0.5)')
            plt.plot(epochs_range, self.valid_fbeta_history, label='Validation F-beta (β=0.5)')
            plt.xlabel('Epochs')
            plt.ylabel('F-beta Score')
            plt.title('Training & Validation F-beta Score')
            plt.legend()
            plt.grid(True)
            plt.show()

            self.model.eval()
            all_preds = []
            all_true = []
            with torch.no_grad():
                for batch_embeddings, batch_labels in self.valid_loader:
                    batch_embeddings = batch_embeddings.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    outputs = self.model(batch_embeddings)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(batch_labels.cpu().numpy())

            report = classification_report(all_true, all_preds,
                                           target_names=[str(label) for label in np.unique(all_true)])
            print("Classification Report on validation set:\n", report)
            report_file = "classification_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_file)

            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                mlflow.pytorch.log_model(self.model, "best_pytorch_model")

    @staticmethod
    def _get_best_threshold(probs, true_labels, min_precision=0.99, num_thresholds=1000):
        best_threshold = None
        best_recall = 0.0
        thresholds_candidates = np.linspace(0, 1, num_thresholds)

        for threshold in thresholds_candidates:
            preds = (probs >= threshold).astype(int)
            if preds.sum() == 0:
                precision = 1.0
                recall = 0.0
            else:
                precision = precision_score(true_labels, preds, zero_division=1)
                recall = recall_score(true_labels, preds)
            if precision >= min_precision and recall > best_recall:
                best_recall = recall
                best_threshold = threshold
        return best_threshold, best_recall

    def find_thresholds(self, val_dataloader, model, class_to_idx, min_precision=0.99, num_thresholds=1000):
        model.eval()
        all_probs = []
        all_labels = []
        device = next(model.parameters()).device
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        thresholds = {}
        for cls, idx in class_to_idx.items():
            if cls == "other":
                continue
            probs_cls = all_probs[:, idx].numpy()
            true_cls = (all_labels.numpy() == idx).astype(int)
            best_threshold, best_recall = self._get_best_threshold(probs_cls, true_cls, min_precision, num_thresholds)
            thresholds[cls] = {"threshold": best_threshold, "recall": best_recall}
            if best_threshold is not None:
                logger.info(f"Label {cls}: best threshold = {best_threshold:.4f}, recall = {best_recall:.4f}")
            else:
                print(f"Label {cls}: no best threshold with precision >= {min_precision}")

        return thresholds

    @staticmethod
    def compute_final_metrics(val_dataloader, model, thresholds, class_to_idx):
        model.eval()
        all_preds = []
        all_trues = []
        device = next(model.parameters()).device
        idx_to_class = {v: k for k, v in class_to_idx.items()}

        with torch.no_grad():
            for inputs, labels in val_dataloader:
                inputs = inputs.to(device)
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
                probs = probs.cpu().numpy()
                labels = labels.cpu().numpy()

                for prob, true_label in zip(probs, labels):
                    candidate = None
                    candidate_prob = -1.0
                    for cls, idx in class_to_idx.items():
                        if cls == "other":
                            continue
                        threshold_val = thresholds.get(cls, {}).get("threshold", None)
                        if threshold_val is None:
                            continue
                        if prob[idx] >= threshold_val and prob[idx] > candidate_prob:
                            candidate = idx
                            candidate_prob = prob[idx]
                    other_idx = class_to_idx.get("other", None)
                    other_prob = prob[other_idx] if other_idx is not None else -1.0
                    if candidate is not None and candidate_prob > other_prob:
                        pred = candidate
                    else:
                        pred = other_idx if other_idx is not None else -1
                    all_preds.append(pred)
                    all_trues.append(true_label)

        target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        report = classification_report(all_trues, all_preds, target_names=target_names, zero_division=0)
        print(report)
        return all_preds, all_trues
