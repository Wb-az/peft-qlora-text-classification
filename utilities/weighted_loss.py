import torch
import numpy as np
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
from torch import nn


def create_weights(labels):
    # Compute balanced weights based on the distribution of labels
    weights = compute_class_weight(class_weight="balanced",
                                   classes=np.unique(labels), y=labels)
    return torch.tensor(weights, dtype=torch.float)


def weighted_ce_loss(outputs, labels, num_items_in_batch=None, weights=None):

    logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
    w = weights.to(device=logits.device, dtype=torch.float32) if weights is not None else None
    loss_fct = nn.CrossEntropyLoss(weight=w, reduction="mean")

    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))