"""
Predict binding affinity for a single protein-ligand complex using an ensemble
of fine-tuned TopoFormer models.

Usage:
    python predict_single.py

Outputs the predicted binding affinity (pKd/pKi, log Ka units) for 1a4k.
Higher values indicate tighter binding.
"""

import os
import sys
import pickle
import numpy as np
import torch
from transformers import BatchFeature

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code_pkg'))
from top_transformer import TopTForImageClassification

# --- Paths ---
FEATURE_PATH = "./output/1a4k.npy"
SCALER_PATH  = "./code_pkg/pretrain_data_standard_minmax_6channel_large.sav"
MODEL_DIR    = "/mnt/home/woldring/TopoFormer/TopTransformer/Output_dir/shared_finetuned_model_casf_2016_at_11212025"
MODEL_NAMES  = ["model_cls_0", "model_cls_1", "model_cls_2"]

# --- Load and scale features ---
feat = np.load(FEATURE_PATH).astype(np.float32)   # shape [6, 100, 143]
feat = feat[np.newaxis, ...]                        # shape [1, 6, 100, 143]

scaler = pickle.load(open(SCALER_PATH, 'rb'))
n, c, h, w = feat.shape
feat_scaled = scaler.transform(feat.reshape(n, -1)).reshape(n, c, h, w)
model_input = BatchFeature({"topological_features": feat_scaled}, tensor_type='pt')

# --- Run ensemble prediction ---
predictions = []
for name in MODEL_NAMES:
    model_path = os.path.join(MODEL_DIR, name)
    model = TopTForImageClassification.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        out = model(**model_input)
    pred = out.logits.squeeze().item()
    print(f"  {name}: {pred:.3f}")
    predictions.append(pred)

ensemble_mean = np.mean(predictions)
print(f"\nEnsemble mean (pKd/pKi): {ensemble_mean:.3f}")
