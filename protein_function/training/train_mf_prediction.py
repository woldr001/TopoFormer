"""Training pipeline for protein Molecular Function GO-term prediction.

Uses HuggingFace ``Trainer`` with a custom subclass (``MultiLabelTrainer``) that
unpacks the three-feature batch format (topology + ESM + ProtTrans) and computes
BCEWithLogitsLoss.

Quickstart
----------
    python protein_function/training/train_mf_prediction.py \\
        --topo_dir ./topo_features \\
        --esm_dir  ./esm_features \\
        --prottrans_dir ./prottrans_features \\
        --label_file ./mf_annotations.tsv \\
        --train_ids  ./train_ids.txt \\
        --val_ids    ./val_ids.txt \\
        --output_dir ./mf_model \\
        --num_train_epochs 50 \\
        --per_device_train_batch_size 32

The training loop follows the conventions established in
``code_pkg/topt_regression_finetuning.py``.
"""

import logging
import os
import sys
import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from protein_function.data.protein_function_dataset import (
    ProteinFunctionDataset,
    build_mf_label_matrix,
    collate_fn,
    fit_topo_scaler,
    load_topo_scaler,
)
from protein_function.models.mini_topt_config import get_mini_topt_config
from protein_function.models.modeling_topo_function import TopoFunctionModel
from protein_function.evaluation.metrics import compute_metrics_for_trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    """Model architecture and initialisation arguments."""

    pretrained_topt_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a pretrained TopoFormer checkpoint. If provided, "
                           "encoder weights are partially transferred (compatible layers only)."},
    )
    hidden_size: int = field(default=256, metadata={"help": "Mini encoder hidden size."})
    num_hidden_layers: int = field(default=4)
    num_attention_heads: int = field(default=4)
    intermediate_size: int = field(default=1024)
    topo_channels: int = field(
        default=6,
        metadata={"help": "Number of topology feature channels. 6 for protein_only; "
                           "12 for ensemble_motion and sidechain_centroid (mean+std)."},
    )
    topo_n_filtrations: int = field(
        default=200,
        metadata={"help": "Number of filtration steps (topology feature height)."},
    )
    topo_n_combinations: int = field(
        default=15,
        metadata={"help": "Number of element/type combinations (topology feature width). "
                           "15 for protein_only and sidechain_centroid; 121 for ensemble_motion."},
    )
    esm_dim: int = field(default=1152, metadata={"help": "ESM embedding dimension."})
    prottrans_dim: int = field(default=1024, metadata={"help": "ProtTrans embedding dimension."})
    seq_proj_dim: int = field(default=512)
    fusion_hidden_dim: int = field(default=512)
    topo_dropout: float = field(default=0.1)
    fusion_dropout: float = field(default=0.3)
    freeze_topo_encoder: bool = field(
        default=False,
        metadata={"help": "Freeze topology encoder weights during training."},
    )


@dataclass
class DataArguments:
    """Data loading arguments."""

    topo_dir: str = field(metadata={"help": "Directory with <protein_id>.npy topology files."})
    esm_dir: str = field(metadata={"help": "Directory with <protein_id>.npy ESM files."})
    prottrans_dir: str = field(metadata={"help": "Directory with <protein_id>.npy ProtTrans files."})
    label_file: str = field(metadata={"help": "Annotation TSV/CSV with protein MF GO labels."})
    train_ids: str = field(metadata={"help": "Text file with one protein ID per line (training set)."})
    val_ids: str = field(metadata={"help": "Text file with one protein ID per line (validation set)."})
    test_ids: Optional[str] = field(default=None, metadata={"help": "Optional test set IDs."})
    topo_scaler_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to pre-fitted scaler .pkl file. If absent, a new scaler "
                           "is fitted on the training set and saved to <output_dir>/topo_scaler.pkl."},
    )
    min_annotation_count: int = field(
        default=50,
        metadata={"help": "Drop MF GO terms with fewer than this many annotations."},
    )
    label_sep: str = field(default="\t")
    protein_col: str = field(default="protein_id")
    go_col: str = field(default="go_term")
    namespace_col: Optional[str] = field(default="namespace")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    """Extends HuggingFace TrainingArguments with task-specific defaults."""

    output_dir: str = field(default="./mf_model")
    do_train: bool = field(default=True)
    do_eval: bool = field(default=True)
    num_train_epochs: int = field(default=50)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=64)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.05)
    lr_scheduler_type: str = field(default="cosine")
    evaluation_strategy: str = field(default="epoch")
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="fmax")
    greater_is_better: bool = field(default=True)
    remove_unused_columns: bool = field(default=False)
    dataloader_num_workers: int = field(default=4)
    early_stopping_patience: int = field(default=10)
    max_grad_norm: float = field(default=1.0)
    seed: int = field(default=42)


# ---------------------------------------------------------------------------
# Custom Trainer
# ---------------------------------------------------------------------------

class MultiLabelTrainer(Trainer):
    """HuggingFace Trainer subclass for multi-label protein function prediction.

    Unpacks the batch dict (which contains topology, ESM, ProtTrans features
    and labels) and calls ``model.forward()`` with the correct keyword arguments.
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # Remove non-tensor keys
        inputs.pop("protein_ids", None)
        outputs = model(
            topo_features=inputs["topo_features"],
            esm_features=inputs["esm_features"],
            prottrans_features=inputs["prottrans_features"],
            labels=labels,
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ------------------------------------------------------------------
    # Load protein IDs
    # ------------------------------------------------------------------
    def _read_ids(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]

    train_ids = _read_ids(data_args.train_ids)
    val_ids   = _read_ids(data_args.val_ids)
    test_ids  = _read_ids(data_args.test_ids) if data_args.test_ids else []
    logger.info(f"Proteins — train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    # ------------------------------------------------------------------
    # Build label matrix
    # ------------------------------------------------------------------
    label_df, go_terms = build_mf_label_matrix(
        annotation_file=data_args.label_file,
        min_annotation_count=data_args.min_annotation_count,
        sep=data_args.label_sep,
        protein_col=data_args.protein_col,
        go_col=data_args.go_col,
        namespace_col=data_args.namespace_col,
    )
    num_mf_labels = len(go_terms)
    logger.info(f"Predicting {num_mf_labels} MF GO terms.")

    # Save GO term list for later use in inference
    os.makedirs(training_args.output_dir, exist_ok=True)
    go_terms_path = os.path.join(training_args.output_dir, "go_terms.txt")
    with open(go_terms_path, "w") as f:
        f.write("\n".join(go_terms))

    # ------------------------------------------------------------------
    # Topology scaler
    # ------------------------------------------------------------------
    scaler_path = data_args.topo_scaler_path or os.path.join(
        training_args.output_dir, "topo_scaler.pkl"
    )
    if data_args.topo_scaler_path and os.path.exists(data_args.topo_scaler_path):
        topo_scaler = load_topo_scaler(data_args.topo_scaler_path)
        logger.info(f"Loaded topology scaler from {data_args.topo_scaler_path}")
    else:
        logger.info("Fitting topology scaler on training set...")
        topo_scaler = fit_topo_scaler(
            protein_ids=train_ids,
            topo_dir=data_args.topo_dir,
            save_path=scaler_path,
        )

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------
    train_ds = ProteinFunctionDataset(
        protein_ids=train_ids,
        topo_dir=data_args.topo_dir,
        esm_dir=data_args.esm_dir,
        prottrans_dir=data_args.prottrans_dir,
        label_df=label_df,
        go_terms=go_terms,
        topo_scaler=topo_scaler,
    )
    val_ds = ProteinFunctionDataset(
        protein_ids=val_ids,
        topo_dir=data_args.topo_dir,
        esm_dir=data_args.esm_dir,
        prottrans_dir=data_args.prottrans_dir,
        label_df=label_df,
        go_terms=go_terms,
        topo_scaler=topo_scaler,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    topt_config = get_mini_topt_config(
        num_channels=model_args.topo_channels,
        image_size=(model_args.topo_n_filtrations, model_args.topo_n_combinations),
        patch_size=(1, model_args.topo_n_combinations),
        hidden_size=model_args.hidden_size,
        num_hidden_layers=model_args.num_hidden_layers,
        num_attention_heads=model_args.num_attention_heads,
        intermediate_size=model_args.intermediate_size,
        hidden_dropout_prob=model_args.topo_dropout,
        attention_probs_dropout_prob=model_args.topo_dropout,
    )
    model = TopoFunctionModel(
        topt_config=topt_config,
        num_mf_labels=num_mf_labels,
        esm_dim=model_args.esm_dim,
        prottrans_dim=model_args.prottrans_dim,
        seq_proj_dim=model_args.seq_proj_dim,
        fusion_hidden_dim=model_args.fusion_hidden_dim,
        topo_dropout=model_args.topo_dropout,
        fusion_dropout=model_args.fusion_dropout,
    )

    if model_args.pretrained_topt_path:
        from protein_function.models.mini_topt_config import load_pretrained_topt_weights
        skipped = load_pretrained_topt_weights(
            model.topo_encoder, model_args.pretrained_topt_path
        )
        logger.info(f"Partial weight transfer done; skipped {len(skipped)} keys.")

    if model_args.freeze_topo_encoder:
        model.freeze_topo_encoder()
        logger.info("Topology encoder weights frozen.")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    callbacks = [EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=val_ds   if training_args.do_eval  else None,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=callbacks,
    )

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    if training_args.do_train:
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    # ------------------------------------------------------------------
    # Evaluate on test set (if provided)
    # ------------------------------------------------------------------
    if test_ids:
        test_ds = ProteinFunctionDataset(
            protein_ids=test_ids,
            topo_dir=data_args.topo_dir,
            esm_dir=data_args.esm_dir,
            prottrans_dir=data_args.prottrans_dir,
            label_df=label_df,
            go_terms=go_terms,
            topo_scaler=topo_scaler,
        )
        logger.info("Evaluating on test set...")
        test_results = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
        logger.info(f"Test results: {test_results}")
        results_path = os.path.join(training_args.output_dir, "test_results.txt")
        with open(results_path, "w") as f:
            for k, v in test_results.items():
                f.write(f"{k}: {v}\n")


if __name__ == "__main__":
    train()
