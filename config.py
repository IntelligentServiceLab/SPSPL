"""Single formal configuration for the standalone SPSPL experiments."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_ROOT = PROJECT_ROOT / "artifacts" / "processed"
TEXT_EMBEDDINGS = PROJECT_ROOT / "artifacts" / "text_embeddings.npy"
RESULTS_ROOT = PROJECT_ROOT / "result"
CHECKPOINT_ROOT = PROJECT_ROOT / "artifacts" / "checkpoints"

TEXT_MODEL = str(PROJECT_ROOT / "pretrained_models" / "all-MiniLM-L6-v2")
TEXT_BACKEND = "sentence-transformer"
TEXT_BATCH_SIZE = 64
TEXT_DEVICE = "cpu"
FORCE_PREPROCESS = False
FORCE_TEXT_ENCODING = False

# This is the only model configuration in the SPSPL repository. All formal,
# ablation, sensitivity, and robustness experiments start from this baseline.
MODEL_CONFIG = {
    "data": {
        "processed_dir": str(PROCESSED_ROOT / "seed_42"),
        "text_embeddings": str(TEXT_EMBEDDINGS),
    },
    "model": {
        "name": "pgrl",
        "embedding_dim": 1024,
        "structure_layers": 1,
        "semantic_layers": 2,
        "include_layer0": True,
        "tag_pooling": "mean",
    },
    "pretrain": {
        "epochs": 300,
        "mask_ratio": 0.3,
        "learning_rate": 1.6e-2,
        "weight_decay": 1e-4,
        "patience": 20,
    },
    "train": {
        "epochs": 300,
        "learning_rate": 5e-4,
        "backbone_learning_rate": 5e-5,
        "selection_split": "val",
        "selection_metric": "ndcg@5",
        "weight_decay": 5e-4,
        "patience": 20,
        "min_epochs_before_early_stop": 10,
        "negative_samples": 5,
        "mu": 0.2,
        "dropout": 0.3,
        "device": "auto",
        "console_progress": True,
        "progress_interval": 10,
        "seed": 42,
        "checkpoint_dir": str(CHECKPOINT_ROOT / "seed_42"),
    },
    "evaluation": {
        "ks": [5, 10, 20],
        "batch_size": 256,
        "evaluate_test": True,
        "save_recommendation_cases": True,
        "case_top_k": 5,
        "output_dir": str(RESULTS_ROOT / "seed_42"),
    },
    "ablation": {
        "use_pretraining": True,
        "use_prompting": True,
        "use_tag_prompt": True,
        "use_text_prompt": True,
        "use_dual_prediction": True,
        "neighbor_aggregation": "attention",
        "gate_mode": "fixed",
        "fixed_gate": 1.0,
    },
    "run": {"name": "spspl", "suite": "formal"},
}

DATA_SPLIT_CONFIG = {
    "min_eval_mashup_degree": 3,
    "min_target_api_degree": 3,
}
