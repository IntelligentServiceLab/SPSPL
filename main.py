"""Formal five-seed entry point for SPSPL."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CHECKPOINT_ROOT,
    DATA_DIR,
    DATA_SPLIT_CONFIG,
    FORCE_PREPROCESS,
    FORCE_TEXT_ENCODING,
    MODEL_CONFIG,
    PROCESSED_ROOT,
    RESULTS_ROOT,
    SEEDS,
    TEXT_BACKEND,
    TEXT_BATCH_SIZE,
    TEXT_DEVICE,
    TEXT_EMBEDDINGS,
    TEXT_MODEL,
)
from data_processing import encode_texts, prepare_dataset
from training import train_from_config


REQUIRED_DATA_FILES = (
    "apibasic.csv",
    "mashup.csv",
    "mashupapi.csv",
    "apicate.csv",
    "mashupcate.csv",
    "category.csv",
)

STUDY_NAME = "main"
SPSPL_RESULTS_ROOT = RESULTS_ROOT / STUDY_NAME
SPSPL_CHECKPOINT_ROOT = CHECKPOINT_ROOT.parent / f"checkpoints_{STUDY_NAME}"


def validate_data_directory(data_dir: str | Path = DATA_DIR) -> Path:
    root = Path(data_dir)
    missing = [name for name in REQUIRED_DATA_FILES if not (root / name).is_file()]
    if missing:
        formatted = "\n  - ".join(missing)
        raise FileNotFoundError(
            f"SPSPL requires normalized ProgrammableWeb CSV files in {root}.\n"
            f"Missing files:\n  - {formatted}\n"
            "Copy the contents of the existing pw_normalized directory into data/."
        )
    return root


def prepare_inputs() -> list[dict[str, Any]]:
    data_dir = validate_data_directory()
    reports = []
    for seed in SEEDS:
        destination = PROCESSED_ROOT / f"seed_{seed}"
        metadata_path = destination / "metadata.json"
        graph_path = destination / "graph.npz"
        if FORCE_PREPROCESS or not metadata_path.is_file() or not graph_path.is_file():
            metadata = prepare_dataset(
                data_dir,
                PROCESSED_ROOT,
                seed=seed,
                min_eval_mashup_degree=DATA_SPLIT_CONFIG["min_eval_mashup_degree"],
                min_target_api_degree=DATA_SPLIT_CONFIG["min_target_api_degree"],
            )
        else:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reports.append({"seed": seed, "processed_dir": str(destination), **metadata})

    source = PROCESSED_ROOT / f"seed_{SEEDS[0]}"
    if FORCE_TEXT_ENCODING or not TEXT_EMBEDDINGS.is_file():
        encode_texts(
            source,
            TEXT_EMBEDDINGS,
            model_name=TEXT_MODEL,
            batch_size=TEXT_BATCH_SIZE,
            device=TEXT_DEVICE,
            backend=TEXT_BACKEND,
        )
    return reports


def seed_config(seed: int) -> dict[str, Any]:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed}")
    config = deepcopy(MODEL_CONFIG)
    config["data"]["processed_dir"] = str(PROCESSED_ROOT / f"seed_{seed}")
    config["data"]["text_embeddings"] = str(TEXT_EMBEDDINGS)
    config["train"]["seed"] = seed
    config["train"]["selection_split"] = "val"
    config["train"]["selection_metric"] = "ndcg@5"
    config["train"]["checkpoint_dir"] = str(SPSPL_CHECKPOINT_ROOT / f"seed_{seed}")
    config["evaluation"]["output_dir"] = str(SPSPL_RESULTS_ROOT / f"seed_{seed}")
    config["evaluation"]["evaluate_test"] = True
    config["evaluation"]["save_recommendation_cases"] = True
    config["run"] = {"name": STUDY_NAME, "suite": "formal_spspl"}
    return config


def aggregate_results(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for summary in summaries:
        row = {"seed": int(summary["seed"]), "best_epoch": int(summary["best_epoch"])}
        row.update({f"validation_{key}": value for key, value in summary["validation"].items()})
        row.update({f"test_{key}": value for key, value in summary["test"].items()})
        row.update(
            {
                "parameters": summary["parameters"],
                "train_seconds": summary["train_seconds"],
                "ranking_milliseconds_per_mashup": summary["ranking_milliseconds_per_mashup"],
            }
        )
        rows.append(row)
    raw = pd.DataFrame(rows).sort_values("seed")
    numeric = [column for column in raw if column != "seed"]
    aggregate = {
        column: {
            "mean": float(np.mean(raw[column].to_numpy(dtype=float))),
            "std": float(np.std(raw[column].to_numpy(dtype=float), ddof=1)),
        }
        for column in numeric
    }
    SPSPL_RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(SPSPL_RESULTS_ROOT / "five_seed_results.csv", index=False)
    (SPSPL_RESULTS_ROOT / "five_seed_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"seeds": rows, "aggregate": aggregate}


def run() -> dict[str, Any]:
    inputs = prepare_inputs()
    summaries = []
    for seed in SEEDS:
        print(f"[SPSPL] training formal configuration, seed={seed}", flush=True)
        summaries.append(train_from_config(seed_config(seed)))
    result = {"inputs": inputs, **aggregate_results(summaries)}
    (SPSPL_RESULTS_ROOT / "run_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return result


def dry_run() -> dict[str, Any]:
    """Return the formal run matrix without preparing data or training."""
    return {
        "study": STUDY_NAME,
        "seeds": list(SEEDS),
        "training_runs": len(SEEDS),
        "early_stopping_split": MODEL_CONFIG["train"]["selection_split"],
        "early_stopping_metric": MODEL_CONFIG["train"]["selection_metric"],
        "reported_split": "test",
        "result_root": str(SPSPL_RESULTS_ROOT),
        "checkpoint_root": str(SPSPL_CHECKPOINT_ROOT),
    }


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="print the run plan")
    arguments = parser.parse_args()
    if arguments.dry_run:
        print(json.dumps(dry_run(), indent=2, ensure_ascii=False))
        return
    run()


if __name__ == "__main__":
    cli()
