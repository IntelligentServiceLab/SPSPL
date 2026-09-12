from __future__ import annotations

import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from data_processing import (
    ProcessedData,
    build_normalized_adjacency,
    directed_edge_index,
    macro_average,
    user_ranking_metrics,
)
from pgrl_model import PGRL


PROMPT_INJECTION_VERSION = 2
PROMPT_VALUE_FUSION_VERSION = 3
PROMPT_SOURCE_VERSION = 2
PROMPT_ARCHITECTURE_VERSION = 3


def checkpoint_config(payload: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(payload["config"])

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _positive_sets(edges: np.ndarray, num_mashups: int) -> list[set[int]]:
    positives = [set() for _ in range(num_mashups)]
    for mashup, api in edges:
        positives[int(mashup)].add(int(api))
    return positives


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("model", {})


def load_candidate_block_edges(
    config: dict[str, Any], data: ProcessedData
) -> np.ndarray | None:
    """Load an optional fixed train-positive mask used only during evaluation.

    Formal training and the existing noise-robustness experiment omit this option
    and therefore keep their original behavior.  Edge-deletion experiments use it
    to keep the candidate universe identical across deletion ratios.
    """

    raw_path = config.get("evaluation", {}).get("candidate_block_edges")
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"Candidate-block edge file does not exist: {path}")
    edges = np.load(path)
    if edges.ndim != 2 or edges.shape[1:] != (2,):
        raise ValueError("evaluation.candidate_block_edges must have shape [N, 2]")
    edges = edges.astype(np.int64, copy=False)
    if edges.size:
        if edges[:, 0].min() < 0 or edges[:, 0].max() >= data.num_mashups:
            raise ValueError("Candidate-block file contains an invalid Mashup ID")
        if edges[:, 1].min() < 0 or edges[:, 1].max() >= data.num_apis:
            raise ValueError("Candidate-block file contains an invalid API ID")
    return edges


def validate_recommendation_case_settings(config: dict[str, Any]) -> tuple[bool, int]:
    """Validate and return the recommendation-case output settings."""

    evaluation = config.get("evaluation", {})
    enabled = evaluation.get("save_recommendation_cases", True)
    if not isinstance(enabled, bool):
        raise ValueError("evaluation.save_recommendation_cases must be a boolean")
    top_k = evaluation.get("case_top_k", 5)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("evaluation.case_top_k must be a positive integer")
    return enabled, top_k


def build_model(config: dict[str, Any], data: ProcessedData, device: torch.device) -> nn.Module:
    model_config = _model_config(config)
    ablation = config.get("ablation", {})
    use_prompting = bool(ablation.get("use_prompting", True))
    name = model_config.get("name", "pgrl").lower()
    if name != "pgrl":
        raise ValueError(f"Only the PGRL model is supported, got: {name}")
    dimension = int(model_config.get("embedding_dim", 64))
    structure_layers = int(model_config.get("structure_layers", 2))
    include_layer0 = bool(model_config.get("include_layer0", True))
    text_path = Path(config["data"]["text_embeddings"])
    text_embeddings = torch.from_numpy(np.load(text_path)).to(device)
    return PGRL(
        num_mashups=data.num_mashups,
        num_apis=data.num_apis,
        num_categories=data.num_categories,
        text_embeddings=text_embeddings,
        tag_ids=torch.from_numpy(data.tag_ids).to(device),
        tag_offsets=torch.from_numpy(data.tag_offsets).to(device),
        tag_available=torch.from_numpy(data.tag_available).to(device),
        text_available=torch.from_numpy(data.text_available).to(device),
        dimension=dimension,
        structure_layers=structure_layers,
        semantic_layers=int(model_config.get("semantic_layers", 1)),
        include_layer0=include_layer0,
        tag_pooling=model_config.get("tag_pooling", "attention"),
        dropout=float(config.get("train", {}).get("dropout", 0.1)),
        use_tag_prompt=use_prompting and bool(ablation.get("use_tag_prompt", True)),
        use_text_prompt=use_prompting and bool(ablation.get("use_text_prompt", True)),
        neighbor_aggregation=str(
            ablation.get("neighbor_aggregation", "attention")
        ),
        attention_dim=int(
            ablation.get("attention_dim", model_config.get("attention_dim", 512))
        ),
        gate_mode=str(ablation.get("gate_mode", "learned")),
        fixed_gate=float(ablation.get("fixed_gate", 0.5)),
        fixed_tag_gate=(
            float(ablation["fixed_tag_gate"])
            if ablation.get("fixed_tag_gate") is not None
            else None
        ),
        fixed_text_gate=(
            float(ablation["fixed_text_gate"])
            if ablation.get("fixed_text_gate") is not None
            else None
        ),
    ).to(device)


def _encode_model(
    model: PGRL,
    adjacency: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return model.encode(adjacency, edge_source, edge_target)


def evaluate_model(
    model: PGRL,
    data: ProcessedData,
    adjacency: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    phase: str,
    ks: list[int],
    batch_size: int = 256,
    blocked_train_edges: np.ndarray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if phase not in {"val", "test"}:
        raise ValueError("phase must be val or test")
    target_edges = data.val_edges if phase == "val" else data.test_edges
    targets: dict[int, set[int]] = defaultdict(set)
    for mashup, api in target_edges:
        targets[int(mashup)].add(int(api))
    train_seen = _positive_sets(
        data.train_edges if blocked_train_edges is None else blocked_train_edges,
        data.num_mashups,
    )
    val_seen = _positive_sets(data.val_edges, data.num_mashups)

    model.eval()
    with torch.no_grad():
            mashup_embeddings, api_embeddings = _encode_model(
                model, adjacency, edge_source, edge_target
        )
    rows: list[dict[str, float]] = []
    eval_users = sorted(targets)
    for start in range(0, len(eval_users), batch_size):
        users = eval_users[start : start + batch_size]
        with torch.no_grad():
            scores = (mashup_embeddings[users] @ api_embeddings.T).cpu().numpy()
        for row_index, mashup in enumerate(users):
            blocked = set(train_seen[mashup])
            if phase == "test":
                blocked.update(val_seen[mashup])
            if blocked:
                scores[row_index, list(blocked)] = -np.inf
            ranking = np.argsort(-scores[row_index]).tolist()
            result = user_ranking_metrics(ranking, targets[mashup], ks)
            result["mashup_id"] = float(mashup)
            rows.append(result)
    metric_rows = [{key: value for key, value in row.items() if key != "mashup_id"} for row in rows]
    return macro_average(metric_rows), rows


def save_test_recommendation_cases(
    data: ProcessedData,
    mashup_embeddings: torch.Tensor,
    api_embeddings: torch.Tensor,
    output_dir: str | Path,
    top_k: int = 5,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Save Prompt-enhanced Top-K recommendations under the test protocol.

    Training and validation APIs are removed from each Mashup's candidates,
    exactly as in :func:`evaluate_model` for the test phase.
    """

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    mappings_path = data.root / "mappings.json"
    if not mappings_path.is_file():

        raise FileNotFoundError(f"mapping file does not exist: {mappings_path}")
    with mappings_path.open("r", encoding="utf-8") as handle:
        mappings = json.load(handle)

    mashup_external_ids = mappings["mashup_external_ids"]
    api_external_ids = mappings["api_external_ids"]
    node_names = mappings["node_names"]
    if len(mashup_external_ids) != data.num_mashups:
        raise ValueError("mappings.json has an invalid number of Mashup IDs")
    if len(api_external_ids) != data.num_apis:
        raise ValueError("mappings.json has an invalid number of API IDs")
    if len(node_names) != data.num_nodes:
        raise ValueError("mappings.json has an invalid number of node names")

    targets: dict[int, set[int]] = defaultdict(set)
    for mashup, api in data.test_edges:
        targets[int(mashup)].add(int(api))
    train_seen = _positive_sets(data.train_edges, data.num_mashups)
    val_seen = _positive_sets(data.val_edges, data.num_mashups)

    mashup_embeddings = mashup_embeddings.detach()
    api_embeddings = api_embeddings.detach()
    csv_rows: list[dict[str, Any]] = []
    json_cases: list[dict[str, Any]] = []
    eval_users = sorted(targets)
    for start in range(0, len(eval_users), batch_size):
        users = eval_users[start : start + batch_size]
        scores = (mashup_embeddings[users] @ api_embeddings.T).cpu().numpy()
        for row_index, mashup in enumerate(users):
            blocked = set(train_seen[mashup])
            blocked.update(val_seen[mashup])
            if blocked:
                scores[row_index, list(blocked)] = -np.inf
            ranking = [
                int(api)
                for api in np.argsort(-scores[row_index]).tolist()
                if np.isfinite(scores[row_index, api])
            ][:top_k]

            truth_ids = sorted(targets[mashup])
            ground_truth = [
                {
                    "internal_id": api,
                    "external_id": api_external_ids[api],
                    "name": str(node_names[data.num_mashups + api]),
                }
                for api in truth_ids
            ]
            recommendations: list[dict[str, Any]] = []
            for rank, api in enumerate(ranking, start=1):
                recommendation = {
                    "rank": rank,
                    "internal_id": api,
                    "external_id": api_external_ids[api],
                    "name": str(node_names[data.num_mashups + api]),
                    "score": float(scores[row_index, api]),
                    "hit": api in targets[mashup],
                }
                recommendations.append(recommendation)
                csv_rows.append(
                    {
                        "mashup_internal_id": mashup,
                        "mashup_external_id": mashup_external_ids[mashup],
                        "mashup_name": str(node_names[mashup]),
                        "ground_truth_api_internal_ids": json.dumps(truth_ids),
                        "ground_truth_api_external_ids": json.dumps(
                            [item["external_id"] for item in ground_truth],
                            ensure_ascii=False,
                        ),
                        "ground_truth_api_names": json.dumps(
                            [item["name"] for item in ground_truth],
                            ensure_ascii=False,
                        ),
                        "rank": rank,
                        "recommended_api_internal_id": api,
                        "recommended_api_external_id": api_external_ids[api],
                        "recommended_api_name": recommendation["name"],
                        "score": recommendation["score"],
                        "hit": recommendation["hit"],
                    }
                )
            json_cases.append(
                {
                    "mashup": {
                        "internal_id": mashup,
                        "external_id": mashup_external_ids[mashup],
                        "name": str(node_names[mashup]),
                    },
                    "ground_truth_apis": ground_truth,
                    "recommendations": recommendations,
                }
            )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "test_top5_recommendations.csv"
    json_path = destination / "test_top5_recommendations.json"
    columns = [
        "mashup_internal_id",
        "mashup_external_id",
        "mashup_name",
        "ground_truth_api_internal_ids",
        "ground_truth_api_external_ids",
        "ground_truth_api_names",
        "rank",
        "recommended_api_internal_id",
        "recommended_api_external_id",
        "recommended_api_name",
        "score",
        "hit",
    ]
    pd.DataFrame(csv_rows, columns=columns).to_csv(csv_path, index=False, encoding="utf-8")
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_cases, handle, indent=2, ensure_ascii=False)
    return {
        "enabled": True,
        "top_k": top_k,
        "mashup_count": len(json_cases),
        "recommendation_count": len(csv_rows),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }

def _save_evaluation(
    metrics: dict[str, float], rows: list[dict[str, float]], output_dir: Path, phase: str
) -> None:
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{phase}_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    pd.DataFrame(rows).to_csv(output_dir / f"{phase}_per_mashup.csv", index=False)

def evaluate_checkpoint(
    checkpoint: str | Path,
    output_root: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, float]:
    checkpoint = Path(checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_config(payload)
    project_root = Path(__file__).resolve().parent
    seed = int(config.get("train", {}).get("seed", payload.get("metadata", {}).get("seed", 42)))
    processed_dir = Path(config.get("data", {}).get("processed_dir", ""))
    if not processed_dir.is_dir():
        config["data"]["processed_dir"] = str(
            project_root / "artifacts" / "processed_pw" / f"seed_{seed}"
        )
    text_embeddings = Path(config.get("data", {}).get("text_embeddings", ""))
    if not text_embeddings.is_file():
        filename = text_embeddings.name or "text_embeddings.npy"
        config["data"]["text_embeddings"] = str(
            project_root / "artifacts" / "processed_pw" / filename
        )
    if output_root is not None and output_dir_override is not None:
        raise ValueError("Use either output_root or output_dir_override, not both")
    if output_dir_override is not None:
        config["evaluation"]["output_dir"] = str(output_dir_override)
    elif output_root is not None:
        config["evaluation"]["output_dir"] = str(
            Path(output_root) / "evaluate" / checkpoint.parent.name
        )
    device = resolve_device(config.get("train", {}).get("device", "auto"))
    data = ProcessedData.load(config["data"]["processed_dir"])
    blocked_train_edges = load_candidate_block_edges(config, data)
    model = build_model(config, data, device)
    model.load_state_dict(payload["model_state"])
    adjacency = build_normalized_adjacency(data.num_mashups, data.num_apis, data.train_edges, device)
    source, target = directed_edge_index(data.num_mashups, data.train_edges, device)
    metrics, rows = evaluate_model(
        model,
        data,
        adjacency,
        source,
        target,
        "test",
        [int(value) for value in config.get("evaluation", {}).get("ks", [5, 10, 20])],
        int(config.get("evaluation", {}).get("batch_size", 256)),
        blocked_train_edges,
    )
    output_dir = Path(config["evaluation"]["output_dir"])
    _save_evaluation(metrics, rows, output_dir, "test")
    save_cases, case_top_k = validate_recommendation_case_settings(config)
    recommendation_cases: dict[str, Any] = {"enabled": False}
    if save_cases:
        model.eval()
        with torch.no_grad():
            case_mashups, case_apis = _encode_model(model, adjacency, source, target)
        recommendation_cases = save_test_recommendation_cases(
            data,
            case_mashups,
            case_apis,
            output_dir,
            top_k=case_top_k,
            batch_size=int(config.get("evaluation", {}).get("batch_size", 256)),
        )
        print(
            "[PGRL] Top-"
            f"{case_top_k} recommendation cases saved: "
            f"{recommendation_cases['csv_path']} and {recommendation_cases['json_path']}",
            flush=True,
        )
    summary_path = output_dir / "summary.json"
    summary: dict[str, Any] = {
        "seed": seed,
        "checkpoint": str(checkpoint),
        "test_evaluated": True,
    }
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    summary["recommendation_cases"] = recommendation_cases
    summary["seed"] = seed
    summary["checkpoint"] = str(checkpoint)
    summary["test_evaluated"] = True
    summary["test"] = metrics
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return metrics
