from __future__ import annotations

import json
import random
import time
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
    lightgcn_propagate,
)
from evaluation import (
    PROMPT_ARCHITECTURE_VERSION,
    PROMPT_INJECTION_VERSION,
    PROMPT_SOURCE_VERSION,
    PROMPT_VALUE_FUSION_VERSION,
    _encode_model,
    _save_evaluation,
    build_model,
    evaluate_model,
    load_candidate_block_edges,
    resolve_device,
    save_test_recommendation_cases,
    validate_recommendation_case_settings,
)
from pgrl_model import PGRL, bpr_loss


def _model_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("model", {})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _report_epoch(epoch: int, total: int, interval: int) -> bool:
    """Keep console progress useful without printing hundreds of lines."""

    current = epoch + 1
    return current == 1 or current == total or current % max(1, interval) == 0


def _positive_sets(edges: np.ndarray, num_mashups: int) -> list[set[int]]:
    positives = [set() for _ in range(num_mashups)]
    for mashup, api in edges:
        positives[int(mashup)].add(int(api))
    return positives


def sample_negatives(
    mashups: np.ndarray,
    num_apis: int,
    known_positives: list[set[int]],
    rng: np.random.Generator,
    negative_samples: int = 1,
) -> np.ndarray:
    if isinstance(negative_samples, bool) or int(negative_samples) <= 0:
        raise ValueError("negative_samples must be a positive integer")
    negative_samples = int(negative_samples)
    negatives = rng.integers(
        0, num_apis, size=(len(mashups), negative_samples), dtype=np.int64
    )
    for index, mashup in enumerate(mashups):
        positives = known_positives[int(mashup)]
        for sample_index in range(negative_samples):
            while int(negatives[index, sample_index]) in positives:
                negatives[index, sample_index] = rng.integers(0, num_apis)
    return negatives

def _masked_edges(edges: np.ndarray, ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    count = max(1, int(len(edges) * ratio))
    permutation = rng.permutation(len(edges))
    return edges[permutation[count:]], edges[permutation[:count]]


def pretrain_structure(
    config: dict[str, Any], data: ProcessedData, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    model_config = _model_config(config)
    section = config.get("pretrain", {})
    dimension = int(model_config.get("embedding_dim", 64))
    structure_layers = int(model_config.get("structure_layers", 2))
    embedding = nn.Embedding(data.num_nodes, dimension).to(device)
    nn.init.normal_(embedding.weight, std=0.1)
    optimizer = torch.optim.Adam(
        embedding.parameters(),
        lr=float(section.get("learning_rate", 1e-3)),
        weight_decay=float(section.get("weight_decay", 1e-4)),
    )
    seed = int(config.get("train", {}).get("seed", 42))
    rng = np.random.default_rng(seed)
    known = _positive_sets(data.all_edges, data.num_mashups)
    best_loss, best_weight, stale = float("inf"), None, 0
    started = time.perf_counter()
    total_epochs = int(section.get("epochs", 200))
    train_section = config.get("train", {})
    negative_samples = int(
        section.get("negative_samples", train_section.get("negative_samples", 1))
    )
    if negative_samples <= 0:
        raise ValueError("pretrain.negative_samples must be a positive integer")
    show_progress = bool(train_section.get("console_progress", True))
    progress_interval = int(train_section.get("progress_interval", 10))
    for epoch in range(total_epochs):
        remaining, masked = _masked_edges(data.train_edges, float(section.get("mask_ratio", 0.2)), rng)
        adjacency = build_normalized_adjacency(data.num_mashups, data.num_apis, remaining, device)
        mashups = masked[:, 0]
        negatives = sample_negatives(
            mashups,
            data.num_apis,
            known,
            rng,
            negative_samples=negative_samples,
        )
        mashup_tensor = torch.from_numpy(mashups).to(device)
        positive_tensor = torch.from_numpy(masked[:, 1]).to(device)
        negative_tensor = torch.from_numpy(negatives).to(device)
        optimizer.zero_grad()
        propagated = lightgcn_propagate(
            embedding.weight, adjacency, structure_layers, include_layer0=True
        )
        mashup_embeddings = propagated[: data.num_mashups]
        api_embeddings = propagated[data.num_mashups :]
        loss = bpr_loss(
            mashup_embeddings, api_embeddings, mashup_tensor, positive_tensor, negative_tensor
        )
        loss.backward()
        optimizer.step()
        current = float(loss.detach().cpu())
        if current < best_loss - 1e-7:
            best_loss = current
            best_weight = embedding.weight.detach().cpu().clone()
            stale = 0
        else:
            stale += 1
        if show_progress and _report_epoch(epoch, total_epochs, progress_interval):
            print(
                f"[PGRL] 预训练 epoch {epoch + 1}/{total_epochs} "
                f"loss={current:.6f} best={best_loss:.6f}",
                flush=True,
            )
        if stale >= int(section.get("patience", 20)):
            if show_progress and not _report_epoch(epoch, total_epochs, progress_interval):
                print(
                    f"[PGRL] 预训练提前停止于 epoch {epoch + 1}/{total_epochs} "
                    f"best={best_loss:.6f}",
                    flush=True,
                )
            break
    assert best_weight is not None
    return best_weight, {
        "best_pretrain_loss": best_loss,
        "pretrain_epochs": epoch + 1,
        "pretrain_seconds": time.perf_counter() - started,
    }


def _optimizer_for(model: PGRL, config: dict[str, Any]) -> torch.optim.Optimizer:
    section = config.get("train", {})
    learning_rate = float(section.get("learning_rate", 1e-3))
    gate_rate = float(section.get("gate_learning_rate", learning_rate))
    backbone_rate = float(section.get("backbone_learning_rate", 1e-4))
    weight_decay = float(section.get("weight_decay", 1e-4))
    if gate_rate <= 0.0:
        raise ValueError("train.gate_learning_rate must be positive")
    backbone = [model.structure_embedding.weight]
    backbone_ids = {id(parameter) for parameter in backbone}
    gate = list(model.prompt_gates.parameters())
    gate_ids = {id(parameter) for parameter in gate}
    other = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and id(parameter) not in gate_ids
    ]
    parameter_groups = [
        {"params": backbone, "lr": backbone_rate},
        {"params": other, "lr": learning_rate},
    ]
    if gate:
        parameter_groups.append({"params": gate, "lr": gate_rate})
    return torch.optim.Adam(parameter_groups, weight_decay=weight_decay)


def _compose_finetuning_loss(
    base_loss: torch.Tensor,
    prompt_loss: torch.Tensor,
    mu: float,
    *,
    use_prompting: bool,
    use_dual_prediction: bool,
) -> torch.Tensor:
    """Apply the formal or ablated prediction objective without changing inference."""

    if not 0.0 <= mu <= 1.0:
        raise ValueError("train.mu must be between 0 and 1")
    if not use_prompting:
        return base_loss
    if not use_dual_prediction:
        return prompt_loss
    return (1 - mu) * base_loss + mu * prompt_loss


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("train", {}).get("seed", 42))
    set_seed(seed)
    device = resolve_device(config.get("train", {}).get("device", "auto"))
    data = ProcessedData.load(config["data"]["processed_dir"])
    blocked_train_edges = load_candidate_block_edges(config, data)
    adjacency = build_normalized_adjacency(
        data.num_mashups, data.num_apis, data.train_edges, device
    )
    edge_source, edge_target = directed_edge_index(data.num_mashups, data.train_edges, device)
    model_name = _model_config(config).get("name", "pgrl").lower()
    output_dir = Path(config.get("evaluation", {}).get("output_dir", "result"))
    checkpoint_dir = Path(config.get("train", {}).get("checkpoint_dir", "artifacts/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ks = [int(value) for value in config.get("evaluation", {}).get("ks", [5, 10, 20])]
    batch_size = int(config.get("evaluation", {}).get("batch_size", 256))
    evaluate_test = bool(config.get("evaluation", {}).get("evaluate_test", True))
    save_cases, case_top_k = validate_recommendation_case_settings(config)
    save_cases = save_cases and evaluate_test
    train_config = config.get("train", {})
    total_epochs = int(train_config.get("epochs", 300))
    negative_samples = int(train_config.get("negative_samples", 1))
    min_epochs_before_early_stop = int(
        train_config.get("min_epochs_before_early_stop", 1)
    )
    if negative_samples <= 0:
        raise ValueError("train.negative_samples must be a positive integer")
    if min_epochs_before_early_stop < 1:
        raise ValueError("train.min_epochs_before_early_stop must be positive")
    if total_epochs < 1:
        raise ValueError("train.epochs must be positive")
    if min_epochs_before_early_stop > total_epochs:
        raise ValueError(
            "train.min_epochs_before_early_stop cannot exceed train.epochs"
        )

    ablation = config.get("ablation", {})
    use_pretraining = bool(ablation.get("use_pretraining", True))
    use_prompting = bool(ablation.get("use_prompting", True))
    use_dual_prediction = bool(ablation.get("use_dual_prediction", True))
    model = build_model(config, data, device)
    if use_pretraining:
        pretrained_weight, pretrain_info = pretrain_structure(config, data, device)
        model.load_structure_pretrain(pretrained_weight.to(device))
        pretrain_info["pretraining_enabled"] = True
    else:
        pretrain_info = {
            "pretraining_enabled": False,
            "best_pretrain_loss": None,
            "pretrain_epochs": 0,
            "pretrain_seconds": 0.0,
        }
    optimizer = _optimizer_for(model, config)
    selection_metric = str(train_config.get("selection_metric", "ndcg@10")).lower()
    allowed_selection_metrics = {
        "recall@5",
        "recall@10",
        "recall@20",
        "ndcg@5",
        "ndcg@10",
        "ndcg@20",
        "mrr",
    }
    if selection_metric not in allowed_selection_metrics:
        raise ValueError(
            "train.selection_metric must be one of recall@5, recall@10, "
            "recall@20, ndcg@5, ndcg@10, ndcg@20, or mrr"
        )
    selection_split = str(train_config.get("selection_split", "val")).lower()
    if selection_split not in {"val", "test"}:
        raise ValueError("train.selection_split must be either val or test")
    rng = np.random.default_rng(seed)
    known = _positive_sets(data.all_edges, data.num_mashups)
    train_mashups = data.train_edges[:, 0]
    train_positive = data.train_edges[:, 1]
    best_score, best_state, best_epoch, stale = -float("inf"), None, 0, 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    show_progress = bool(train_config.get("console_progress", True))
    progress_interval = int(train_config.get("progress_interval", 10))
    for epoch in range(total_epochs):
        model.train()
        negatives = sample_negatives(
            train_mashups,
            data.num_apis,
            known,
            rng,
            negative_samples=negative_samples,
        )
        users = torch.from_numpy(train_mashups).to(device)
        positives = torch.from_numpy(train_positive).to(device)
        negative_tensor = torch.from_numpy(negatives).to(device)
        optimizer.zero_grad()
        base, enhanced, train_diagnostics = model.encode_all(adjacency, edge_source, edge_target)
        base_loss = bpr_loss(
            base[: data.num_mashups],
            base[data.num_mashups :],
            users,
            positives,
            negative_tensor,
        )
        prompt_loss = bpr_loss(
            enhanced[: data.num_mashups],
            enhanced[data.num_mashups :],
            users,
            positives,
            negative_tensor,
        )
        mu = float(train_config.get("mu", 0.5))
        loss = _compose_finetuning_loss(
            base_loss,
            prompt_loss,
            mu,
            use_prompting=use_prompting,
            use_dual_prediction=use_dual_prediction,
        )
        loss.backward()
        optimizer.step()

        selection_metrics, _ = evaluate_model(
            model,
            data,
            adjacency,
            edge_source,
            edge_target,
            selection_split,
            ks,
            batch_size,
            blocked_train_edges,
        )
        if selection_metric not in selection_metrics:
            raise ValueError(
                f"Selection metric {selection_metric} is unavailable; "
                f"evaluation metrics are {sorted(selection_metrics)}"
            )
        selection_score = float(selection_metrics[selection_metric])
        history_row: dict[str, Any] = {
            "epoch": epoch + 1,
            "negative_samples": negative_samples,
            "total_loss": float(loss.detach().cpu()),
            "selection_split": selection_split,
            "selection_metric": selection_metric,
            "selection_score": selection_score,
            f"{selection_split}_recall@5": float(
                selection_metrics.get("recall@5", np.nan)
            ),
            f"{selection_split}_recall@10": float(
                selection_metrics.get("recall@10", np.nan)
            ),
            f"{selection_split}_recall@20": float(
                selection_metrics.get("recall@20", np.nan)
            ),
            f"{selection_split}_ndcg@5": float(selection_metrics.get("ndcg@5", np.nan)),
            f"{selection_split}_ndcg@10": float(selection_metrics.get("ndcg@10", np.nan)),
            f"{selection_split}_ndcg@20": float(selection_metrics.get("ndcg@20", np.nan)),
            f"{selection_split}_mrr": float(selection_metrics.get("mrr", np.nan)),
        }
        history_row.update(
            {
                "base_bpr": float(base_loss.detach().cpu()),
                "prompt_bpr": float(prompt_loss.detach().cpu()),
                "base_norm": float(base.detach().norm(dim=1).mean().cpu()),
                "prompt_norm": float(enhanced.detach().norm(dim=1).mean().cpu()),
                "delta_norm": float(
                    train_diagnostics.get("delta", torch.zeros(1, device=device))
                    .detach()
                    .norm(dim=-1)
                    .mean()
                    .cpu()
                ),
                "residual_norm": float(
                    train_diagnostics.get("residual", torch.zeros(1, device=device))
                    .detach()
                    .norm(dim=-1)
                    .mean()
                    .cpu()
                ),
            }
        )
        history.append(history_row)
        score = selection_score
        if score > best_score + 1e-8:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if show_progress and _report_epoch(epoch, total_epochs, progress_interval):
            print(
                f"[PGRL] 微调 epoch {epoch + 1}/{total_epochs} "
                f"loss={history_row['total_loss']:.6f} "
                f"base_bpr={history_row['base_bpr']:.6f} "
                f"prompt_bpr={history_row['prompt_bpr']:.6f} "
                f"{selection_split}_{selection_metric}={selection_score:.6f} "
                f"best={best_score:.6f}",
                flush=True,
            )
        if (
            epoch + 1 >= min_epochs_before_early_stop
            and stale >= int(train_config.get("patience", 20))
        ):
            if show_progress and not _report_epoch(epoch, total_epochs, progress_interval):
                print(
                    f"[PGRL] 微调提前停止于 epoch {epoch + 1}/{total_epochs} "
                    f"best_{selection_split}_{selection_metric}={best_score:.6f}",
                    flush=True,
                )
            break

    assert best_state is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    model.load_state_dict(best_state)
    evaluation_started = time.perf_counter()
    val_metrics, val_rows = evaluate_model(
        model,
        data,
        adjacency,
        edge_source,
        edge_target,
        "val",
        ks,
        batch_size,
        blocked_train_edges,
    )
    test_metrics: dict[str, float] = {}
    test_rows: list[dict[str, Any]] = []
    if evaluate_test:
        test_metrics, test_rows = evaluate_model(
            model,
            data,
            adjacency,
            edge_source,
            edge_target,
            "test",
            ks,
            batch_size,
            blocked_train_edges,
        )
    evaluation_seconds = time.perf_counter() - evaluation_started
    _save_evaluation(val_metrics, val_rows, output_dir, "val")
    if evaluate_test:
        _save_evaluation(test_metrics, test_rows, output_dir, "test")
    recommendation_cases: dict[str, Any] = {"enabled": False}
    if save_cases:
        model.eval()
        with torch.no_grad():
            case_mashups, case_apis = _encode_model(
                model, adjacency, edge_source, edge_target
            )
        recommendation_cases = save_test_recommendation_cases(

            data,
            case_mashups,
            case_apis,
            output_dir,
            top_k=case_top_k,
            batch_size=batch_size,
        )
        print(
            "[PGRL] Top-"
            f"{case_top_k} recommendation cases saved: "
            f"{recommendation_cases['csv_path']} and {recommendation_cases['json_path']}",
            flush=True,
        )
    diagnostic_summary: dict[str, float] = {}
    if isinstance(model, PGRL):
        from scipy.stats import spearmanr

        model.eval()
        with torch.no_grad():
            diagnostic_base, diagnostic_prompt, diagnostics = model.encode_all(
                adjacency, edge_source, edge_target
            )
        beta_tensor = diagnostics.get("beta")
        beta = (
            beta_tensor.cpu().numpy()
            if beta_tensor is not None
            else np.full((data.num_nodes, 3), np.nan, dtype=np.float32)
        )
        gamma = diagnostics.get("gamma")
        tag_prompt = diagnostics.get("prompt_tag", torch.zeros_like(diagnostic_base))
        text_prompt = diagnostics.get("prompt_text", torch.zeros_like(diagnostic_base))
        gamma_tag = diagnostics.get("gamma_tag")
        gamma_text = diagnostics.get("gamma_text")
        tag_residual = diagnostics.get("residual_tag", torch.zeros_like(diagnostic_base))
        text_residual = diagnostics.get("residual_text", torch.zeros_like(diagnostic_base))
        degrees = np.zeros(data.num_nodes, dtype=np.int64)
        np.add.at(degrees, data.train_edges[:, 0], 1)
        np.add.at(degrees, data.train_edges[:, 1] + data.num_mashups, 1)
        frame = pd.DataFrame(
            {
                "node_id": np.arange(data.num_nodes),
                "node_type": np.where(np.arange(data.num_nodes) < data.num_mashups, "mashup", "api"),
                "train_degree": degrees,
                "beta_structure": beta[:, 0],
                "beta_tag": beta[:, 1],
                "beta_text": beta[:, 2],
                "gamma_mean": gamma.mean(dim=1).cpu().numpy() if gamma is not None else np.nan,
                "gamma_tag_mean": (
                    gamma_tag.mean(dim=1).cpu().numpy() if gamma_tag is not None else np.nan
                ),
                "gamma_text_mean": (
                    gamma_text.mean(dim=1).cpu().numpy() if gamma_text is not None else np.nan
                ),
                "base_norm": diagnostic_base.norm(dim=1).cpu().numpy(),
                "prompt_norm": diagnostic_prompt.norm(dim=1).cpu().numpy(),
                "tag_prompt_norm": tag_prompt.norm(dim=1).cpu().numpy(),
                "text_prompt_norm": text_prompt.norm(dim=1).cpu().numpy(),
                "tag_residual_norm": tag_residual.norm(dim=1).cpu().numpy(),
                "text_residual_norm": text_residual.norm(dim=1).cpu().numpy(),
                "delta_norm": diagnostics.get("delta", torch.zeros_like(diagnostic_base))
                .norm(dim=1)
                .cpu()
                .numpy(),
                "residual_norm": diagnostics.get("residual", torch.zeros_like(diagnostic_base))
                .norm(dim=1)
                .cpu()
                .numpy(),
                "total_residual_norm": diagnostics.get(
                    "residual", torch.zeros_like(diagnostic_base)
                )
                .norm(dim=1)
                .cpu()
                .numpy(),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_dir / "prompt_diagnostics.csv", index=False)
        mashup_mask = frame["node_type"] == "mashup"
        degree_values = frame.loc[mashup_mask, "train_degree"]
        beta_values = frame.loc[mashup_mask, "beta_structure"].dropna()
        if (
            len(beta_values) == len(degree_values)
            and degree_values.nunique() > 1
            and beta_values.nunique() > 1
        ):
            correlation = spearmanr(degree_values, beta_values)
            correlation_value = float(correlation.statistic)
            correlation_p_value = float(correlation.pvalue)
        else:
            correlation_value, correlation_p_value = 0.0, 1.0
        diagnostic_summary = {
            "mean_structure_norm": float(diagnostic_base.norm(dim=1).mean().cpu()),
            "mean_prompt_norm": float(diagnostic_prompt.norm(dim=1).mean().cpu()),
            "mean_gamma_tag": (
                float(gamma_tag.mean().cpu()) if gamma_tag is not None else 0.0
            ),
            "mean_gamma_text": (
                float(gamma_text.mean().cpu()) if gamma_text is not None else 0.0
            ),
            "mean_tag_prompt_norm": float(tag_prompt.norm(dim=1).mean().cpu()),
            "mean_text_prompt_norm": float(text_prompt.norm(dim=1).mean().cpu()),
            "mean_tag_residual_norm": float(tag_residual.norm(dim=1).mean().cpu()),
            "mean_text_residual_norm": float(text_residual.norm(dim=1).mean().cpu()),
            "mean_delta_norm": float(
                diagnostics.get("delta", torch.zeros_like(diagnostic_base)).norm(dim=1).mean().cpu()
            ),
            "mean_residual_norm": float(
                diagnostics.get("residual", torch.zeros_like(diagnostic_base))
                .norm(dim=1)
                .mean()
                .cpu()
            ),
            "mean_total_residual_norm": float(
                diagnostics.get("residual", torch.zeros_like(diagnostic_base))
                .norm(dim=1)
                .mean()
                .cpu()
            ),
        }
        if beta_tensor is not None:
            diagnostic_summary.update(
                {
                    "mean_beta_structure": float(beta[:, 0].mean()),
                    "mean_beta_tag": float(beta[:, 1].mean()),
                    "mean_beta_text": float(beta[:, 2].mean()),
                    "degree_structure_beta_spearman": correlation_value,
                    "degree_structure_beta_p_value": correlation_p_value,
                }
            )
        with (output_dir / "prompt_diagnostics_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(diagnostic_summary, handle, indent=2)
    checkpoint_path = checkpoint_dir / "best.pt"
    torch.save(
        {
            "model_state": best_state,
            "config": config,
            "metadata": data.metadata,
            "model_name": model_name,
            "prompt_injection_version": PROMPT_INJECTION_VERSION,
            "prompt_architecture_version": PROMPT_ARCHITECTURE_VERSION,
            "prompt_source_version": PROMPT_SOURCE_VERSION,
            "prompt_value_fusion_version": PROMPT_VALUE_FUSION_VERSION,
        },
        checkpoint_path,
    )
    summary = {
        "model": model_name,
        "run_name": config.get("run", {}).get("name", model_name),
        "config_sha256": config.get("run", {}).get("config_sha256"),
        "seed": seed,
        "prompt_architecture_version": PROMPT_ARCHITECTURE_VERSION,
        "prompt_source_version": PROMPT_SOURCE_VERSION,
        "prompt_sources": list(model.prompt_sources),
        "selection_split": selection_split,
        "selection_metric": selection_metric,
        "negative_samples": negative_samples,
        "min_epochs_before_early_stop": min_epochs_before_early_stop,
        "optimizer": type(optimizer).__name__,
        "mu": float(train_config.get("mu", 0.5)),
        "gate_learning_rate": float(
            train_config.get(
                "gate_learning_rate", train_config.get("learning_rate", 1e-3)
            )
        ),
        "ablation": {
            "use_pretraining": use_pretraining,
            "use_prompting": use_prompting,
            "use_tag_prompt": use_prompting
            and bool(ablation.get("use_tag_prompt", True)),
            "use_text_prompt": use_prompting
            and bool(ablation.get("use_text_prompt", True)),
            "use_dual_prediction": use_dual_prediction,
            "neighbor_aggregation": str(
                ablation.get("neighbor_aggregation", "attention")
            ),
            "attention_dim": int(
                ablation.get(
                    "attention_dim",
                    config.get("model", {}).get("attention_dim", 512),
                )
            ),
            "gate_mode": str(ablation.get("gate_mode", "learned")),
            "fixed_gate": float(ablation.get("fixed_gate", 0.5)),
            "fixed_tag_gate": float(model.fixed_tag_gate),
            "fixed_text_gate": float(model.fixed_text_gate),
        },
        "weight_decay": float(train_config.get("weight_decay", 1e-4)),
        "best_selection_score": float(best_score),
        "device": str(device),
        "epochs": epoch + 1,
        "best_epoch": best_epoch,
        "last_epoch_selection_score": float(history[-1]["selection_score"]),
        "best_to_last_selection_delta": float(
            best_score - float(history[-1]["selection_score"])
        ),
        "train_seconds": time.perf_counter() - started,
        "test_evaluated": evaluate_test,
        "evaluation_seconds": evaluation_seconds,
        "ranking_milliseconds_per_mashup": (
            1000
            * evaluation_seconds
            / ((2 if evaluate_test else 1) * len(data.eval_mashups))
        ),
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
        "validation": val_metrics,
        "prompt_diagnostics": diagnostic_summary,
        "recommendation_cases": recommendation_cases,
        **pretrain_info,
    }
    if evaluate_test:
        summary["test"] = test_metrics
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary
