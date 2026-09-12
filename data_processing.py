from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import torch


@dataclass
class ProcessedData:
    root: Path
    train_edges: np.ndarray
    val_edges: np.ndarray
    test_edges: np.ndarray
    all_edges: np.ndarray
    tag_ids: np.ndarray
    tag_offsets: np.ndarray
    tag_available: np.ndarray
    text_available: np.ndarray
    eval_mashups: np.ndarray
    metadata: dict

    @property
    def num_mashups(self) -> int:
        return int(self.metadata["num_mashups"])

    @property
    def num_apis(self) -> int:
        return int(self.metadata["num_apis"])

    @property
    def num_nodes(self) -> int:
        return self.num_mashups + self.num_apis

    @property
    def num_categories(self) -> int:
        return int(self.metadata["num_categories"])

    @classmethod
    def load(cls, root: str | Path) -> "ProcessedData":
        root = Path(root)
        arrays = np.load(root / "graph.npz")
        with (root / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        return cls(
            root=root,
            train_edges=arrays["train_edges"].astype(np.int64),
            val_edges=arrays["val_edges"].astype(np.int64),
            test_edges=arrays["test_edges"].astype(np.int64),
            all_edges=arrays["all_edges"].astype(np.int64),
            tag_ids=arrays["tag_ids"].astype(np.int64),
            tag_offsets=arrays["tag_offsets"].astype(np.int64),
            tag_available=arrays["tag_available"].astype(bool),
            text_available=arrays["text_available"].astype(bool),
            eval_mashups=arrays["eval_mashups"].astype(np.int64),
            metadata=metadata,
        )


def _read_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    names = ["apibasic", "mashup", "mashupapi", "apicate", "mashupcate", "category"]
    return {name: pd.read_csv(raw_dir / f"{name}.csv") for name in names}


def _constrained_holdout(
    edges: set[tuple[int, int]],
    seed: int,
    min_eval_mashup_degree: int,
    min_target_api_degree: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]], list[int]]:
    mashup_degree: dict[int, int] = {}
    api_degree: dict[int, int] = {}
    for mashup, api in edges:
        mashup_degree[mashup] = mashup_degree.get(mashup, 0) + 1
        api_degree[api] = api_degree.get(api, 0) + 1

    eligible_neighbors: dict[int, list[int]] = {}
    for mashup, api in edges:
        if api_degree[api] >= min_target_api_degree:
            eligible_neighbors.setdefault(mashup, []).append(api)
    eval_mashups = sorted(
        mashup
        for mashup, degree in mashup_degree.items()
        if degree >= min_eval_mashup_degree and len(eligible_neighbors.get(mashup, [])) >= 2
    )

    rng = np.random.default_rng(seed)
    graph = nx.DiGraph()
    source, sink = ("source", -1), ("sink", -1)
    for mashup in eval_mashups:
        m_node = ("m", mashup)
        graph.add_edge(source, m_node, capacity=2, weight=0)
        for api in eligible_neighbors[mashup]:
            graph.add_edge(
                m_node,
                ("a", api),
                capacity=1,
                weight=int(rng.integers(0, 1_000_000)),
            )
    for api, degree in api_degree.items():
        if degree >= min_target_api_degree:
            graph.add_edge(("a", api), sink, capacity=degree - 1, weight=0)

    flow = nx.max_flow_min_cost(graph, source, sink)
    flow_value = sum(flow[source].values())
    required = 2 * len(eval_mashups)
    if flow_value != required:
        raise RuntimeError(f"Warm-start holdout is infeasible: required {required}, got {flow_value}")

    val_edges: set[tuple[int, int]] = set()
    test_edges: set[tuple[int, int]] = set()
    for mashup in eval_mashups:
        chosen = [
            node[1]
            for node, amount in flow[("m", mashup)].items()
            if node[0] == "a" and amount == 1
        ]
        if len(chosen) != 2:
            raise RuntimeError(f"Mashup {mashup} received {len(chosen)} holdout edges")
        rng.shuffle(chosen)
        val_edges.add((mashup, chosen[0]))
        test_edges.add((mashup, chosen[1]))

    train_edges = edges - val_edges - test_edges
    train_api_degree: dict[int, int] = {}
    for _, api in train_edges:
        train_api_degree[api] = train_api_degree.get(api, 0) + 1
    if any(train_api_degree.get(api, 0) == 0 for _, api in val_edges | test_edges):
        raise AssertionError("A validation/test API disappeared from the training graph")
    return train_edges, val_edges, test_edges, eval_mashups


def _edge_array(edges: Iterable[tuple[int, int]], mashup_map: dict[int, int], api_map: dict[int, int]) -> np.ndarray:
    return np.asarray(
        sorted((mashup_map[mashup], api_map[api]) for mashup, api in edges),
        dtype=np.int64,
    ).reshape(-1, 2)


def prepare_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    min_eval_mashup_degree: int = 3,
    min_target_api_degree: int = 3,
) -> dict:
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    tables = _read_tables(raw_dir)
    api_ids = set(int(value) for value in tables["apibasic"]["ID"])
    mashup_ids = set(int(value) for value in tables["mashup"]["ID"])

    raw_edges = tables["mashupapi"][["MashupID", "ApiID"]].drop_duplicates()
    edges = {
        (int(mashup), int(api))
        for mashup, api in raw_edges.itertuples(index=False, name=None)
        if int(api) != -1 and int(mashup) in mashup_ids and int(api) in api_ids
    }
    active_mashups = sorted({mashup for mashup, _ in edges})
    active_apis = sorted({api for _, api in edges})
    mashup_map = {external: index for index, external in enumerate(active_mashups)}
    api_map = {external: index for index, external in enumerate(active_apis)}

    train, val, test, eval_external = _constrained_holdout(
        edges, seed, min_eval_mashup_degree, min_target_api_degree
    )
    all_array = _edge_array(edges, mashup_map, api_map)
    train_array = _edge_array(train, mashup_map, api_map)
    val_array = _edge_array(val, mashup_map, api_map)
    test_array = _edge_array(test, mashup_map, api_map)
    eval_mashups = np.asarray([mashup_map[value] for value in eval_external], dtype=np.int64)

    category_ids = sorted(int(value) for value in tables["category"]["ID"].unique())
    category_map = {external: index for index, external in enumerate(category_ids)}
    node_categories: list[list[int]] = [[] for _ in range(len(active_mashups) + len(active_apis))]
    for mashup, category in tables["mashupcate"][["MashupID", "CateID"]].itertuples(index=False, name=None):
        if int(mashup) in mashup_map and int(category) in category_map:
            node_categories[mashup_map[int(mashup)]].append(category_map[int(category)])
    api_offset = len(active_mashups)
    for api, category in tables["apicate"][["ApiID", "CateID"]].itertuples(index=False, name=None):
        if int(api) in api_map and int(category) in category_map:
            node_categories[api_offset + api_map[int(api)]].append(category_map[int(category)])
    node_categories = [sorted(set(values)) for values in node_categories]
    tag_offsets = [0]
    tag_ids: list[int] = []
    for values in node_categories:
        tag_ids.extend(values)
        tag_offsets.append(len(tag_ids))

    mashup_rows = tables["mashup"].set_index("ID")
    api_rows = tables["apibasic"].set_index("ID")
    texts, node_names, text_available = [], [], []
    for external in active_mashups:
        row = mashup_rows.loc[external]
        name = str(row["Name"])
        description = row["Description"]
        valid_description = pd.notna(description) and bool(str(description).strip())
        texts.append(str(description).strip() if valid_description else name)
        node_names.append(name)
        text_available.append(bool(valid_description))
    for external in active_apis:
        row = api_rows.loc[external]
        name = str(row["Name"])
        description = row["Description"]
        valid_description = pd.notna(description) and bool(str(description).strip())
        texts.append(str(description).strip() if valid_description else name)
        node_names.append(name)
        text_available.append(bool(valid_description))

    destination = output_dir / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "graph.npz",
        train_edges=train_array,
        val_edges=val_array,
        test_edges=test_array,
        all_edges=all_array,
        tag_ids=np.asarray(tag_ids, dtype=np.int64),
        tag_offsets=np.asarray(tag_offsets, dtype=np.int64),
        tag_available=np.asarray([bool(values) for values in node_categories]),
        text_available=np.asarray(text_available, dtype=bool),
        eval_mashups=eval_mashups,
    )
    metadata = {
        "seed": seed,
        "num_mashups": len(active_mashups),
        "num_apis": len(active_apis),
        "num_nodes": len(active_mashups) + len(active_apis),
        "num_categories": len(category_ids),
        "num_all_edges": len(all_array),
        "num_train_edges": len(train_array),
        "num_val_edges": len(val_array),
        "num_test_edges": len(test_array),
        "num_eval_mashups": len(eval_mashups),
        "min_eval_mashup_degree": min_eval_mashup_degree,
        "min_target_api_degree": min_target_api_degree,
    }
    with (destination / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    with (destination / "mappings.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "mashup_external_ids": active_mashups,
                "api_external_ids": active_apis,
                "category_external_ids": category_ids,
                "node_names": node_names,
            },
            handle,
            ensure_ascii=False,
        )
    with (destination / "texts.json").open("w", encoding="utf-8") as handle:
        json.dump(texts, handle, ensure_ascii=False)
    return metadata


def _hash_embeddings(texts: list[str], dimension: int = 384) -> np.ndarray:
    embeddings = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = text.lower().split()
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            embeddings[row, index] += sign
        norm = np.linalg.norm(embeddings[row])
        if norm:
            embeddings[row] /= norm
    return embeddings


def encode_texts(
    processed_dir: str | Path,
    output_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    device: str = "cpu",
    backend: str = "sentence-transformer",
) -> tuple[int, int]:
    processed_dir, output_path = Path(processed_dir), Path(output_path)
    with (processed_dir / "texts.json").open("r", encoding="utf-8") as handle:
        texts = json.load(handle)
    if backend == "hash":
        embeddings = _hash_embeddings(texts)
    elif backend == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
    else:
        raise ValueError(f"Unknown text backend: {backend}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    return embeddings.shape


# Sparse graph operations used by PGRL.
def build_normalized_adjacency(
    num_mashups: int,
    num_apis: int,
    edges: np.ndarray | torch.Tensor,
    device: torch.device | str,
) -> torch.Tensor:
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    num_nodes = num_mashups + num_apis
    if edge_tensor.numel() == 0:
        indices = torch.empty((2, 0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(
            indices, values, (num_nodes, num_nodes), check_invariants=False
        ).coalesce()
    mashups = edge_tensor[:, 0]
    apis = edge_tensor[:, 1] + num_mashups
    source = torch.cat([mashups, apis])
    target = torch.cat([apis, mashups])
    degree = torch.zeros(num_nodes, dtype=torch.float32, device=device)
    degree.scatter_add_(0, source, torch.ones_like(source, dtype=torch.float32))
    degree_inv_sqrt = degree.clamp_min(1.0).pow(-0.5)
    values = degree_inv_sqrt[source] * degree_inv_sqrt[target]
    return torch.sparse_coo_tensor(
        torch.stack([source, target]),
        values,
        (num_nodes, num_nodes),
        device=device,
        check_invariants=False,
    ).coalesce()


def directed_edge_index(
    num_mashups: int, edges: np.ndarray | torch.Tensor, device: torch.device | str
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_tensor = torch.as_tensor(edges, dtype=torch.long, device=device)
    mashups = edge_tensor[:, 0]
    apis = edge_tensor[:, 1] + num_mashups
    return torch.cat([mashups, apis]), torch.cat([apis, mashups])


def lightgcn_propagate(
    initial: torch.Tensor,
    adjacency: torch.Tensor,
    layers: int,
    include_layer0: bool = True,
    noise_epsilon: float = 0.0,
) -> torch.Tensor:
    states = [initial] if include_layer0 else []
    current = initial
    for _ in range(layers):
        current = torch.sparse.mm(adjacency, current)
        if noise_epsilon:
            noise = torch.nn.functional.normalize(torch.rand_like(current), dim=-1)
            current = current + noise_epsilon * torch.sign(current) * noise
        states.append(current)
    if not states:
        return initial
    return torch.stack(states, dim=0).mean(dim=0)


def segment_softmax(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    maxima = torch.full((size,), -torch.inf, dtype=values.dtype, device=values.device)
    maxima.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    exponentials = torch.exp(values - maxima[index])
    denominators = torch.zeros(size, dtype=values.dtype, device=values.device)
    denominators.scatter_add_(0, index, exponentials)
    return exponentials / denominators[index].clamp_min(1e-12)


def segment_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = torch.zeros((size, values.shape[-1]), dtype=values.dtype, device=values.device)
    output.scatter_add_(0, index[:, None].expand_as(values), values)
    return output


# Full-ranking recommendation metrics.
def user_ranking_metrics(
    ranking: Iterable[int], targets: set[int], ks: Iterable[int]
) -> dict[str, float]:
    ranking = list(ranking)
    if not targets:
        raise ValueError("targets must not be empty")
    results: dict[str, float] = {}
    target_ranks = [position + 1 for position, item in enumerate(ranking) if item in targets]
    for k in ks:
        hits = sum(1 for item in ranking[:k] if item in targets)
        precision = hits / k
        recall = hits / len(targets)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        dcg = sum(1.0 / np.log2(rank + 1) for rank in target_ranks if rank <= k)
        ideal_hits = min(len(targets), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        results[f"precision@{k}"] = float(precision)
        results[f"recall@{k}"] = float(recall)
        results[f"f1@{k}"] = float(f1)
        results[f"ndcg@{k}"] = float(dcg / idcg if idcg else 0.0)
    results["mrr"] = float(1.0 / min(target_ranks) if target_ranks else 0.0)
    return results


def macro_average(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
