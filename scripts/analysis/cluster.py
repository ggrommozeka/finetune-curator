"""Cluster annotations with embeddings or a TF-IDF fallback. Tiny clusters are noise."""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from collections import Counter

from scripts.analysis.annotate import annotation_text
from scripts.analysis.store import PatternStore

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text or "") if len(token) > 1]


def tfidf_vectors(docs: list[str]) -> list[dict[str, float]]:
    counts = [Counter(tokenize(doc)) for doc in docs]
    df: Counter[str] = Counter()
    for count in counts:
        df.update(count.keys())
    total_docs = max(1, len(docs))
    vectors = []
    for count in counts:
        length = sum(count.values()) or 1
        vectors.append(
            {
                term: (seen / length) * (math.log((1 + total_docs) / (1 + df[term])) + 1)
                for term, seen in count.items()
            }
        )
    return vectors


def cosine(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[term] * right[term] for term in left.keys() & right.keys())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def embed_texts(texts: list[str], *, base_url: str, api_key: str | None, model: str, timeout: float = 60) -> list[list[float]]:
    url = base_url.rstrip("/") + "/embeddings"
    payload = json.dumps({"model": model, "input": texts}).encode()
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode())
    rows = sorted(body.get("data") or [], key=lambda item: item.get("index") or 0)
    return [row["embedding"] for row in rows]


def _as_sparse(vector: list[float]) -> dict[str, float]:
    return {str(index): value for index, value in enumerate(vector)}


def greedy_cluster(vectors: list[dict[str, float]], threshold: float) -> list[list[int]]:
    groups: list[list[int]] = []
    centroids: list[dict[str, float]] = []
    for index, vector in enumerate(vectors):
        best = -1.0
        best_at = -1
        for group_index, centroid in enumerate(centroids):
            score = cosine(vector, centroid)
            if score > best:
                best = score
                best_at = group_index
        if best_at >= 0 and best >= threshold:
            groups[best_at].append(index)
        else:
            groups.append([index])
            centroids.append(vector)
    return groups


def _dominant(values: list[str]) -> str:
    cleaned = [value for value in values if value]
    if not cleaned:
        return "mixed"
    return Counter(cleaned).most_common(1)[0][0]


def cluster_annotations(
    store: PatternStore,
    *,
    min_size: int = 3,
    threshold: float = 0.34,
    embeddings: list[list[float]] | None = None,
) -> dict:
    rows = store.annotated_rows()
    if not rows:
        store.replace_clusters([])
        return {"clusters": 0, "noise": 0}
    texts = [annotation_text(row["annotation"]) for row in rows]
    if embeddings is not None:
        if len(embeddings) != len(rows):
            raise ValueError("embeddings length does not match annotated sessions")
        vectors = [_as_sparse(vector) for vector in embeddings]
    else:
        vectors = tfidf_vectors(texts)
    groups = greedy_cluster(vectors, threshold)
    clusters = []
    noise = 0
    for index, members in enumerate(groups):
        chosen = [rows[member] for member in members]
        annotations = [row["annotation"] for row in chosen]
        machines = sorted({label for row in chosen for label in row.get("sources") or []})
        projects = sorted({(row.get("project") or ann.get("project") or "") for row, ann in zip(chosen, annotations) if (row.get("project") or ann.get("project"))})
        task_types = sorted({ann.get("task_type") or "other" for ann in annotations})
        fails = sum(1 for ann in annotations if ann.get("outcome") == "fail")
        fail_rate = fails / len(chosen)
        is_noise = len(chosen) < min_size
        if is_noise:
            noise += 1
        error_type = _dominant([ann.get("error_type") or "" for ann in annotations])
        clusters.append(
            {
                "id": "c_" + hashlib.sha1(error_type.encode()).hexdigest()[:8] + f"_{index}",
                "label": error_type,
                "error_type": error_type,
                "size": len(chosen),
                "frequency": len(chosen),
                "task_types": len(task_types),
                "task_type_names": task_types,
                "projects": len(projects),
                "project_names": projects,
                "machines": len(machines),
                "machine_names": machines,
                "fail_rate": round(fail_rate, 3),
                "noise": is_noise,
                "session_ids": [row["global_id"] for row in chosen],
                "outcomes": [ann.get("outcome") for ann in annotations],
            }
        )
    store.replace_clusters(clusters)
    return {"clusters": len(clusters) - noise, "noise": noise}
