import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from typing import Any
import modules.globals


def find_cluster_centroids(embeddings, max_k: int | None = None, method: str | None = None) -> Any:
    if embeddings is None or len(embeddings) == 0:
        return []
    X = np.array(embeddings)
    n_samples = len(X)
    max_k = max_k or getattr(modules.globals, 'cluster_max_k', 10)
    max_k = max(2, min(max_k, n_samples))
    method = method or getattr(modules.globals, 'cluster_method', 'elbow')

    candidates = list(range(2, max_k + 1))
    models = []
    inertias = []
    silhouettes = []

    for k in candidates:
        km = KMeans(n_clusters=k, random_state=0)
        km.fit(X)
        models.append(km)
        inertias.append(km.inertia_)
        try:
            silhouettes.append(silhouette_score(X, km.labels_))
        except Exception:
            silhouettes.append(-1.0)

    chosen_idx = 0
    if method == 'silhouette' and any(s > -1.0 for s in silhouettes):
        chosen_idx = int(np.argmax(silhouettes))
    else:
        # Elbow: choose largest drop in inertia
        diffs = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)] if len(inertias) > 1 else [0]
        chosen_idx = int(np.argmax(diffs)) + 1 if len(diffs) > 0 else 0

    return models[chosen_idx].cluster_centers_

def find_closest_centroid(centroids: list, normed_face_embedding, min_similarity: float | None = None) -> list:
    try:
        centroids = np.array(centroids)
        normed_face_embedding = np.array(normed_face_embedding)
        similarities = np.dot(centroids, normed_face_embedding)
        closest_centroid_index = int(np.argmax(similarities))
        max_sim = float(similarities[closest_centroid_index]) if similarities.size else -1.0
        if min_similarity is not None and max_sim < min_similarity:
            return None, None
        return closest_centroid_index, centroids[closest_centroid_index]
    except ValueError:
        return None