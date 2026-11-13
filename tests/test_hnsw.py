import math
import random

from python_bindings import Index


def _unit(vec):
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def test_hnsw_recall_against_bruteforce():
    rng = random.Random(0)
    n, dim, k = 300, 32, 10
    data = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    data = [_unit(vec) for vec in data]
    queries = data[:20]

    index = Index(space="cosine", dim=dim)
    index.init_index(max_elements=n, M=16, ef_construction=40, random_seed=100)
    index.set_ef(50)
    index.add_items(data)
    labels, _ = index.knn_query(queries, k=k)

    hits = 0
    for query, found in zip(queries, labels):
        expected = sorted(
            range(n),
            key=lambda i: -sum(a * b for a, b in zip(query, data[i])),
        )[:k]
        hits += len(set(expected) & set(found))
    recall = hits / (len(queries) * k)
    assert recall >= 0.9, recall


def test_save_load(tmp_path):
    index = Index(space="cosine", dim=8)
    index.init_index(max_elements=10, M=4, ef_construction=20, random_seed=1)
    index.add_items([[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6])
    path = tmp_path / "index.pkl"
    index.save_index(str(path))

    loaded = Index(space="cosine", dim=8)
    loaded.load_index(str(path))
    labels, distances = loaded.knn_query([[1.0] + [0.0] * 7], k=1)
    assert labels[0][0] == 0
    assert distances[0][0] < 1e-5
