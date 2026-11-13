# hnswlib-python

Pure-Python port of the core `HierarchicalNSW` implementation from
[`nmslib/hnswlib`](https://github.com/nmslib/hnswlib).  It mirrors the C++
algorithm and the Python binding API used by table-union-search experiments:

```python
from python_bindings import Index

index = Index(space="cosine", dim=768)
index.init_index(max_elements=len(vectors), M=32, ef_construction=40,
                 random_seed=100, level_rng="mt19937",
                 build_backend="numpy")
index.set_ef(10)
index.add_items(vectors, ids=None)
labels, distances = index.knn_query(queries, k=100)
```

`build_backend="numpy"` accelerates only offline index construction by
batching distances with NumPy; the query path remains pure Python.  The default
is `"python"` for a dependency-free build.

## Alignment with upstream

The port follows `hnswlib/hnswalg.h`:

- `M`, `maxM = M`, `maxM0 = 2*M`; `ef_construction`; `ef`.
- Random level: `int(-log(U) * 1 / log(M))`, with
  `U ~ std::uniform_real_distribution<double>(0, 1)`.
- Greedy descent on layers above `level`, `ef` search on layer 0.
- `getNeighborsByHeuristic2` diversity selection for new links and for
  shrinking an existing link list.
- Bidirectional links; layer-0 degree limit `2*M`, upper layers `M`.
- `searchBaseLayerST`-style stopping rule: `candidate_dist > lowerBound` with a
  top-candidate queue of size `ef`.

`level_rng` chooses the C++ `std::default_random_engine` implementation:
`"mt19937"` (MSVC, the default here) or `"minstd_rand0"` (libstdc++).  Both
reproduce the generator used by the corresponding C++ standard library; the
level distribution is therefore aligned with the library build you compare
against.

## File layout

The implementation mirrors the upstream implementation tree one-for-one:

| upstream | this port |
|---|---|
| `hnswlib/bruteforce.h` | `hnswlib_python/bruteforce.py` |
| `hnswlib/hnswalg.h` | `hnswlib_python/hnswalg.py` |
| `hnswlib/hnswlib.h` | `hnswlib_python/hnswlib.py` |
| `hnswlib/space_ip.h` | `hnswlib_python/space_ip.py` |
| `hnswlib/space_l2.h` | `hnswlib_python/space_l2.py` |
| `hnswlib/stop_condition.h` | `hnswlib_python/stop_condition.py` |
| `hnswlib/visited_list_pool.h` | `hnswlib_python/visited_list_pool.py` |
| `python_bindings/{__init__,LazyIndex,bindings.cpp,setup}` | `python_bindings/{__init__,LazyIndex,bindings,setup}.py` |
| 11 repository-root files | 11 repository-root files |

The seven core modules and four binding files keep the upstream names and
roles; `.h`/`.cpp` become `.py`.  C++-only build/example directories are not
ported.

## Scope and limitations

- Single-threaded; `set_num_threads` is a no-op.
- Cosine, inner-product, and L2 distances; cosine/IP vectors are normalized on
  insertion and query, as in the upstream cosine space.
- No deletions, filters, updates, or replacement of deleted elements.
- `save_index`/`load_index` use a pickle of the graph and vectors; this is not
  the upstream binary index format.
- Pure Python is much slower than the C++ extension; use it for fair
  same-language comparisons, not for production indexing.

## Tests

```bash
python -m pytest
```
