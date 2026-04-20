# oceanbase-haystack

Haystack 2.x integration for [OceanBase](https://www.oceanbase.com/) vector search. It mirrors the API surface of [milvus-haystack](https://github.com/milvus-io/milvus-haystack) while using [pyobvector](https://github.com/oceanbase/pyobvector) (`ObVecClient`) for SQLAlchemy-style access to OceanBase `VECTOR` indexes.

## Requirements

- Python 3.9+
- A running OceanBase instance with vector support (see OceanBase documentation for version and vector features).
- Dependencies: `haystack-ai`, `pyobvector`, `sqlalchemy` (declared in `pyproject.toml`).

## Installation

```bash
pip install oceanbase-haystack
```

Or from a checkout (in this monorepo, use the `oceanbase-haystack` directory):

```bash
cd oceanbase-haystack
pip install -e .
```

## Quick start

```python
from haystack import Document
from oceanbase_haystack import OceanBaseDocumentStore, OceanBaseEmbeddingRetriever

store = OceanBaseDocumentStore(
    collection_name="HaystackCollection",  # table name in OceanBase
    connection_args={
        "host": "127.0.0.1",
        "port": "2881",
        "user": "root@test",
        "password": "",
        "db_name": "test",
    },
    index_params={"metric_type": "L2", "index_type": "HNSW", "params": {}},
    drop_old=True,
)

store.write_documents(
    [
        Document(content="hello", embedding=[0.1] * 128, meta={"source": "demo"}),
    ]
)

retriever = OceanBaseEmbeddingRetriever(document_store=store, top_k=5)
result = retriever.run(query_embedding=[0.1] * 128)
print(result["documents"])
```

## Connection arguments

Unlike Milvus URI-style `connection_args`, this integration expects a flat dict:

| Key        | Description                          |
|-----------|---------------------------------------|
| `host`    | OceanBase host                        |
| `port`    | Port (often `2881`)                   |
| `user`    | Username (e.g. `root@tenant`)         |
| `password`| Password                              |
| `db_name` | Database name                         |

Use Haystack `Secret` for passwords in production and serialize with `to_dict` / `from_dict` as usual.

## Document layout

- **Table name**: `collection_name` maps to a MySQL/OceanBase **table** name (same parameter name as milvus-haystack for familiarity).
- **Columns**: Primary key (`id`), dense vector (`vector`), text (`text`), and JSON metadata (`meta`). Optional `sparse_vector_field` adds a sparse vector column when enabled.
- **Filters**: Haystack nested filters are translated to SQL predicates on the JSON `meta` column (e.g. `meta.type` → `JSON_EXTRACT(meta, '$.type')`).

## Components

| Class | Role |
|-------|------|
| `OceanBaseDocumentStore` | Implements Haystack document store operations: write, delete, count, filter, dense/sparse/hybrid retrieval. |
| `OceanBaseEmbeddingRetriever` | Dense vector retrieval (same idea as `MilvusEmbeddingRetriever`). |
| `OceanBaseSparseEmbeddingRetriever` | Sparse vector ANN search when `sparse_vector_field` is configured. |
| `OceanBaseHybridRetriever` | Combines dense and sparse results with **Reciprocal Rank Fusion (RRF)** (not pymilvus `RRFRanker`). |

## Differences from milvus-haystack

- **No Milvus built-in BM25**: `builtin_function` is not supported; use model-produced `SparseEmbedding` for sparse search.
- **Hybrid fusion**: Uses RRF in the client layer instead of Milvus hybrid + `RRFRanker`.
- **Metadata**: Stored in a single JSON column rather than Milvus dynamic scalar fields.

## Configuration tips

- **`embedding_dim`**: If set at construction time and the table does not exist, the store can create the table before the first write.
- **`index_params`**: Milvus-style keys (`metric_type`, `index_type`, `params`) are mapped to OceanBase vector index settings where possible.
- **`search_params`**: e.g. `{"efSearch": 64}` for HNSW search-time behavior (passed through to pyobvector where applicable).

## Development

```bash
pip install -e .
pip install pytest
pytest tests/
```

## License

Apache-2.0 (see `pyproject.toml`).
