# Tutorial Datasets and Preparation Pipeline

The preparation script `prepare.jl` builds the experimental datasets and linguistic models referenced in the [Multilingual Paragraph Search Tutorial](../manual/tutorials/paragraph-search.qmd):

| Corpus Identifier | Source Collection | Language | Document Count | Approximate Paragraph Count |
| :--- | :--- | :--- | ---:| ---:|
| `gutenberg-en` | Project Gutenberg | English | 100 books | ~200,000 |
| `gutenberg-es` | Project Gutenberg | Spanish | 100 books | ~200,000 |
| `gutenberg-pt` | Project Gutenberg | Portuguese | 100 books | ~200,000 |
| `wikipedia-en` | Wikipedia (2023-11-01 dump) | English | 1,000 articles | ~60,000 |

Paragraphs serve as the fundamental unit of retrieval and linguistic profile calibration throughout the benchmark suite.

---

## Execution Instructions

Execute `prepare.jl` from the repository root:

```sh
# Full benchmark dataset preparation (~1 hour execution time)
julia --project=tutorials tutorials/prepare.jl

# Lightweight smoke-test build (~5 minutes execution time)
julia --project=tutorials tutorials/prepare.jl --smoke

# Target specific pipeline stages
julia --project=tutorials tutorials/prepare.jl corpora
julia --project=tutorials tutorials/prepare.jl profiles projects

# Force re-execution of previously completed stages
julia --project=tutorials tutorials/prepare.jl --force
```

### Pipeline Stages

1. **`corpora`**: Downloads raw texts and segments documents into paragraph-delimited JSONL files under `data/`.
2. **`profiles`**: Executes `textsearch fit` to construct `TextProfile` models with LSI embeddings and query-expansion networks under `profiles/`.
3. **`projects`**: Indexes paragraph corpora into persistent `SimilaritySearchEngine` project databases under `projects/`.

Each stage caches intermediate artifacts to disk, allowing interrupted runs to resume without re-downloading or re-fitting data.

---

## Environmental Dependencies

- **Local `TextSearch.jl` Repository**: By default located at `../../TextSearch.jl` (or specified via the `TEXTSEARCH_REPO` environment variable).
- **Network Access**: HTTP access to `gutendex.com` and `gutenberg.org` for initial corpus retrieval. Downloaded texts are cached locally under `data/raw/`.

---

## Rendering the Quarto Manual

To render the complete manual locally:

```sh
XDG_RUNTIME_DIR=$(mktemp -d) quarto render manual
```
