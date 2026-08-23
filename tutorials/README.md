# Tutorial corpora and projects

`prepare.jl` builds everything the [paragraph-search
tutorial](../manual/tutorials/paragraph-search.qmd) reads: four paragraph corpora, a fitted
[TextSearch](https://github.com/sadit/TextSearch.jl) profile for each, and a
`SimilaritySearchEngine` project for each.

| corpus | source | language | documents | paragraphs (approx.) |
|---|---|---|---:|---:|
| `gutenberg-en` | Project Gutenberg | English | 100 books | ~200k |
| `gutenberg-es` | Project Gutenberg | Spanish | 100 books | ~200k |
| `gutenberg-pt` | Project Gutenberg | Portuguese | 100 books | ~200k |
| `wikipedia-en` | Wikipedia 2023-11-01 | English | 1000 articles | ~60k |

Paragraphs are the retrieval unit throughout: a search answers with the paragraph that contains
the answer, not with the book or article around it. That is also the unit the profiles are
fitted on, so what a profile calls a document and what a project indexes are the same thing.

## Running it

```sh
# from the repository root
julia --project=tutorials tutorials/prepare.jl              # the documented scale
julia --project=tutorials tutorials/prepare.jl --smoke      # a few books and articles
julia --project=tutorials tutorials/prepare.jl corpora      # just one step
julia --project=tutorials tutorials/prepare.jl profiles projects
julia --project=tutorials tutorials/prepare.jl --force      # redo work already done
```

Three steps, in order: `corpora` (download and split), `profiles` (`textsearch fit`), `projects`
(index into the engine). Each is separately runnable and caches to disk, so a run that fails
halfway resumes instead of restarting — a book already downloaded is not fetched again, a
profile already fitted is not refitted. `--smoke` writes to its own filenames, so a smoke run
and a full run can coexist.

Expect roughly an hour at the documented scale, most of it in `profiles` (each fit runs an LSI
over the corpus) and `projects`. Everything lands under `data/`, `profiles/` and `projects/`,
all three gitignored: the recipe is versioned, the output is not.

## What it needs

**A local TextSearch checkout**, by default at `../../TextSearch.jl`, overridable with
`TEXTSEARCH_REPO`. Three things come from it and none is in the released package:

- `apps/textsearch` — the CLI that fits profiles. Its environment must be instantiated:
  `julia --project=$TEXTSEARCH_REPO/apps/textsearch -e 'using Pkg; Pkg.instantiate()'`
- `corpus-profiles/lib/parquet_to_jsonl.jl` — the parquet→JSONL converter, reused rather than
  reimplemented so a paragraph here is split exactly the way a paragraph in TextSearch's own
  published profiles was.
- `corpus-profiles/raw/wikipedia/20231101.en/*.parquet` — the Wikipedia dump TextSearch already
  downloaded to fit those profiles. See `corpus-profiles/corpora/wikipedia.sh` if you don't have
  it.

**Network access** to `gutendex.com` (the catalogue) and `gutenberg.org` (the books). Downloads
are cached under `data/raw/`, so this is a first-run cost only.

`prepare.jl` checks all of this before doing any work, and prints every problem it finds at
once rather than one per run.

## Rendering the tutorial

```sh
XDG_RUNTIME_DIR=$(mktemp -d) quarto render manual
```

The page executes its own Julia cells, so its numbers and search results are real. The
`XDG_RUNTIME_DIR` is not optional on a machine where `/run/user/$UID` is not writable: Quarto's
Julia engine fails there with `PermissionDenied … mkdir '/run/user/…/julia'` before running a
single cell.
