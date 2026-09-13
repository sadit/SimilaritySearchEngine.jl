#!/usr/bin/env bash
#
# Publishes https://sadit.github.io/SimilaritySearchEngine.jl to the gh-pages branch:
#
#   /       the Quarto manual, taken from the committed manual/docs/ -- the prose for BOTH
#           packages: the engine's architecture and tutorials, and the server's manual and
#           tutorial, in one navigation with one search index
#   /api/   the Documenter reference, built here by docs/make.jl from the docstrings of both
#           packages: Engine API and Server API
#
# One repository, one Pages site. GitHub serves a single site per repository, so the two
# packages share it; laying them out as one manual and one reference, rather than as two
# documentation sets on the same domain, is what makes that a feature. The engine's pages own
# the root because that is where a reader arrives.
#
# Run it from the repository root after changing anything in src/ or server/src/ (which
# changes the docstrings the reference is generated from) or in manual/ (re-render first, see
# below).
#
# Why a script and not a GitHub Actions workflow. The reason originally recorded here -- the
# package built only against unreleased local checkouts, so no runner could resolve its
# environment -- expired on 2026-09-09: SimilaritySearch 1.4.1, TextSearch 1.1.2, RocksDB
# 1.0.0 and RocksDB_jll 11.1.2+0 are all in the General registry, and ci.yml resolves and
# tests both packages from there with no local checkouts at all. docs/Project.toml carries a
# `[sources]` block pointing at this repository's two packages by path, so a runner can build
# the reference from a plain checkout too.
#
# The manual's data dependency is a separate matter, and -- despite the obvious guess -- it
# is not what keeps publishing manual. The rendered manual is committed under manual/docs/
# (see .gitignore) precisely so that publishing only has to copy it; a runner can do that
# from a plain checkout. What the data dependency blocks is RE-RENDERING one page:
# manual/tutorials/paragraph-search.qmd overrides the project's `execute: enabled: false`
# with its own `engine: julia` + `execute: enabled: true`, and runs against the gitignored
# corpora that tutorials/prepare.jl builds (~500k paragraphs, an hour, a network
# connection). That is a step a human takes when that page changes, not a step publishing
# takes.
#
# So nothing blocks a workflow any more; what is left is a decision nobody has made. This
# script publishes both halves as ONE orphan commit, force-pushed. Documenter's own
# deploydocs() would fight it -- it rewrites the branch root's index.html with a stable/dev
# redirect, and the branch root is where the manual's landing page lives now -- which is why
# docs/make.jl no longer calls it. Wire a workflow if you like, but pick one publishing model.
#
# Re-rendering the manual, when its pages change -- ONE PAGE AT A TIME:
#
#     XDG_RUNTIME_DIR=$(mktemp -d) quarto render manual/index.qmd
#
# Not `quarto render manual`: rendering the project empties manual/docs/ before it starts and
# then dies on paragraph-search.qmd's julia engine, leaving the published site half deleted.
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/sadit/SimilaritySearchEngine.jl.git}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f manual/docs/index.html ]] || {
    echo "manual/docs/ is not rendered. Run:  XDG_RUNTIME_DIR=\$(mktemp -d) quarto render manual/index.qmd" >&2
    exit 1
}
[[ -f manual/docs/server/manual.html ]] || {
    echo "manual/docs/server/ is missing: the server's pages have never been rendered." >&2
    exit 1
}

echo "==> building the API reference (both packages)"
julia --project=docs docs/make.jl

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> assembling the site"
cp -r manual/docs/. "$STAGE/"
mkdir -p "$STAGE/api"
cp -r docs/build/. "$STAGE/api/"
# Without this, Pages runs Jekyll and drops every path component starting with an
# underscore -- which is most of what Documenter and Quarto emit.
touch "$STAGE/.nojekyll"

echo "==> publishing to gh-pages"
# A fresh single-commit branch each time, force-pushed. The published site is entirely
# reproducible from this repository, so its history carries no information worth keeping,
# and an accumulating history of 7 MB snapshots is a clone nobody wants.
cd "$STAGE"
git init -q
git checkout -q --orphan gh-pages 2>/dev/null || true
git add -A
git commit -q -m "Publish the manual and the API reference

Built from $(git -C "$ROOT" rev-parse --short HEAD) by publish-docs.sh."
git remote add origin "$REPO_URL"
git push --force -q origin gh-pages

echo "==> published: https://sadit.github.io/SimilaritySearchEngine.jl/"
