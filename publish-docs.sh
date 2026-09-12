#!/usr/bin/env bash
#
# Publishes https://sadit.github.io/SimilaritySearchEngine.jl to the gh-pages branch:
#
#   dev/     the Documenter API reference, built here by docs/make.jl
#   manual/  the Quarto manual, taken from the committed manual/docs/
#
# Run it from the repository root after changing anything in src/ (which changes the
# docstrings the reference is generated from) or in manual/ (re-render first, see below).
#
# Why a script and not a GitHub Actions workflow. The reason originally recorded here -- the
# package built only against unreleased local checkouts, so no runner could resolve its
# environment -- expired on 2026-09-09: SimilaritySearch 1.4.1, TextSearch 1.1.2, RocksDB
# 1.0.0 and RocksDB_jll 11.1.2+0 are all in the General registry, and ci.yml resolves and
# tests the package from there with no local checkouts at all.
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
# deploydocs() instead checks out gh-pages and replaces only its own `dev/` subfolder
# (gitrm_copy), leaving `manual/` untouched -- but it does rewrite the branch root's
# index.html with its own stable/dev redirect, which is not the two-halves redirect written
# below. So the two models can each publish this site and neither can share the branch with
# the other: every run of this script erases what the workflow deployed, and every run of
# the workflow replaces the root redirect. Pick one before wiring a docs workflow.
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
    echo "manual/docs/ is not rendered. Run:  XDG_RUNTIME_DIR=\$(mktemp -d) quarto render manual" >&2
    exit 1
}

echo "==> building the API reference"
julia --project=docs docs/make.jl

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

echo "==> assembling the site"
mkdir -p "$STAGE/dev" "$STAGE/manual"
cp -r docs/build/. "$STAGE/dev/"
cp -r manual/docs/. "$STAGE/manual/"
# Without this, Pages runs Jekyll and drops every path component starting with an
# underscore -- which is most of what Documenter and Quarto emit.
touch "$STAGE/.nojekyll"
cat > "$STAGE/index.html" <<'HTML'
<!doctype html>
<meta charset="utf-8">
<title>SimilaritySearchEngine.jl</title>
<link rel="canonical" href="https://sadit.github.io/SimilaritySearchEngine.jl/dev/">
<meta http-equiv="refresh" content="0; url=dev/">
<p>Redirecting to the <a href="dev/">API reference</a>. See also the
<a href="manual/">manual</a> &mdash; architecture and tutorials.</p>
HTML

echo "==> publishing to gh-pages"
# A fresh single-commit branch each time, force-pushed. The published site is entirely
# reproducible from this repository, so its history carries no information worth keeping,
# and an accumulating history of 7 MB snapshots is a clone nobody wants.
cd "$STAGE"
git init -q
git checkout -q --orphan gh-pages 2>/dev/null || true
git add -A
git commit -q -m "Publish the API reference and the pre-rendered manual

Built from $(git -C "$ROOT" rev-parse --short HEAD) by publish-docs.sh."
git remote add origin "$REPO_URL"
git push --force -q origin gh-pages

echo "==> published: https://sadit.github.io/SimilaritySearchEngine.jl/"
