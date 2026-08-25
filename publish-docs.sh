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
# Why a script and not a GitHub Actions workflow: this package builds only against
# unreleased local checkouts of its dependencies. SimilaritySearch 1.2.0 and TextSearch
# 1.1.0 are what the code requires and the registry has 1.1.2 and 1.0.0, so a runner
# cannot resolve the environment at all. A workflow becomes possible once those are
# registered -- or, at minimum, once TextSearch 1.1.0 is pushed, since it is the one the
# runner could otherwise clone.
#
# The manual is not rendered here either, and that one is not a packaging problem: its
# paragraph-search tutorial executes Julia against corpora that are gitignored (523k
# paragraphs, an hour and a network connection). Re-render it yourself when its pages
# change, and commit the result:
#
#     XDG_RUNTIME_DIR=$(mktemp -d) quarto render manual
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
