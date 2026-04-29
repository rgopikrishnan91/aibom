#!/usr/bin/env bash
# Push the current branch to a HuggingFace Space, swapping in the HF-flavored
# README so the Space picks up the required YAML frontmatter.
#
# Prerequisites:
#   - `git remote add hf https://huggingface.co/spaces/<you>/aibom`
#   - HF access token with write permission, configured for git auth.
#
# Usage: bash scripts/deploy_to_hf_spaces.sh

set -euo pipefail

if ! git remote get-url hf >/dev/null 2>&1; then
  cat <<EOF
Error: no 'hf' git remote found.

Add it first:
  git remote add hf https://huggingface.co/spaces/<your-username>/aibom
EOF
  exit 1
fi

if [ ! -f README_HF.md ]; then
  echo "Error: README_HF.md not found at repo root."
  exit 1
fi

# We want to push a tree where README.md == README_HF.md, but without
# polluting the working copy. Use a throwaway branch.
DEPLOY_BRANCH="hf-deploy-$(date +%s)"
ORIGINAL_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "Creating temporary deploy branch: ${DEPLOY_BRANCH} (from ${ORIGINAL_BRANCH})"

git checkout -b "${DEPLOY_BRANCH}"

# Swap the README content
cp README.md README.md.bak
cp README_HF.md README.md
git add README.md
git commit -m "deploy: use HF Spaces README" --no-verify

# Push to the Space's main branch (force, since HF Space history may diverge)
echo "Pushing to hf main..."
git push hf "${DEPLOY_BRANCH}:main" --force

# Restore working copy
git checkout "${ORIGINAL_BRANCH}"
mv README.md.bak README.md
git branch -D "${DEPLOY_BRANCH}"

echo
echo "Deployed. Build progress at the Space URL (Settings -> Logs)."
